#!/usr/bin/env python3
"""第1・第2層を枝刈りし、同じ行列からCOO版・CSR版のヘッダーを作る。

例（閾値0.01は使用例であり、推奨値・最適値ではない）:
  python3 compress_weights.py 学習/70_70/model_parameters.h --threshold 0.01
  python3 compress_weights.py 学習/70_70/model_parameters.h \
      --threshold1 0.005 --threshold2 0.01 --output-dir 学習/70_70/compressed_test1

出力: coo/model_parameters.h, csr/model_parameters.h, compression_report.json,
      README.md。元ファイルは変更せず、既存の出力フォルダへは上書きしない。
圧縮対象はweight_1・weight_2。weight_3と全バイアスは変更しない。
標準ライブラリのみ使用。生成ヘッダーには、別途疎形式のpredict()が必要。
"""

import argparse
import hashlib
import json
import math
import re
import struct
from datetime import datetime, timezone
from pathlib import Path


ARRAY_NAMES = ('weight_1', 'bias_1', 'weight_2', 'bias_2', 'weight_3', 'bias_3')
NUMBER = re.compile(r'[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[fF]?\Z')


def float32(value):
    """Arduinoのfloat配列に格納された値としてfloat32へ丸める。"""
    try:
        result = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    except (OverflowError, ValueError) as exc:
        raise ValueError('float32で表せない重みがあります') from exc
    if not math.isfinite(result):
        raise ValueError('NaN・無限大の重みは変換できません')
    return result


def parse_header(text):
    """生成済みのfloat配列のみ読む。C++コードや式は実行しない。"""
    clean = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    pattern = r'\bfloat\s+(weight_[123]|bias_[123])\s*\[\s*(\d*)\s*\]\s*=\s*\{([^{}]*)\}\s*;'
    arrays = {}
    for match in re.finditer(pattern, clean):
        name, declared, body = match.groups()
        if name in arrays:
            raise ValueError(f'{name}が重複しています')
        tokens = body.split(',')
        if tokens and not tokens[-1].strip():
            tokens.pop()  # C++で有効な末尾のカンマ。
        values = []
        for token in tokens:
            literal = token.strip()
            if not NUMBER.fullmatch(literal):
                raise ValueError(f'{name}: 数値以外の要素は扱えません: {literal[:60]!r}')
            values.append(float32(literal.rstrip('fF')))
        if declared and int(declared) != len(values):
            raise ValueError(f'{name}: 宣言された長さと初期化値の数が不一致です')
        arrays[name] = values
    if set(arrays) != set(ARRAY_NAMES):
        missing = sorted(set(ARRAY_NAMES) - set(arrays))
        raise ValueError('6つの通常配列を持つ非圧縮ヘッダーが必要です。不足: ' + ', '.join(missing))
    h1, h2, output_size = (len(arrays[name]) for name in ('bias_1', 'bias_2', 'bias_3'))
    if not h1 or not h2 or output_size != 3:
        raise ValueError('隠れ層は1以上、出力はRGBの3次元である必要があります')
    expected = {'weight_1': h1 * 3, 'weight_2': h2 * h1, 'weight_3': 3 * h2}
    for name, length in expected.items():
        if len(arrays[name]) != length:
            raise ValueError(f'{name}: {length}要素が必要ですが{len(arrays[name])}要素です')
    return arrays, (3, h1, h2, 3)


def index_type(maximum):
    for limit, name, size in ((255, 'uint8_t', 1), (65535, 'uint16_t', 2),
                              (4294967295, 'uint32_t', 4)):
        if 0 <= maximum <= limit:
            return name, size
    raise ValueError('索引がuint32_tの範囲を超えています')


def sparsify(weights, rows, cols, threshold):
    """|w| < threshold の場合だけ0にする。列方向は元と同じ昇順。"""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('閾値は0以上の有限値にしてください')
    pruned = [0.0 if abs(value) < threshold else value for value in weights]
    values, row_indices, col_indices, row_ptr = [], [], [], [0]
    for row in range(rows):
        for col in range(cols):
            value = pruned[row * cols + col]
            if value != 0.0:
                values.append(value)
                row_indices.append(row)
                col_indices.append(col)
        row_ptr.append(len(values))
    # PC上で両形式を復元し、枝刈り後の行列と一致することを確認。
    coo_restored, csr_restored = [0.0] * len(weights), [0.0] * len(weights)
    for value, row, col in zip(values, row_indices, col_indices):
        coo_restored[row * cols + col] = value
    for row in range(rows):
        for pos in range(row_ptr[row], row_ptr[row + 1]):
            csr_restored[row * cols + col_indices[pos]] = values[pos]
    if coo_restored != pruned or csr_restored != pruned:
        raise ValueError('疎形式からの復元確認に失敗しました')
    return {'rows': rows, 'cols': cols, 'threshold': threshold, 'values': values,
            'row_indices': row_indices, 'col_indices': col_indices, 'row_ptr': row_ptr,
            'original_nonzero': sum(value != 0.0 for value in weights),
            'pruned_nonzero': sum(old != 0.0 and new == 0.0 for old, new in zip(weights, pruned))}


def cpp_float(value):
    result = format(value, '.9g')
    if '.' not in result and 'e' not in result.lower():
        result += '.0'
    # ヘッダーの10進表記でfloat32の値が変わらないことも確認。
    if struct.pack('<f', float32(result)) != struct.pack('<f', value):
        raise ValueError('float32のヘッダー出力確認に失敗しました')
    return result + 'f'


def emit_array(name, dtype, values, item_bytes):
    """NNZ=0でも標準C++で有効になるよう1要素確保し、その容量も数える。"""
    stored = values if values else [0.0 if dtype == 'float' else 0]
    literals = [cpp_float(v) if dtype == 'float' else str(v) for v in stored]
    lines = [f'{dtype} {name}[] = {{']
    for start in range(0, len(literals), 8):
        lines.append('    ' + ', '.join(literals[start:start + 8]) + ',')
    lines.append('};')
    info = {'type': dtype, 'logical_elements': len(values), 'stored_elements': len(stored),
            'element_bytes': item_bytes, 'bytes': len(stored) * item_bytes}
    return '\n'.join(lines), info


def make_header(fmt, arrays, shape, layers, source_sha):
    dense_bytes = sum(len(values) * 4 for values in arrays.values())
    lines = ['#pragma once', '#include <stdint.h>', '#include <stddef.h>',
             '// First and second weight matrices are sparse. Biases and weight_3 are unchanged.',
             '// Requires a COO/CSR predict(); do not use the original dense predict().',
             f'#define MODEL_STORAGE_{fmt.upper()} 1',
             'static_assert(sizeof(float) == 4, "4-byte float is required");']
    for label, size in zip(('INPUT_SIZE', 'HIDDEN_SIZE1', 'HIDDEN_SIZE2', 'OUTPUT_SIZE'), shape):
        lines.append(f'constexpr uint32_t MODEL_{label} = {size};')
    lines.append(f'constexpr char MODEL_SOURCE_SHA256[] = "{source_sha}";')
    info = {}
    for number, layer in enumerate(layers, 1):
        prefix = f'weight_{number}'
        lines += [f'constexpr double MODEL_THRESHOLD_{number} = {repr(layer["threshold"])};',
                  f'constexpr uint32_t WEIGHT_{number}_NNZ = {len(layer["values"])};']
        descriptors = [(prefix + '_values', 'float', layer['values'], 4)]
        if fmt == 'coo':
            dtype, size = index_type(layer['rows'] - 1)
            descriptors.append((prefix + '_rows', dtype, layer['row_indices'], size))
        dtype, size = index_type(layer['cols'] - 1)
        descriptors.append((prefix + '_cols', dtype, layer['col_indices'], size))
        if fmt == 'csr':
            dtype, size = index_type(len(layer['values']))
            descriptors.append((prefix + '_row_ptr', dtype, layer['row_ptr'], size))
        descriptors.append((f'bias_{number}', 'float', arrays[f'bias_{number}'], 4))
        for name, dtype, values, size in descriptors:
            code, detail = emit_array(name, dtype, values, size)
            lines.append(code)
            info[name] = detail
    for name in ('weight_3', 'bias_3'):
        code, detail = emit_array(name, 'float', arrays[name], 4)
        lines.append(code)
        info[name] = detail
    total_bytes = sum(entry['bytes'] for entry in info.values())
    sizeof_expression = ' + '.join(f'sizeof({name})' for name in info)
    lines += [f'constexpr size_t MODEL_PARAMETER_BYTES = {sizeof_expression};',
              f'constexpr size_t MODEL_DENSE_PARAMETER_BYTES = {dense_bytes};',
              f'static_assert(MODEL_PARAMETER_BYTES == {total_bytes}, "Parameter byte count mismatch");',
              'constexpr double MODEL_COMPRESSION_RATIO_PERCENT =',
              '    100.0 * MODEL_PARAMETER_BYTES / MODEL_DENSE_PARAMETER_BYTES;']
    return '\n\n'.join(lines) + '\n', {'parameter_bytes': total_bytes,
            'compression_ratio_percent': 100.0 * total_bytes / dense_bytes, 'arrays': info}


def convert(source, output_dir, threshold1, threshold2):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f'出力先が既に存在します。別のフォルダ名を指定してください: {output_dir}')
    raw = source.read_bytes()
    arrays, shape = parse_header(raw.decode('utf-8-sig'))
    source_sha = hashlib.sha256(raw).hexdigest()
    layers = [sparsify(arrays['weight_1'], shape[1], shape[0], threshold1),
              sparsify(arrays['weight_2'], shape[2], shape[1], threshold2)]
    headers, formats = {}, {}
    for fmt in ('coo', 'csr'):
        headers[fmt], formats[fmt] = make_header(fmt, arrays, shape, layers, source_sha)
    layer_reports = []
    for layer in layers:
        count = layer['rows'] * layer['cols']
        layer_reports.append({'rows': layer['rows'], 'cols': layer['cols'],
                              'threshold': layer['threshold'], 'total_elements': count,
                              'original_nonzero_elements': layer['original_nonzero'],
                              'nonzero_elements': len(layer['values']),
                              'pruned_nonzero_elements': layer['pruned_nonzero'],
                              'zero_fraction_after_pruning': 1.0 - len(layer['values']) / count,
                              'coo_reconstruction_matches': True, 'csr_reconstruction_matches': True})
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'source_header': str(source), 'source_sha256': source_sha,
              'architecture': list(shape), 'float_bytes': 4,
              'pruning_rule': 'abs(weight) < threshold -> 0; equality is retained',
              'compressed_parameters': ['weight_1', 'weight_2'],
              'unchanged_parameters': ['bias_1', 'bias_2', 'weight_3', 'bias_3'],
              'dense_parameter_bytes': sum(len(v) * 4 for v in arrays.values()),
              'layers': layer_reports, 'formats': formats,
              'zero_nnz_note': 'Zero-NNZ data arrays use one dummy element; its sizeof is included.',
              'arduino_measurement_performed': False}
    # 検証・生成を終えてから新規ディレクトリへ書き出す。上書きはしない。
    output_dir.mkdir(parents=True, exist_ok=False)
    for fmt, header in headers.items():
        folder = output_dir / fmt
        folder.mkdir()
        (folder / 'model_parameters.h').write_text(header, encoding='utf-8')
    (output_dir / 'compression_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    readme = f'''# 圧縮データ

- 構成: {' → '.join(map(str, shape))}
- 元ヘッダー: {source}
- 元ヘッダーのSHA-256: {source_sha}
- 第1層の閾値: {threshold1}、第2層の閾値: {threshold2}
- 条件記録: compression_report.json
- 非圧縮パラメータ容量: {report['dense_parameter_bytes']} B
- COO容量: {formats['coo']['parameter_bytes']} B（圧縮率 {formats['coo']['compression_ratio_percent']:.3f}%）
- CSR容量: {formats['csr']['parameter_bytes']} B（圧縮率 {formats['csr']['compression_ratio_percent']:.3f}%）

第1・第2層の同じ枝刈り後の行列から両形式を生成し、PC上で復元一致を確認した。第3層の重みと全バイアスは変更していない。容量は全パラメータ配列と索引のsizeof相当値であり、ヘッダーの文字数やArduino全体の使用メモリではない。NNZ=0の配列には標準C++のためのダミー1要素があり、その容量も計上する。

両形式ともweight_1・weight_2の非圧縮配列は含まない。元のdense用predict()と静的アサートでは使えないため、COO用・CSR用の推論処理が必要。計測用parameterBytes()も、MODEL_PARAMETER_BYTESまたは生成した全配列のsizeof合計に対応させること。形式と層ごとの閾値も記録する。Arduino上では元の行列へ復元せず、疎配列から直接推論する。

学習や実機操作、MSE・推論時間の測定はこのスクリプトでは行わない。元ファイルも変更しない。
'''
    (output_dir / 'README.md').write_text(readme, encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='非圧縮のmodel_parameters.h')
    parser.add_argument('--threshold', type=float, help='第1・第2層の共通閾値')
    parser.add_argument('--threshold1', type=float, help='第1層の閾値（threshold2と一緒に指定）')
    parser.add_argument('--threshold2', type=float, help='第2層の閾値（threshold1と一緒に指定）')
    parser.add_argument('--output-dir', type=Path, help='新規出力フォルダ。省略時は入力ファイルと同じフォルダ内に作成')
    args = parser.parse_args()
    if args.threshold is not None:
        if args.threshold1 is not None or args.threshold2 is not None:
            parser.error('--thresholdと--threshold1/--threshold2は併用できません')
        t1 = t2 = args.threshold
    else:
        if args.threshold1 is None or args.threshold2 is None:
            parser.error('--threshold、または--threshold1と--threshold2の両方を指定してください')
        t1, t2 = args.threshold1, args.threshold2
    if not all(math.isfinite(t) and t >= 0 for t in (t1, t2)):
        parser.error('閾値は0以上の有限値にしてください')
    slug = lambda t: str(t).replace('.', 'p').replace('+', '')
    output_dir = args.output_dir or args.input.resolve().parent / f'compressed_t1_{slug(t1)}_t2_{slug(t2)}'
    try:
        report = convert(args.input, output_dir, t1, t2)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(1, f'エラー: {exc}\n')
    print('保存先:', output_dir.resolve())
    print('構成:', ' → '.join(map(str, report['architecture'])))
    print('非圧縮容量:', report['dense_parameter_bytes'], 'B')
    for i, layer in enumerate(report['layers'], 1):
        print(f'第{i}層: 閾値={layer["threshold"]}, 非ゼロ={layer["nonzero_elements"]}/{layer["total_elements"]}, 復元確認=一致')
    for fmt, data in report['formats'].items():
        print(f'{fmt.upper()}: {data["parameter_bytes"]} B, 圧縮率={data["compression_ratio_percent"]:.3f}%')
    print('ArduinoではCOO/CSR用の推論処理が別途必要です。実機測定は行っていません。')


if __name__ == '__main__':
    main()
