# 圧縮データ

- 構成: 3 → 70 → 70 → 3
- 元ヘッダー: /Users/sensui/Documents/2_2S/アジャイル２/実験/train_python/学習/70_70/model_parameters.h
- 元ヘッダーのSHA-256: 96bf967cfafd9037c5a7ea57f6ec4fe34c01da292988d391ba96f8d6d3cc07a4
- 第1層の閾値: 0.03、第2層の閾値: 0.03
- 条件記録: compression_report.json
- 非圧縮パラメータ容量: 21852 B
- COO容量: 2078 B（圧縮率 9.509%）
- CSR容量: 2109 B（圧縮率 9.651%）

第1・第2層の同じ枝刈り後の行列から両形式を生成し、PC上で復元一致を確認した。第3層の重みと全バイアスは変更していない。容量は全パラメータ配列と索引のsizeof相当値であり、ヘッダーの文字数やArduino全体の使用メモリではない。NNZ=0の配列には標準C++のためのダミー1要素があり、その容量も計上する。

両形式ともweight_1・weight_2の非圧縮配列は含まない。元のdense用predict()と静的アサートでは使えないため、COO用・CSR用の推論処理が必要。計測用parameterBytes()も、MODEL_PARAMETER_BYTESまたは生成した全配列のsizeof合計に対応させること。形式と層ごとの閾値も記録する。Arduino上では元の行列へ復元せず、疎配列から直接推論する。

学習や実機操作、MSE・推論時間の測定はこのスクリプトでは行わない。元ファイルも変更しない。

## 測定の準備

対応するCOO・CSRヘッダーを、それぞれ`実験/scanner_coo/`と`実験/scanner_csr/`へ配置した。Arduino UNO R4 WiFi向けのコンパイルと、実際の推論処理を用いたPC上の12入力の一致確認が完了した。結果は`validation_report.json`と各形式の`arduino_build.txt`を参照する。0.03の実機測定は未実施。

実機へ書き込み後、9600 baudで`i`を送って両層の閾値が0.03、隠れ層が70・70であることを確認し、`r`で測定する。測定したPPMは`coo/coo_70_t0p03.ppm`と`csr/csr_70_t0p03.ppm`へ保存する。手順の詳細は`実験/測定条件.md`を参照する。
