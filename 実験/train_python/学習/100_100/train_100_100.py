"""既存ノートブックの学習条件で3→100→100→3の元モデルを作成する。"""

import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = OUTPUT_DIR.parents[1]
SEED = 42
EPOCHS = 50_000
LEARNING_RATE = 0.01
LAMBDA_L1 = 1e-6


def read_ppm(path, size):
    tokens = " ".join(line.split("#", 1)[0] for line in path.read_text().splitlines()).split()
    if tokens[:4] != ["P3", str(size[0]), str(size[1]), "255"]:
        raise ValueError(f"PPMの形式・サイズが異なります: {path}")
    values = np.array([int(v) for v in tokens[4:]], dtype=np.float32)
    if values.size != size[0] * size[1] * 3 or not np.all((0 <= values) & (values <= 255)):
        raise ValueError(f"PPMの画素値が不正です: {path}")
    return values.reshape(-1, 3) / 255.0


def read_target(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise ValueError(f"正解画像を読み込めません: {path}")
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST).reshape(-1, 3) / 255.0


class ColorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 3),
        )

    def forward(self, x):
        return self.model(x)


def main():
    # 閾値ごとに学習し直したり、作成済みの元モデルを上書きしたりしない。
    artifacts = ["model_parameters.h", "model_state.pt", "training_report.json", "training_data.json"]
    if any((OUTPUT_DIR / name).exists() for name in artifacts):
        raise FileExistsError("学習済みの成果物があります。再学習時は別フォルダを使用してください。")
    source_files = {
        "notebook": TRAIN_DIR / "train_L1_normalization.ipynb",
        "input_3x3": TRAIN_DIR.parent / "scanner_ver2/before_1669_59.ppm",
        "input_6x6": TRAIN_DIR / "week3_6_6.ppm",
        "target_3x3": TRAIN_DIR / "img/reference_image1_cmyk_large.png",
        "target_6x4": TRAIN_DIR / "img/reference_image2_cmyk_large.png",
    }
    x3 = read_ppm(source_files["input_3x3"], (3, 3))
    x6 = read_ppm(source_files["input_6x6"], (6, 6)).reshape(6, 6, 3)[:4].reshape(24, 3)
    x = np.concatenate([x3, x6]).astype(np.float32)
    y = np.concatenate([
        read_target(source_files["target_3x3"], (3, 3)),
        read_target(source_files["target_6x4"], (6, 4)),
    ]).astype(np.float32)
    if x.shape != (33, 3) or y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError("学習データの形状・値が不正です。")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = ColorNet().cpu()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    xt, yt = torch.tensor(x), torch.tensor(y)
    started = time.perf_counter()
    print(f"architecture=3→100→100→3, samples=33, epochs={EPOCHS}, seed={SEED}", flush=True)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        mse = criterion(model(xt), yt)
        l1 = torch.tensor(0.0, requires_grad=False)
        for name, p in model.named_parameters():
            if "weight" in name:
                l1 = l1 + p.abs().sum()
        loss = mse + LAMBDA_L1 * l1
        if not torch.isfinite(loss):
            raise ValueError(f"学習中に非有限値が発生しました: epoch={epoch}")
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] MSE: {mse.item():.9f} L1: {l1.item():.6f} Loss: {loss.item():.9f}", flush=True)
    with torch.no_grad():
        predictions = model(xt)
        final_mse = criterion(predictions, yt).item()
        final_l1 = sum(p.abs().sum().item() for name, p in model.named_parameters() if "weight" in name)
    lines = []
    for index, layer in enumerate((layer for layer in model.model if isinstance(layer, nn.Linear)), 1):
        for name, tensor in [(f"weight_{index}", layer.weight), (f"bias_{index}", layer.bias)]:
            values = tensor.detach().numpy().flatten()
            if not np.isfinite(values).all():
                raise ValueError(f"学習後の重みが不正です: {name}")
            # float32を再現する9桁の有効数字で、既存と同じ行優先の6配列を保存する。
            literals = []
            for value in values:
                literal = format(float(value), ".9g")
                if "." not in literal and "e" not in literal.lower():
                    literal += ".0"
                literals.append(literal + "f")
            lines.append(f"float {name}[] = {{" + ", ".join(literals) + "};")
    header = OUTPUT_DIR / "model_parameters.h"
    header.write_text("\n".join(lines) + "\n")
    torch.save(model.state_dict(), OUTPUT_DIR / "model_state.pt")
    data_path = OUTPUT_DIR / "training_data.json"
    data_path.write_text(json.dumps({"X": x.tolist(), "Y": y.tolist(), "predictions": predictions.numpy().tolist()}, indent=2) + "\n")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": [3, 100, 100, 3], "samples": 33,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
        "lambda_l1": LAMBDA_L1, "l1_targets": "all Linear weights, excluding biases",
        "seed": SEED, "seed_note": "既存ノートブックはシード未指定。今回は再現用に42を指定した。",
        "device": "cpu", "dtype": "float32", "num_threads": 1,
        "deterministic_algorithms": True,
        "training_duration_seconds": time.perf_counter() - started,
        "final_training_mse_normalized": final_mse,
        "final_weight_l1": final_l1,
        "final_training_loss": final_mse + LAMBDA_L1 * final_l1,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dense_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
        "sources": {key: {"path_from_train_python": str(path.relative_to(TRAIN_DIR)) if path.is_relative_to(TRAIN_DIR) else "../scanner_ver2/before_1669_59.ppm", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in source_files.items()},
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "note": "学習33画素に対するMSE。固定評価9画素の実機MSE・実機推論時間ではない。以後の閾値比較には同じ元モデルを使う。",
        "physical_measurement_performed": False,
    }
    (OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"学習完了: MSE={final_mse:.9f}, parameters={report['parameter_count']}, bytes={report['dense_parameter_bytes']}", flush=True)


if __name__ == "__main__":
    main()
