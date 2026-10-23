"""P3形式のPPMからRGBを読み込み、各成分を0.0～1.0に変換する。"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="入力PPMファイル")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    # PPMのコメントと改行を除いて読み込む。
    tokens = " ".join(
        line.split("#", 1)[0] for line in args.input.read_text().splitlines()
    ).split()
    if len(tokens) < 4 or tokens[0] != "P3":
        parser.error("P3形式のPPMファイルを指定してください。")

    width, height, maximum = map(int, tokens[1:4])
    values = list(map(int, tokens[4:]))
    if width <= 0 or height <= 0 or maximum != 255:
        parser.error("画像サイズは正の整数、最大値は255である必要があります。")
    if len(values) != width * height * 3:
        parser.error(f"RGB値は{width * height * 3}個必要ですが、{len(values)}個あります。")
    if any(value < 0 or value > 255 for value in values):
        parser.error("RGB値は0～255の範囲である必要があります。")

    # 左上から行ごとに、1画素につきR・G・Bをそれぞれ255で割る。
    pixels = [
        [value / 255.0 for value in values[i:i + 3]]
        for i in range(0, len(values), 3)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "week3_6_6_normalized.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["R", "G", "B"])
        writer.writerows(pixels)

    array_path = args.output_dir / "week3_6_6_normalized_X.py"
    rows = ",\n".join(f"    {pixel!r}" for pixel in pixels)
    array_path.write_text(
        "import numpy as np\n\n"
        "# 左上から行ごとに並べた36画素のRGB。各成分は0.0～1.0。\n"
        f"X = np.array([\n{rows}\n], dtype=np.float32)\n",
        encoding="utf-8",
    )

    print(f"{width}×{height}（{len(pixels)}画素）を変換しました。")
    print(f"先頭画素: {values[:3]} → {pixels[0]}")
    print(csv_path)
    print(array_path)


if __name__ == "__main__":
    main()
