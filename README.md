# アジャイルワーク２

2年後期「アジャイルワーク２」のソースコードと実験用データを保存するリポジトリです。

## フォルダ構成

- `week1/`: 初期のArduinoスキャナーコードとPPMデータ
- `week3/scanner_ver2/`: スキャナー第2版
- `week3/scanner_ver3/`: スキャナー第3版と学習済みモデルのパラメータ
- `week3/train_python/`: PythonによるRGB正規化、学習ノートブック、入力データ、参照画像
- `week3/mse_calculator.html`: MSE計算用のHTMLツール
- `scanner/`: スキャナー関連のメモ
- `実験計画書/`: 実験計画書のLaTeXソース

## 利用方法

Arduinoの各スケッチはArduino IDEで開きます。`scanner_ver3.ino`には同じフォルダの`model_parameters.h`が必要です。

Python関連のコードは`week3/train_python/`にあります。学習ノートブックはこのフォルダを作業ディレクトリとして開き、セルを上から順に実行します。依存関係の記録は`requirements.txt`にあります。ノートブックではPyTorchも使用していますが、現時点の`requirements.txt`には含まれていません。

PPMのRGB値を正規化する例:

```sh
cd week3/train_python
python3 normalize_rgb.py week3_6_6.ppm
```

## 保存対象

元のフォルダ構成を保ち、ソースコード、ノートブック、ヘッダー、メモ、実験用データ・画像を保存します。仮想環境、キャッシュ、授業PDF、スライド、ZIP、LaTeXの生成ファイルは`.gitignore`で除外しています。

## 変更の保存

このリポジトリのルートで以下を実行します。

```sh
git add .
git commit -m "実験コードを更新"
git push
```
