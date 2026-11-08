# text-check: SynthID Text / テキストウォーターマーク検知システム

SynthID Text と汎用テキストウォーターマーク（Kirchenbauer 系ソフトウォーターマーク）を検知するシステムです。Google Colab での動作を想定しています。

## 機能

- **SynthID Text 検知**: 各トークンの g値 (0..3) の分布に対するカイ二乗適合度検定で、故意に埋め込まれたパターンを検出
- **ソフトウォーターマーク検知**: コンテキストハッシュによるグリーンリスト方式の z スコア検定で、watermarked テキストを検出
- **CLI / ライブラリ / Colab ノートブック** の3形態で利用可能

## セットアップ (Colab)

```python
!git clone https://github.com/kamimai2033/text-sheck.git
%cd text-sheck
!pip install -r requirements.txt
```

## 使い方

### Colab ノートブック

`notebooks/synthid_watermark_detection.ipynb` を開いて上から実行してください。Hugging Face のトークナイザを使った実テキスト検知デモが含まれます。

### CLI

```bash
# テキストのソフトウォーターマーク検知
python -m watermark_detect "ここに検査したいテキスト"

# SynthID g値列の検知
python -m watermark_detect --g-values 0,1,2,3,0,1,2,3

# JSON 出力
python -m watermark_detect "テキスト" --json
```

### ライブラリ

```python
from transformers import AutoTokenizer
from watermark_detect import detect

tokenizer = AutoTokenizer.from_pretrained("gpt2")
report = detect("検査したいテキスト", tokenizer=tokenizer)
print(report)
```

## 検定の仕組み

| 方式 | 統計量 | 判定基準 |
|---|---|---|
| ソフトウォーターマーク | z = (green数 − γT) / √(Tγ(1−γ)) | z ≥ 4.0 で検出 |
| SynthID Text | g値分布のカイ二乗統計量 | p < 0.05 で検出 |

ソフトウォーターマークは直前トークンのハッシュからグリーンリストを構成し、正当な生成では約 γ の割合でしかグリーンに当たらないことを利用します。SynthID Text は各トークンの g値 (2bit) が一様分布から大きく外れることを検出します。

## テスト

```bash
python -m unittest discover -s tests -v
```

## 注意事項

- SynthID 検知には生成時に記録された g値列が必要です（g値は通常 API レスポンス等から取得します）
- 検知は統計的推定であり、短いテキストでは誤検出/見逃しが発生します
