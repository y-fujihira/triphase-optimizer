README_JA.md

## Triphase Optimizer（トライフェーズ最適化器）

情報コヒーレンスを高め、ノイズ環境に強い深層学習のための新しい損失関数

Triphase Optimizer は、モデルの 安定性（Stability）、情報の一貫性（Coherence）、
そして ノイズ耐性（Noise Robustness） を高めるために設計された
軽量で実用的な損失関数です。

損失関数は次の 3 つの項から構成されます：

① 正相（H+）：精度 … CrossEntropy

② 負相（H−）：安定性 … L2 正則化（重力・張力の役割）

③ 虚相（iHim）：情報コヒーレンス … エントロピー最小化

この「三相（Triphase）」の組み合わせが、モデル内部の迷いを減らし、
高ノイズ環境でもより確実な判断を下せるようにします。

## この最適化器の目的

現代の深層学習においては、精度だけでなく、

重みの安定性

確率分布の一貫性（coherence）

ノイズに対する頑強性

内部エントロピーの制御

が性能の鍵を握ります。

Triphase Optimizer は、これらを 1つの損失関数で同時に制御します。

## Triphase Optimizer の特徴

ノイズ環境下で標準学習より高い性能
迷いの少ない予測（エントロピー低減）が達成される
学習初期の収束が安定
実装が極めてシンプル（10行前後）
あらゆる PyTorch モデルに適用可能
半教師あり学習・ロバスト学習に相性が良い

“新しい物理仮説にインスパイアされた設計” でありながら、使い方は非常に簡単です。

## 実験結果（CNN + Gaussian Noise 1.2）

学習データに 強いノイズ（標準偏差 1.2） を加えた状況で、
テストデータはクリーンな状態のまま評価しています。

1. テスト精度（高いほど良い）

Balanced Triphase は標準 SGD と同等以上の精度を示しました。

<img src="./result_accuracy.png" width="640">
2. 情報位相コヒーレンス（iHim / エントロピー）

Triphase Optimizer は、
予測確率のエントロピー（迷い）を大きく抑制します。

<img src="./result_coherence.png" width="640">

これはモデルがより“整った”内部表現を獲得していることを示します。

## 仕組み：三相干渉構造

Triphase Optimizer の全体損失は次のように定義されています：

Loss = H+（CrossEntropy） 
      + α * H−（L2 正則化）
      + β * iHim（エントロピー最小化）


各相の役割：

相（Phase）	役割	AIにおける効果
H+（正相）	現実との整合（誤差）	精度向上
H−（負相）	張力・拘束・重力	過学習防止、安定
iHim（虚相）	情報の位相、迷いの最小化	コヒーレンス向上、確率分布の引き締め

これらが干渉し合うことで、
“最適な三相バランス点” が自然に生まれます。

## インストール

git clone https://github.com/yourname/triphase-optimizer.git
cd triphase-optimizer
pip install torch torchvision matplotlib

## 使用例（PyTorch）

from triphase_loss import TriphaseLoss

criterion = TriphaseLoss(alpha=0.0005, beta=0.2)

for data, target in train_loader:
    optimizer.zero_grad()
    outputs = model(data)
    loss, l_pos, l_neg, l_im = criterion(outputs, target, model)
    loss.backward()
    optimizer.step()


標準の CrossEntropy を置き換えるだけで使用できます。

## プロジェクト構成
.
├─ triphase_loss.py
├─ experiment.py
├─ result_accuracy.png
├─ result_coherence.png
├─ README.md          # 英語版
├─ README_JA.md       # 日本語版（このファイル）
└─ pdf/
     └─ Triphase_Cosmology__Interference_Structure_of_the_Universe.pdf

## 関連資料

Triphase Optimizer は、
同梱の PDF 「Triphase Cosmology」 に記述された
“干渉テンソル構造” から着想を得ています。

ただし 理論を読まなくても使えます。

## コントリビューション

以下の貢献を歓迎します：

実験の追加（CIFAR, Fashion-MNIST, Transformer系）

精度比較の改善

α, β の自動チューニング

ドキュメント整備

PyPI パッケージ化

## ライセンス

MIT License