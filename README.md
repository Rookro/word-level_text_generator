# word-level_text_generator

GRUを用いて与えられた単語（単語列）から次の単語を予測（生成）するニューラルネットワークである．\\
入力に単語，あるいは単語列を与え，次に出現する単語を出力する．

以下のようにそのテキストファイル（例えばcorpus.txt）を引数にして実行することで，そのテキストでモデルの訓練を行い，エポックごとに予測結果をExample_<実行日時>.txtとして出力する．
```
$ python word-level_text_gen.py　corpus.txt
```

## 準備

単語ごとにスペースで区切られている，訓練させるテキストデータ
（後半で，例としてJUMMAN++を用いたコーパスファイルの作成方法を説明する．）

ソフトウェア環境
python 3.6
Keras 2.2.4
numpy 1.15.2
pydot 1.4.1
tensorboard 1.11.0
tensorflow 1.11.0
その他，word2vec内のファイルの実行にはgensimが必要．
詳細は，requirements.txtを参照． 


訓練にはCPUだと時間がかかるのでGPUの利用できる環境が望ましい．
以下に例として作者の環境を記載する．
  OS: Ubuntu 18.04.1 x86\_64
  CPU: intel Core i7-7700k @4.20GHz
  GPU: Nvidia GeForce GTX 1080
  RAM: 32GB
  python 3.6.7
  tensorflow-gpu 1.11.0
  keras 2.2.4
  gensim 3.4.0
  CUDA 9.0
  cuDNN 7.3.1
