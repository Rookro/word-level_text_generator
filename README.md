# word-level_text_generator

GRUを用いて与えられた単語（単語列）から次の単語を予測（生成）するニューラルネットワークです．
入力に単語，あるいは単語列を与え，次に出現する単語を出力します．

まず，単語に区切られたコーパス（テキストファイル）を用意します．
以下のようにそのテキストファイル（例えばcorpus.txt）を引数にして実行することで，そのテキストでモデルの訓練を行い，エポックごとに予測結果をExample*.txtとして出力します．
```
word-level_text_gen.py　corpus.txt
```
