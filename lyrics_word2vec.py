"""
1．歌詞データ取得
以下のサイト参照
https://qiita.com/nekoumei/items/b1afca7cfb9e54303ab4
"""


"""
2. コーパス作成
"""

def lang_extract(text):
    # 分かち書きの文章を返す
    m = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/")
    node = m.parseToNode(text)

    corpus = []
    pos_list = ['名詞','動詞','形容詞']
    while node:
        # 基本形を利用する
        word = node.feature.split(',')[6]
        pos = node.feature.split(",")[0]
        
        if pos in pos_list:
            corpus.append(word)
        node = node.next
    
    return " ".join(corpus)

artist_df['Corpus'] = artist_df['Lyric'].apply(lambda x: lang_extract(x))

# 最初の三行をサンプルで出力
artist_df[:3]

# コーパスの作成
corpus = []
tmp_corpus = []

for text in artist_df["Corpus"]:
    text_list = text.split(' ')   
    tmp_corpus.append(text_list)


"""
3. ストップワードの抽出/削除
"""
import collections
import itertools

# 歌詞データの中で頻出単語表示
# tmp_corpus = list(itertools.chain.from_iterable(tmp_corpus))
# collections.Counter(tmp_corpus).most_common(100)

# 意味を持たない文字はストップワードとしてリスト化。学習の対象外とする。
stop_word = ['*','くれ','とめ','きっと','とれる','それぞれ','つけ','とっ','つけ','いる','の','し','い','こと','さ','ない','ん','よう','なっ','いい','しまっ','てる','みたい','する','れ','て','a','もん','き','まま','なる','is','ちゃっ','in','・','the','そう','でき','m','あっ','よ','行っ','く','しよう','つい','せ','これ','なり','やっ','み','かけ','うち','たっ','なら','られる','よそ','なん','ため','いか','あな','いろ','こ','なれ','なれる','したっ','しよ','まみれ','した','しまい','っ','はず','もと','あれ','いら','ちゃえ','ほう','したい','are']

for text in tmp_corpus:
    result = list(filter(lambda x: x not in stop_word, text))
    corpus.append(result)

# ストップワード削除前後の件数出力
print(len(list(itertools.chain.from_iterable(tmp_corpus))),len(list(itertools.chain.from_iterable(corpus))))

"""
4. 学習（word2vec)
"""
from gensim.models import word2vec

model = word2vec.Word2Vec(corpus,
                          size=100,  # ベクトルの次元
                          min_count=2,  # 最小2回以上出ている単語のみの使用
                          window=5,  # window幅
                          iter=1000  # 学習の繰り返し数
                         )

# 学習した言語数
len(model.wv.key_to_index)


"""
5. 分析
"""

# 言葉の足し算
posi_word = ['posi_word1','posi_word2']
result = model.wv.most_similar(positive=posi_word,topn=10)    
df_result = pd.DataFrame(result,columns=["単語","類似度"])
display(df_result.T)

# 言葉の引き算
posi_word = ['posi_word1']
nega_word = ['nega_word1']
result = model.wv.most_similar(positive=posi_word,negative=nega_word,topn=10) 
df_result = pd.DataFrame(result,columns=["単語","類似度"])
display(df_result.T)
