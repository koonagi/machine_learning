"""
0．PKE実行に必要なライブラリのインストール
"""

# !pip install ginza nltk ja-ginza -q
# !pip install git+https://github.com/boudinfl/pke.git -q

"""
1．歌詞データ取得
以下のサイト参照
https://qiita.com/nekoumei/items/b1afca7cfb9e54303ab4
"""

"""
2. キーフレーズ抽出
"""
import ginza
import nltk 
nltk.download('stopwords')
import spacy
import pke 
from  spacy.lang.ja import stop_words

pke.base.lang_stopwords['ja_ginza'] = 'japanese'
spacy_model = spacy.load("ja_ginza") 
stopwords = list(stop_words.STOP_WORDS)
nltk.corpus.stopwords.words_org = nltk.corpus.stopwords.words
nltk.corpus.stopwords.words = lambda lang : stopwords if lang == 'japanese' else nltk.corpus.stopwords.words_org(lang)

def get_key_phrase(spacy_model, text, n):
    # キーフレーズの抽出
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='ja_ginza', normalization=None, spacy_model= spacy_model )
    extractor.candidate_selection( pos={'NOUN', 'PROPN', 'ADJ', 'NUM'})
    extractor.candidate_weighting(threshold=0.74, method='average', alpha=1.1)
    key_phrase = extractor.get_n_best(n)
    
    return  key_phrase

# キーフレーズを抽出し、データフレームに追加
key_phrase_list = []
for Lyric in artist_df['Lyric']:
    tmp_key_phrase_list = get_key_phrase(spacy_model,Lyric,3)
    tmp_list = []
    for i in tmp_key_phrase_list:
        tmp_list.append(i[0])
    key_phrase_list.append(tmp_list)
    
key_phrase_df =  pd.DataFrame(key_phrase_list)
key_phrase_df.columns = ['Key_Phrase_1', 'Key_Phrase_2', 'Key_Phrase_3']
artist_df_all = pd.concat([artist_df, key_phrase_df], axis=1)

# 分析結果の表示
artist_df_all[['SongName','Key_Phrase_1', 'Key_Phrase_2', 'Key_Phrase_3']]
