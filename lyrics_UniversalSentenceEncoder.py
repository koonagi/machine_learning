"""
1．歌詞データ取得
以下のサイト参照
https://qiita.com/nekoumei/items/b1afca7cfb9e54303ab4
"""

"""
2. 類似度算出
"""
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans
import japanize_matplotlib
import seaborn as sns
import networkx as nx

# Universal Sentence Encoderモデルのロード
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# 結果出力用Dataframe作成
vectors = embed(artist_df['Lyric'])
df_result = pd.DataFrame(index=artist_df['SongName'])

for i, name in enumerate(artist_df['SongName']):
    calc_cos_sim_list= []
    
    # 歌詞同士のCOS類似度算出
    for v in vectors:
        calc_cos_sim_list.append(np.inner(vectors[i],v))
    
    df_result[name] = calc_cos_sim_list

df_result[:10]


"""
3. 類似度のヒートマップの作成
"""
plt.figure(figsize=(20, 12))
sns.heatmap(df_result, fmt='g', cmap='Blues')


"""
4. 各曲の一番類似度が高い曲取得
"""
result = pd.concat([pd.DataFrame(df_result.T.apply(lambda x: x.nlargest(2).idxmin())), pd.DataFrame(df_result.T.apply(lambda x: x.nlargest(2).min()))], axis=1)
result.columns = ['類似度が一番高い曲','類似度']
display(result.sort_values("類似度",ascending=False))


"""
5. 各曲の一番類似度が高い曲どうしのネットワークグラフ図の作成(networkx)
"""
g = nx.Graph()

#シード値の固定
np.random.seed(5)

# 各曲の一番類似度が高い曲を繋げる
for i in result.iterrows():
    g.add_edge(i[0],i[1]['類似度が一番高い曲'])

pr = nx.pagerank(g)
plt.figure(figsize=(20, 10))
pos = nx.spring_layout(g)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_nodes(g, pos, node_color=list(pr.values()), cmap=plt.cm.summer)
# 日本語でも読み込めるように'IPAexGothic'をフォントで使用する
nx.draw_networkx_labels(g, pos,font_family='IPAexGothic')

plt.show()

# ページランクトップ10
pd.DataFrame(pr.values(), index=pr.keys()).sort_values(0,ascending=False)[:10]
