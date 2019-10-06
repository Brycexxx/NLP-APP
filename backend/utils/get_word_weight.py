from gensim.models import Word2Vec 
from backend.utils.params import params
from tqdm import tqdm

def get_weight_and_word_vec_file():
    """
    weight_file.txt: each line is w word and its frequency, separate by space
    word_vec_file.txt: each line is a word and its word vector, 
                    line[0] is word, line[1:] is vector, separate by space
    """
    p = params()
    with open(p.stop_words_path, 'r') as f:
        stop_words = set([line[:-1] for line in f.readlines()])
        stop_words.add('\n')
    model = Word2Vec.load(str(p.word2vec_path))
    vlookup = model.wv.vocab
    min_df = 8
    total_words = sum(
        vlookup[word].count for word in vlookup if word not in stop_words and vlookup[word].count > min_df
    )
    weight_file = open(p.word2vec_path.parent.parent / 'weight_file.txt', 'w', encoding='utf8')
    word_vec_file = open(p.word2vec_path.parent.parent / 'word_vec_file.txt', 'w', encoding='utf8')
    for word in tqdm(vlookup):
        if word in stop_words or vlookup[word].count <= min_df: continue
        frequency = vlookup[word].count / total_words
        weight_file.write(word + ' ' + str(frequency) + '\n')
        word_vec_file.write(word + ' ' + ' '.join(map(str, model.wv[word])) + '\n')
    weight_file.close()
    word_vec_file.close()


if __name__ == "__main__":
    # get_weight_and_word_vec_file()
    import jieba
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    p = params()
    model = Word2Vec.load(str(p.word2vec_path))
    a = '你好美丽'
    b = '你好漂亮'
    a_cut = list(jieba.cut(a))
    b_cut = list(jieba.cut(b))
    a_weight = np.array([3.1264584255315635e-06, 5.309067952431014e-05])
    b_weight = np.array([3.1264584255315635e-06, 1.0607626800910661e-05])
    a_embedding = np.zeros((1, 300))
    b_embedding = np.zeros((1, 300))
    for i, word in enumerate(a_cut):
        if word in model.wv:
            a_embedding += model.wv[word] * a_weight[i]
    for i, word in enumerate(b_cut):
        if word in model.wv:
            b_embedding += model.wv[word] * b_weight[i]
    print(cosine_similarity(a_embedding.reshape(1, -1), b_embedding.reshape(1, -1)))
    # a_embedding /= 2
    # b_embedding /= 2
    # sentences = np.concatenate([a_embedding, b_embedding], axis=0)
    # svd = TruncatedSVD(1, random_state=1)
    # x = svd.fit(sentences)
    # v = svd.components_.reshape(-1, 1)
    # sentences -= sentences @ (v@v.T)
    # print(cosine_similarity(sentences[0].reshape(1, -1), sentences[1].reshape(1, -1)))


