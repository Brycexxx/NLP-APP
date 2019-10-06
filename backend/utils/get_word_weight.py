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
    get_weight_and_word_vec_file()
     