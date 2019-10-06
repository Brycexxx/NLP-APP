from backend.utils import data_io, params, SIF_embedding
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from pyltp import SentenceSplitter
from pathlib import Path
from typing import List, Tuple, Dict
import re
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TextRankSummary:

    def __init__(self):
        self.path = Path.cwd()
        self.word_file = self.path.parent / 'data/word_vec_file.txt'
        self.weight_file = self.path.parent / 'data/weight_file.txt'
        self.params = params.params()
        self.words, self.vectors = data_io.getWordmap(self.word_file)
        self.word2weight = data_io.getWordWeight(self.weight_file, self.params.a)
        self.weight4ind = data_io.getWeight(self.words, self.word2weight)

    def _preprocessing(self, text: str, top_k: int) -> Tuple[List[str], np.ndarray, int]:
        """split text to sentences, use SIF weighted to get sentence embedding
        return
        ----------
        sentences: List[str], cut from text
        embedding: np.ndarray, sentence embedding
        """
        text = re.sub(r'\s+', '', text)
        sentences = list(SentenceSplitter.split(text))
        top_k = min(top_k, len(sentences))
        sentences_cut = [' '.join(jieba.cut(sentence)) for sentence in sentences]
        x, m = data_io.sentences2idx(sentences_cut, self.words)
        w = data_io.seq2weight(x, m, self.weight4ind)
        embedding = SIF_embedding.SIF_embedding(self.vectors, x, w, self.params)
        return sentences, embedding, top_k

    def _text_rank(self, text: str, top_k: int) -> Tuple[Dict[int, float], List[str]]:
        """use TextRank to get scores of scores
        return
        -------------
        scores: dict, key is index of sentence, value is score used in rank
        sentences: list, a list of sentences, not cut, used in reorganizing into summary
        """
        sentences, sentences_embedding, top_k = self._preprocessing(text, top_k)
        num_sentences = sentences_embedding.shape[0]
        similarity_matrix = np.zeros((num_sentences, num_sentences))
        # construct similarity matrix
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                similarity_matrix[i, j] = cosine_similarity(sentences_embedding[i, None], sentences_embedding[j, None])
                similarity_matrix[j, i] = similarity_matrix[i, j]
        graph = nx.from_numpy_array(similarity_matrix)
        # nx.draw_networkx(graph)
        # plt.show()
        scores = nx.pagerank(graph, max_iter=5000)
        return scores, sentences

    def _summary_by_similarity(self, text: str, top_k: int) -> Tuple[Dict[int, float], List[str]]:
        """use cosine similarity to calculate the similarity between each sentence and the text
        return
        -------------
        scores: dict, key is index of sentence, value is score used in rank
        sentences: list, a list of sentences, not cut, used in reorganizing into summary
        """
        sentences, sentences_embedding, top_k = self._preprocessing(text, top_k)
        text_embedding = sentences_embedding.mean(axis=0, keepdims=True)
        scores = {}
        for i, embedding in enumerate(sentences_embedding):
            similarity = cosine_similarity(embedding.reshape(1, -1), text_embedding)
            scores[i] = similarity
        return scores, sentences

    def summary(self, text: str, top_k: int = 2, algorithm: str = 'TextRank') -> str:
        """use TextRank or Cosine Similarity to summary
        text: extract summary for the text
        top_k: select top-k sentences as the summary of text
        algorithm: default TextRank, or Cosine
        return summary
        """
        if algorithm == 'TextRank':
            scores, sentences = self._text_rank(text, top_k)
        elif algorithm == 'Cosine':
            scores, sentences = self._summary_by_similarity(text, top_k)
        else:
            raise ValueError('only support TextRank and Cosine Similarity!')
        ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = ranked_sentences[:top_k]
        reductive_word_order = sorted(selected)
        return ''.join(sentences[i] for i, _ in reductive_word_order)


if __name__ == '__main__':
    import json

    current = Path.cwd()
    summary = TextRankSummary()
    with open(current.parent / 'data/nlpcc2017textsummarization/train_with_summ.txt',
              encoding='utf8') as f:
        examples = [json.loads(line[:-1]) for line in f.readlines()]


    def cut(sent): return [word for word in jieba.cut(sent)]


    corpus = [item['article'] for item in examples]
    example = corpus[0].replace('<Paragraph>', '')
    summ = summary.summary(example, 2, algorithm='Cosine')
    print(summ)
