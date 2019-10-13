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
import logging

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextRankSummary:

    def __init__(self):
        self.path = Path.cwd()
        self.word_file = self.path.parent / 'data/word_vec_file.txt'
        self.weight_file = self.path.parent / 'data/weight_file.txt'
        self.params = params.params()
        self.words, self.vectors = data_io.getWordmap(self.word_file)
        self.word2weight = data_io.getWordWeight(self.weight_file, self.params.a)
        self.weight4ind = data_io.getWeight(self.words, self.word2weight)

    def _preprocessing(self, text: str, top_k: int, use_sif: bool) -> Tuple[List[str], np.ndarray, int]:
        """split text to sentences, use SIF weighted or average word embedding to get sentence embedding
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
        if use_sif:
            w = data_io.seq2weight(x, m, self.weight4ind)
            embedding = SIF_embedding.SIF_embedding(self.vectors, x, w, self.params)
        else:
            embedding = np.zeros((len(sentences), 300))
            for i in range(embedding.shape[0]):
                tmp = np.zeros((1, 300))
                count = 0
                for j in range(x.shape[1]):
                    if m[i, j] > 0 and x[i, j] >= 0:
                        tmp += self.vectors[x[i, j]]
                        count += 1
                embedding[i, :] = tmp / count
        return sentences, embedding, top_k

    def _page_rank(self, matrix: np.ndarray, tol: float = 1e-3, d: float = 0.85,
                   max_iter: int = 100, verbose: bool = False) -> Dict[int, float]:
        """init a score vector, use it to do matrix multiplication with recommendation matrix
        matrix: recommendation matrix, matrix[i, j] is the cosine similarity between sentence i and sentence j
        tol: tolerance for stopping criteria
        d: damping factor that can be set between 0 and 1
        max_iter: maximum number of iterations taken for the solvers to converge
        verbose: print information of solver
        return scores: the score of each sentence
        """
        num_sentences = matrix.shape[0]
        scores = np.ones(num_sentences) / num_sentences
        l2_norm = 1
        it = 1
        while l2_norm > tol and it < max_iter:
            tmp = scores
            scores = (1 - d) + d * scores @ (matrix / matrix.sum(axis=1, keepdims=True))
            scores /= scores.sum()
            l2_norm = np.linalg.norm(scores - tmp) / num_sentences
            if verbose: logger.info(f'iter {it} times, l2_norm is {l2_norm}, tol is {tol}')
            it += 1
        return {i: score for i, score in enumerate(scores.squeeze())}

    def _text_rank(self, text: str, top_k: int, use_sif: bool) -> Tuple[Dict[int, float], List[str]]:
        """use TextRank to get scores of scores
        return
        -------------
        scores: dict, key is index of sentence, value is score used in rank
        sentences: list, a list of sentences, not cut, used in reorganizing into summary
        """
        sentences, sentences_embedding, top_k = self._preprocessing(text, top_k, use_sif)
        num_sentences = sentences_embedding.shape[0]
        similarity_matrix = np.zeros((num_sentences, num_sentences))
        # construct similarity matrix
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                similarity_matrix[i, j] = cosine_similarity(sentences_embedding[i, None], sentences_embedding[j, None])
                similarity_matrix[j, i] = similarity_matrix[i, j]
        similarity_matrix /= (similarity_matrix.sum(axis=1, keepdims=True) + 1e-6)
        # graph = nx.from_numpy_array(similarity_matrix)
        # nx.draw_networkx(graph)
        # plt.show()
        # scores = nx.pagerank(graph, max_iter=5000)
        scores = self._page_rank(similarity_matrix)
        return scores, sentences

    def _summary_by_similarity(self, text: str, top_k: int, use_sif: bool) -> Tuple[Dict[int, float], List[str]]:
        """use cosine similarity to calculate the similarity between each sentence and the text
        return
        -------------
        scores: dict, key is index of sentence, value is score used in rank
        sentences: list, a list of sentences, not cut, used in reorganizing into summary
        """
        sentences, sentences_embedding, top_k = self._preprocessing(text, top_k, use_sif)
        text_embedding = sentences_embedding.mean(axis=0, keepdims=True)
        scores = {}
        for i, embedding in enumerate(sentences_embedding):
            similarity = cosine_similarity(embedding.reshape(1, -1), text_embedding)
            scores[i] = similarity
        return scores, sentences

    def summary(self, text: str, top_k: int = 2, algorithm: str = 'TextRank', use_sif:bool=False) -> str:
        """use TextRank or Cosine Similarity to summary
        text: extract summary for the text
        top_k: select top-k sentences as the summary of text
        algorithm: default TextRank, or Cosine
        use_sif: whether to use sif embedding
        return summary
        """
        if algorithm == 'TextRank':
            scores, sentences = self._text_rank(text, top_k, use_sif)
        elif algorithm == 'Cosine':
            scores, sentences = self._summary_by_similarity(text, top_k, use_sif)
        else:
            raise ValueError('only support TextRank and Cosine Similarity!')
        ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = ranked_sentences[:top_k]
        reductive_word_order = sorted(selected)
        return ''.join(sentences[i] for i, _ in reductive_word_order)


if __name__ == '__main__':
    # import json
    #
    # current = Path.cwd()
    summary = TextRankSummary()
    # with open(current.parent / 'data/nlpcc2017textsummarization/train_with_summ.txt',
    #           encoding='utf8') as f:
    #     examples = [json.loads(line[:-1]) for line in f.readlines()]
    #
    #
    # def cut(sent): return [word for word in jieba.cut(sent)]
    #
    #
    # corpus = [item['article'] for item in examples]
    # example = corpus[1].replace('<Paragraph>', '')
    # print(example)
    # example = '你好漂亮。你好美丽。你怎么这样'
    example = '网易娱乐7月21日报道林肯公园主唱查斯特·贝宁顿 Chester Bennington于今天早上,' \
              '在洛杉矶帕洛斯弗迪斯的一个私人庄园自缢身亡,年仅41岁。此消息已得到洛杉矶警方证实。' \
              '洛杉矶警方透露, Chester的家人正在外地度假, Chester独自在家,上吊地点是家里的二楼。' \
              '一说是一名音乐公司工作人员来家里找他时发现了尸体,也有人称是佣人最早发现其死亡。' \
              '林肯公园另一位主唱麦克信田确认了 Chester Bennington自杀属实,并对此感到震惊和心痛,称稍后官方会发布声明。' \
              'Chester昨天还在推特上转发了一条关于曼哈顿垃圾山的新闻。粉丝们纷纷在该推文下留言,不相信 Chester已经走了。' \
              '外媒猜测,Chester选择在7月20日自杀的原因跟他极其要好的朋友Soundgarden(声音花园)乐队以及AudioslaveChris乐队主唱 Cornell有关,' \
              '因为7月20日是 Chris CornellChris的诞辰。而 Cornell于今年5月17日上吊自杀,享年52岁。 Chris去世后, Chester还为他写下悼文。' \
              '对于 Chester的自杀,亲友表示震惊但不意外,因为 Chester曾经透露过想自杀的念头,他曾表示自己童年时被虐待,导致他医生无法走出阴影,' \
              '也导致他长期酗酒和嗑药来疗伤。目前,洛杉矶警方仍在调查Chester的死因。据悉, Chester与毒品和酒精斗争多年,年幼时期曾被成年男子性侵,' \
              '导致常有轻生念头。 Chester生前有过2段婚姻,育有6个孩子。林肯公园在今年五月发行了新专辑《多一丝曙光OneMoreLight》,' \
              '成为他们第五张登顶ilboard排行榜的专辑。而昨晚刚刚发布新单《 Talking To Myself》MV'
    summ = summary.summary(example, 4)
    print(summ)
    # print(summary.summary(example))

    """
    此消息已得到洛杉矶警方证实。洛杉矶警方透露,Chester的家人正在外地度假,Chester独自在家,上吊地点是家里的二楼。
    林肯公园另一位主唱麦克信田确认了ChesterBennington自杀属实,并对此感到震惊和心痛,称稍后官方会发布声明。
    对于Chester的自杀,亲友表示震惊但不意外,因为Chester曾经透露过想自杀的念头,他曾表示自己童年时被虐待,导致他医生无法走出阴影,也导致他长期酗酒和嗑药来疗伤。

    """

