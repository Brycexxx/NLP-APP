# @Time    : 2019/11/15 12:59
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
from backend.utils.params import params
import pickle
import numpy as np
import jieba
from scipy.spatial.distance import cosine
from functools import reduce
import logging

jieba.setLogLevel(logging.INFO)

params = params()


# 获取停用词
def get_stop_words(stop_words_path):
    with stop_words_path.open(mode='r', encoding='gbk') as f:
        stop_words = set(line.strip() for line in f.readlines() if line.strip())
        stop_words.add('\n')
    return stop_words


stop_words = get_stop_words(params.stop_words_path)


# 分词并去掉停用词
def clean(string):
    return ' '.join(token for token in jieba.cut(string) if token not in stop_words)


class Dialog:

    def __init__(self):
        tfidf_path = params.tfidf
        business_ans_path = params.business_ans
        business_binary_vec_path = params.business_binary_vec
        kmeans_path = params.kmeans

        # load files, which were preprocessed by questions2tfidf
        with tfidf_path.open(mode='rb') as f:
            self.tfidf = pickle.load(f)

        with business_ans_path.open(mode='r', encoding='utf8') as f:
            self.business_ans = [line.strip() for line in f.readlines()]

        with business_binary_vec_path.open(mode='rb') as f:
            self.business_binary_vec = pickle.load(f).split()

        with kmeans_path.open(mode='rb') as f:
            self.kmeans = pickle.load(f)

    def _bool_search_answer_of_business_question(self, question_vec: np.ndarray) -> str:
        """use bool search to get answer in the qa corpus.
        label 1 means the question belongs to business question
        """
        key_word_ids = np.where(question_vec)[0]
        vec_len = len(self.business_binary_vec[0].strip())
        # convert vector string e.g. '01010101111' to int type so that use `&` conveniently
        answers_binary_vec = [int(self.business_binary_vec[id_].strip(), 2) for id_ in key_word_ids]
        answer_ids = reduce(lambda x, y: x & y, answers_binary_vec)
        assert answer_ids != 0
        binary_vec = bin(answer_ids)[2:].rjust(vec_len, '0')
        # use cosine similarity between question and answer to sort candidate answers
        candidate_answer_ids = np.where(list(map(int, binary_vec)))[0]
        candidate_answers = [clean(self.business_ans[id_]) for id_ in candidate_answer_ids]
        candidate_answers_vec = self.tfidf.transform(candidate_answers).toarray()
        sorted_answer_ids = sorted(
            zip(candidate_answers_vec, candidate_answer_ids),
            key=lambda x: cosine(x[0], question_vec)
        )
        return self.business_ans[sorted_answer_ids[0][1]]

    def answer(self, question: str) -> str:
        """output answer of the input question"""
        question = clean(question)
        question_vec = self.tfidf.transform([question]).toarray()[0]
        label = self.kmeans.predict(question_vec.reshape(1, -1))[0]

        if label == 1:  # business question, use bool search to get answer
            return self._bool_search_answer_of_business_question(question_vec)
        else:
            pass


if __name__ == '__main__':
    import pandas as pd

    qa = pd.read_csv(params.qa_path, header=0)
    qa.dropna(inplace=True)
    example = qa[qa.question.apply(lambda line: line.endswith('业务'))]['question'].values[0]
    true_ans = qa[qa.question.apply(lambda line: line.endswith('业务'))]['answer'].values[0]
    bot = Dialog()
    print(example)
    print(f'搜索结果：{bot.answer(example)}')
    print(f'真实结果：{true_ans}')
