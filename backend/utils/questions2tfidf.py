# @Time    : 2019/11/15 13:05
# @Author  : Xu Huipeng
# @Blog    : https://brycexxx.github.io/
from backend.utils.params import params
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba
import numpy as np
import pickle
import pandas as pd
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


params = params()

stop_words_path = params.stop_words_path
qa_corpus = params.qa_path

# 获取停用词
def get_stop_words(stop_words_path):
    with stop_words_path.open(mode='r', encoding='gbk') as f:
        stop_words = set(line.strip() for line in f.readlines() if line.strip())
        stop_words.add('\n')
    return stop_words
stop_words = get_stop_words(stop_words_path)

# 分词并去掉停用词
def clean(string):
    return ' '.join(token for token in jieba.cut(string) if token not in stop_words)

# 获取问题，答案文本
qa = pd.read_csv(qa_corpus, header=0).dropna()
qa['question_clean'] = qa['question'].apply(clean)
questions = qa['question_clean'].values
ans = qa['answer'].values

# tf-idf
tfidf = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2, max_features=8000)
questions_vec = tfidf.fit_transform(questions).toarray().astype(np.float16)
with open('../data/tfidf_model.pickle', mode='wb') as f:
    pickle.dump(tfidf, f)
    logger.info('保存 tfidf 模型...')

# 是否以“业务结尾”
def is_business(string):
    return string.endswith('业务')
qa['业务问题'] = qa['question_clean'].apply(is_business)

# 分是否业务问题保存问题向量
questions_vec_transposed = questions_vec.T
questions_vec_transposed = np.where(questions_vec_transposed > 0, 1, 0)
is_business_mask = qa['业务问题']
business_questions_vec = questions_vec_transposed[:, is_business_mask]
business_ans = ans[is_business_mask]
assert business_questions_vec.shape[1] == business_ans.shape[0]
with open('business_ans.txt', 'w', encoding='utf8') as f:
    texts = '\n'.join(business_ans)
    f.write(texts)
    logger.info('保存业务问题答案...')
with open('../data/business_binary_vec.pickle', mode='wb') as f:
    vec = '\n'.join(''.join(map(str, line)) for line in business_questions_vec)
    pickle.dump(vec, f)
    logger.info('保存布尔搜索向量...')


# 问题文本聚类模型
assert questions_vec.dtype == np.float16
kmeans = KMeans(n_clusters=2, n_jobs=4, verbose=5)
kmeans.fit(questions_vec)
with open('../data/kmeans.pickle', mode='wb') as f:
    pickle.dump(kmeans, f)
    logger.info('保存 KMeans 模型...')
example_business_vec = questions_vec[is_business_mask][0].reshape(1, -1)
business_label = kmeans.predict(example_business_vec)[0]
logger.info(f'业务问题标签是{business_label}')

