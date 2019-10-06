from pyltp import (
    SentenceSplitter, 
    Segmentor, Postagger, 
    NamedEntityRecognizer, 
    Parser, SementicRoleLabeller
)
import re
import numpy as np 
from pathlib import Path
import pickle
from gensim.models import Word2Vec
import jieba
from typing import Tuple, List


class SpeechExtractor:

    def __init__(self, cws_model_path:str, 
                       pos_model_path:str,
                       ner_model_path:str,
                       parser_model_path:str,
                       srl_model_path:str,
		       word_vec_path:str):
        self.segmentor = Segmentor() 
        self.postagger = Postagger() 
        self.ner = NamedEntityRecognizer() 
        self.parser = Parser()  
        self.labeller = SementicRoleLabeller()

        self.segmentor.load(str(cws_model_path)) 
        self.postagger.load(str(pos_model_path)) 
        self.ner.load(str(ner_model_path))
        self.parser.load(str(parser_model_path))
        self.labeller.load(str(srl_model_path))
        self.w2v = Word2Vec.load(str(word_vec_path))
        
    def extract_by_labeller(self, sentence:str, says:list) -> Tuple[str, str]:
        """
        通过语义角色标注提取单个言论
        """
        words = list(self.segmentor.segment(sentence))  
        postags = list(self.postagger.postag(words))  
        # netags = list(self.ner.recognize(words, postags))  
        arcs = self.parser.parse(words, postags) 
        # arcs_lst = [(arc.head, arc.relation) for arc in arcs]   
        roles = self.labeller.label(words, postags, arcs)
        
        person = speech = None
        for role in roles:
            if words[role.index] in says:
                for arg in role.arguments:
                    # if arg.name == 'A0' and any(map(lambda ind: netags[ind] == 'S-Nh', range(arg.range.start, arg.range.end+1))):
                    if arg.name == 'A0':
                        person = ''.join(words[arg.range.start:arg.range.end+1])
                    if arg.name == 'A1':
                        tmp = []
                        for ind in range(arg.range.start, arg.range.end+1):
                            if words[ind] == '。': break
                            tmp.append(words[ind])
                        speech = ''.join(tmp)
            if person and speech: break
            person = speech = None
        return person, speech

    def extract_by_arcs(self, sentence:str, says:list) -> Tuple[str, str]:
        """
        通过依存弧提取单个言论
        """
        words = list(self.segmentor.segment(sentence))  
        postags = list(self.postagger.postag(words))  
        # netags = list(self.ner.recognize(words, postags))  
        arcs = self.parser.parse(words, postags) 
        arcs_lst = [(arc.head, arc.relation) for arc in arcs]
        
        who = speech = None
        for i, (head, relation) in enumerate(arcs_lst):
            if head != 0: continue # 如果不设定为句子的根依存，有可能将说的话里的谓词作为根，但这样也会出现错误
            if words[i] not in says: continue
            # 寻找谓词对应的主语
            for j, (head, relation) in enumerate(arcs_lst):
                if head == i+1 and relation == 'SBV':
                    who = words[j]
                    break
            # 寻找说的话，直接从表达说的谓词后一个开始直至遇到句号结束
            speeches = []
            i += 1
            while words[i] in ['，', '：', '“']:
                i += 1
            for k in range(i, len(words)):
                if words[k] == '。':
                    break
                speeches.append(words[k])
            speech = ''.join(speeches)
        return who, speech

    def extract_speech_list(self, sentence:str, says:list) -> Tuple[List[str], List[str], List[str]]:
        """
        通过命名实体以及依存弧提取多个人物的言论
        """
        words = list(self.segmentor.segment(sentence))  
        postags = list(self.postagger.postag(words))  
        netags = list(self.ner.recognize(words, postags))  
        arcs = self.parser.parse(words, postags) 
        arcs_lst = [(arc.head, arc.relation) for arc in arcs]
        
        persons, predicates, speeches = [], [], []
        i = 0
        while i < len(words):
            # 寻找命名实体
            j = i
            person = lower = None
            flag = False
            while j < len(words):
                if netags[j] == 'S-Nh':
                    # 在命名实体前后 10 个词内寻找与‘说’意义相近的谓词
                    lower, upper = max(0, j-10), min(j+10, len(words))
                    while lower < upper:
                        if words[lower] in says and arcs_lst[j][0] == lower+1 and arcs_lst[j][1] == 'SBV':
                            person = re.sub(r'[，。！]', '', words[j]) # 发现有些分词发生错误，将标点包括在人名里，e.g., 。小刚
                            persons.append(person)
                            predicates.append(words[lower])
                            flag = True
                            break
                        lower += 1
                if flag: break
                j += 1
            if lower is None or person is None: return persons, predicates, speeches
            start = end = lower
            # 如果谓词后面跟了冒号引号，那么确定为当前人物的言论
            if lower+1 < len(words) and words[lower+1] == '：':
                end = start = lower + 3 # 跳过冒号和引号
                # 直至找到另一半引号作为结束
                while end < len(words):
                    if end == '”':
                        break
                    end += 1
                candidate = words[start: end]
            elif lower+1 < len(words) and words[lower+1] == '，':
                end = start = lower + 2
                # 如果是逗号，则找到下一个句号、逗号、分号、感叹号作为当前言论的结束
                while end < len(words):
                    if words[end] in ['。', '，', '；', '！']:
                        break
                    end += 1
                candidate = words[start: end]
            else: # 前两种都不是，说明言论在前面
                end = lower - 1
                while end >= 0 and words[end] != '”':
                    end -= 1
                start = end 
                while start >= 0 and words[start] != '“':
                    start -= 1
                candidate = words[start+1: end] if 0 <= start <= end < len(words) else ['']
                # speeches.append(''.join(candidate))
            i = max(j, lower, start, end) # 更新继续搜索下一个言论的下标
            k = end
            if i == end: # 判断是否结束
                sim = 1.0
                k += 1
                s = k
                while sim > 0.75 and k < len(words):
                    while k < len(words) and words[k] not in ['，', '。', '！', '；']: 
                        k += 1
                    sim = self._sentence_similarity(candidate, words[s: k])
                    s = k = k + 1
            speeches.append(''.join(candidate + words[end: k]))
            i = max(i, k)
        return persons, predicates, speeches

    def _sentence_similarity(self, sent1: List[str], sent2: List[str]) -> float:
        """
        平均词向量计算句子余弦相似度
        """
        def get_sent_vec(sent: List[str]) -> np.ndarray:
            sent_vector_list = []
            for sent in sent:
                if sent in self.w2v.wv.vocab:
                    sent_vector_list.append(self.w2v[sent])
                else:
                    sent_vector_list.append(np.zeros(200, dtype=np.float32))
            return np.mean(np.array(sent_vector_list), axis=0)

        sent1_vector = get_sent_vec(sent1)
        sent2_vector = get_sent_vec(sent2)
        cosine_similarities = np.sum(sent1_vector * sent2_vector) / (
            np.linalg.norm(sent1_vector) * np.linalg.norm(sent2_vector)
            )
        return cosine_similarities

    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.ner.release()
        self.parser.release()
        self.labeller.release()


if __name__ == '__main__':
    LTP_DATA_DIR = Path('./data/ltp_data') 
    cws_model_path = LTP_DATA_DIR / 'cws.model'
    pos_model_path = LTP_DATA_DIR / 'pos.model' 
    ner_model_path = LTP_DATA_DIR / 'ner.model' 
    parser_model_path = LTP_DATA_DIR / 'parser.model'
    srl_model_path = LTP_DATA_DIR / 'pisrl_win.model'
    word_vec_path = Path('./data/word2vec/word_vecs.model')

    ext = Extractor(cws_model_path, pos_model_path, ner_model_path, parser_model_path, srl_model_path, word_vec_path)

    with open(r'C:\Users\xxx\Desktop\NLP\project-01\Automatic-Extract-Speech\backend\data\say.pickle', 'rb') as f:
        says = pickle.load(f)

    # sent1 = '你好漂亮啊'
    # sent2 = '你好漂亮啊'

    # print(ext._sentence_similarity(list(jieba.cut(sent1)), list(jieba.cut(sent2))))
    # sentence = '“我们养活了美国，但这届政府对待小规模农民很糟糕”，小明抱怨。小刚称，当前韩国海军陆战队拥有2个师和2个旅。'
    sentence = '小易说，廖俊涛创作很厉害，廖俊涛创作好厉害'
    who, verbs, speech = ext.extract_speech_list(sentence, says)
    print(speech)
