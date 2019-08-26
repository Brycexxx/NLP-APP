from pyltp import (
    SentenceSplitter, 
    Segmentor, Postagger, 
    NamedEntityRecognizer, 
    Parser, SementicRoleLabeller
)
import re
from typing import Tuple, List

class Extractor:

    def __init__(self, cws_model_path:str, 
                       pos_model_path:str,
                       ner_model_path:str,
                       parser_model_path:str,
                       srl_model_path:str):
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
        
    def extract_by_labeller(self, sentence:str, says:list) -> Tuple[str, str]:
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
            lower = None
            flag = False
            while j < len(words):
                if netags[j] == 'S-Nh':
                    # 在命名实体前后 10 个词内寻找与‘说’意义相近的谓词
                    lower, upper = max(0, j-10), min(j+10, len(words))
                    while lower < upper:
                        if words[lower] in says and arcs_lst[j][0] == lower+1 and arcs_lst[j][1] == 'SBV':
                            p = re.sub(r'[，。！]', '', words[j]) # 发现有些分词发生错误，将标点包括在人名里，e.g., 。小刚
                            persons.append(p)
                            predicates.append(words[lower])
                            flag = True
                            break
                        lower += 1
                if flag: break
                j += 1
            if lower is None: return persons, predicates, speeches
            start = end = lower
            # 如果谓词后面跟了冒号引号，那么确定为当前人物的言论
            if lower+1 < len(words) and words[lower+1] == '：':
                end = start = lower + 3 # 跳过冒号和引号
                # 直至找到另一半引号作为结束
                while end < len(words):
                    if end == '”':
                        break
                    end += 1
                speeches.append(''.join(words[start: end]))
            elif lower+1 < len(words) and words[lower+1] == '，':
                end = start = lower + 2
                # 如果是逗号，则找到下一个句号作为当前言论的结束
                while end < len(words):
                    if words[end] == '。':
                        break
                    end += 1
                speeches.append(''.join(words[start: end]))
            else: # 前两种都不是，说明言论在前面
                end = lower - 1
                while end >= 0 and words[end] != '”':
                    end -= 1
                start = end 
                while start >= 0 and words[start] != '“':
                    start -= 1
                candidate = words[start+1: end] if 0 <= start <= end < len(words) else ''
                speeches.append(''.join(candidate))
            i = max(j, lower, start, end) # 更新继续搜索下一个言论的下标
        return persons, predicates, speeches

    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.ner.release()
        self.parser.release()
        self.labeller.release()