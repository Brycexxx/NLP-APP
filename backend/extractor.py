
from pyltp import SentenceSplitter, Segmentor, Postagger, NamedEntityRecognizer, Parser, SementicRoleLabeller
from typing import Tuple

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

    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.ner.release()
        self.parser.release()
        self.labeller.release()