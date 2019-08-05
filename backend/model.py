# -*- coding: utf-8 -*
from pyltp import SentenceSplitter, Segmentor, Postagger, NamedEntityRecognizer, Parser
from pathlib import Path

LTP_DATA_DIR = Path('C:\\Users\\xxx\\Desktop\\NLP\\Automatic-Extract-Speech\\backend\\data\\ltp_data')
# LTP_DATA_DIR = Path('data\\ltp_data')
cws_model_path = LTP_DATA_DIR / 'cws.model'
pos_model_path = LTP_DATA_DIR / 'pos.model'
ner_model_path = LTP_DATA_DIR / 'ner.model'
parser_model_path = LTP_DATA_DIR / 'parser.model'

segmentor = Segmentor()
postagger = Postagger()
ner = NamedEntityRecognizer()
parser = Parser()

segmentor.load(str(cws_model_path))
postagger.load(str(pos_model_path))
ner.load(str(ner_model_path))
parser.load(str(parser_model_path))

examples = [
    '“我们养活了美国，但这届政府对待小规模农民很糟糕”，波伊抱怨',  # 和第三个属于同一种类型，依存解析会有错误，将主语的动作依附于说的话里面的动词或其他
    '博弈说他们太差劲了', # 可通过说的同义动词找到该动作的主语，并且说的话里面的依存根也会依赖于这个说的动作，然后说的话里的主语依附于依存根，这就找到了说的话里的主语，也就是开始
    '“你们怎么能干这种事呢”，小明大吼', 
    '不少网友留言嘲讽：“这似乎是韩国海军陆战队争取国防预算的软文”，“记者大概是海军陆战队退役的吧”',
    '但韩国网友对“韩国海军陆战队世界第二”的说法不以为然'
]

words = list(segmentor.segment(examples[4]))
postags = list(postagger.postag(words))
netags = list(ner.recognize(words, postags))
arcs = [(arc.head, arc.relation) for arc in list(parser.parse(words, postags))]

print(words)
print(postags)
print(netags)
print([((arc[0], arc[1]), word, i+1) for i, (arc, word) in enumerate(zip(arcs, words))])
verb = 2
person = None
for i, tag in enumerate(netags):
    if tag == 'S-Nh' and arcs[i][0] == verb and arcs[i][1] == 'SBV':
        person = words[i]


segmentor.release()
postagger.release()
ner.release()
parser.release()