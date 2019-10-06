from pathlib import Path

class params(object):
    
    def __init__(self):
        self.LW = 1e-5
        self.LC = 1e-5
        self.eta = 0.05
        self.a = 1e-3 # SIF 平滑参数
        self.rmpc = 1
        self.word2vec_path = Path.cwd().parent / 'data/zhwiki_word2vec/word2vec.model'
        self.stop_words_path = Path.cwd().parent / 'data/stop_words.txt'

    def __str__(self):
        t = "LW", self.LW, ", LC", self.LC, ", eta", self.eta
        t = map(str, t)
        return ' '.join(t)

if __name__ == "__main__":
    print(Path.cwd())