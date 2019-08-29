from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from random import randint
import pickle, json
from pathlib import Path
from extractor import Extractor
from snownlp import SnowNLP


LTP_DATA_DIR = Path('./models/ltp_data') 
cws_model_path = LTP_DATA_DIR / 'cws.model'
pos_model_path = LTP_DATA_DIR / 'pos.model' 
ner_model_path = LTP_DATA_DIR / 'ner.model' 
parser_model_path = LTP_DATA_DIR / 'parser.model'
srl_model_path = LTP_DATA_DIR / 'pisrl_win.model'
word_vec_path = Path('./models/word2vec/word_vecs.model')

ext = Extractor(cws_model_path, pos_model_path, ner_model_path, parser_model_path, srl_model_path, word_vec_path)

with open(r'./models/say.pickle', 'rb') as f:
    says = pickle.load(f)

says.append('抱怨')

app = Flask(__name__,
            static_folder = "../dist/static",
            template_folder = "../dist")

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template("index.html")

@app.route('/api/extract')
def extract():
    sentence = request.args.get('sentence', '', type=str)
    persons, predicates, speeches = ext.extract_speech_list(sentence, says)
    scores = [round(SnowNLP(speech).sentiments, 2) for speech in speeches]
    sentiments = ['正' if score > 0.5 else '负' for score in scores]
    response = {
        'persons': persons,
        'predicates': predicates,
        'speeches': speeches,
        'scores': scores,
        'sentiments': sentiments
    }
    return jsonify(response)

@app.route('/api/draw')
def draw():
    with open('test.json', 'r') as f:
        resp = json.load(f)
    return jsonify(resp)


if __name__ == '__main__':
    app.run(debug=True)
