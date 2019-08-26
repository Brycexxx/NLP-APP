from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from random import randint
import pickle
from pathlib import Path
from extractor import Extractor


LTP_DATA_DIR = Path('C:\\Users\\xxx\\Desktop\\NLP\\project-01\\Automatic-Extract-Speech\\backend\\models\\ltp_data') 
cws_model_path = LTP_DATA_DIR / 'cws.model'
pos_model_path = LTP_DATA_DIR / 'pos.model' 
ner_model_path = LTP_DATA_DIR / 'ner.model' 
parser_model_path = LTP_DATA_DIR / 'parser.model'
srl_model_path = LTP_DATA_DIR / 'pisrl_win.model'

ext = Extractor(cws_model_path, pos_model_path, ner_model_path, parser_model_path, srl_model_path)

with open(r'C:\Users\xxx\Desktop\NLP\project-01\Automatic-Extract-Speech\backend\models\say.pickle', 'rb') as f:
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
    response = {
        'persons': persons,
        'predicates': predicates,
        'speeches': speeches
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
    # sentence = '“我们养活了美国，但这届政府对待小规模农民很糟糕”，小明抱怨。小刚称，当前韩国海军陆战队拥有2个师和2个旅。'
    # who, verbs, speech = ext.extract_speech_list(sentence, says)
    # print(speech)