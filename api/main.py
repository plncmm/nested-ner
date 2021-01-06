from flask import Flask, render_template, request
from src.inference import model_predict
from model import BiLSTM
import json
from dataset.data_preprocesing import create_dataset
##creating a flask app and naming it "app"
app = Flask(__name__)
with open('params.json', 'r') as json_file:
        params = json.load(json_file)
      
dtrain, dval, test_data, vocab, tags, vocab_dict = create_dataset('entities.conll', 0.9, 0.1, 0.1)

model = BiLSTM(vocab_size = len(vocab), n_tags = len(tags), embedding_dim=params["embedding_dim"], lstm_dim= params["lstm_dim"], \
             embedding_dropout=params["embedding_dropout"], lstm_dropout=params["lstm_dropout"], \
             output_layer_dropout=params["linear_dropout"], lstm_layers= params["lstm_layers"], embedding_weights = None, use_bilstm = params["use_bilstm"], static_embeddings = params["static_embeddings"])
  
model.load_state('best_model.pt')
model.eval()

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        pred = model_predict(model, text, vocab, tags)
        print(pred)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)