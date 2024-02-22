from flask import Flask
from flask_restx import Api, Resource
import urllib.request
import torch
from torchtext.data.utils import get_tokenizer

REST_API_PORT = 8080

# perform preprocessing for inference, so that it executes faster


# deserialize keras model
print("Downloading Sentiment model...")
urllib.request.urlretrieve("https://storage.googleapis.com/stefans-modelle/sentiment-model.pt", "sentiment-model-latest.pt")

print("Downloading text data...")
urllib.request.urlretrieve("https://storage.googleapis.com/stefans-modelle/text.pkl", "text.pkl")

from torch import nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        return self.fc(pooled)
    

# Define a tokenizer function to preprocess the text
tokenizer = get_tokenizer('basic_english')

import pickle
from torchtext.vocab import build_vocab_from_iterator

# load input reference text
file = open("text.pkl",'rb')
texts = pickle.load(file)
file.close()


# Build the vocabulary from the text data
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Define a text pipeline function that tokenizes and numericalizes a given sentence using the vocabulary
text_pipeline = lambda x: vocab(tokenizer(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a function that predicts the sentiment of a given sentence using the model
def predict_sentiment(model, sentence):
    model.eval()
    text = torch.tensor(text_pipeline(sentence)).unsqueeze(1).to(device)
    prediction = model(text)

    pred = torch.sigmoid(prediction).item()
    return "positive" if pred > 0.5 else "negative"

print("Deserializing PyTorch model")
loaded_model = TextClassificationModel(vocab_size = len(vocab), embedding_dim = 100, output_dim = 1)
loaded_model.load_state_dict(torch.load('sentiment-model-latest.pt'))

print("Done preprocessing")

app = Flask(__name__)
api = Api(app, 
          version='0.2', 
          title='Sentiment Inference API', 
          description='API that demos how to infer the sentiment out of a movie rating')

ns = api.namespace('sentiment')
@ns.route('/<string:sentence>')
@ns.response(200, 'Inference was successful')
@ns.response(400, 'Invalid sentence provided')
@ns.param('sentence', f'The sentence for which the sentiment is determined as "positive" or "negative"')
class GenderInference(Resource):
    def get(self,sentence):
        """
            Inferes the sentiment based on a sentence
        """
        if not sentence:
            api.abort(400, "A sentence needs to be provided with a minium of 1 character")

        try:
            prediction = predict_sentiment(loaded_model, sentence)
            
            data = {"sentiment" : prediction}
            
            return data
        except Exception as e:
            api.abort(500, "something went terribly wrong: " + e)

        return data     

if __name__ == '__main__':
    # host="0.0.0.0" is critical
    app.run(host="0.0.0.0", port=REST_API_PORT)
