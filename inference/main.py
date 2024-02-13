from flask import Flask
from flask_restx import Api, Resource
import urllib.request

import keras 
from keras.utils import pad_sequences

REST_API_PORT = 5000

# perform preprocessing for inference

# create word to index dictionary
print("Creating word lookup")
NUM_WORDS=50000 # only use top 50000 words
MAX_WORDS = 200
INDEX_FROM=3   # word index offset
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3
id_to_word = {value:key for key,value in word_to_id.items()}

# deserialize keras model
print("Downloading Sentiment model...")
urllib.request.urlretrieve("https://storage.googleapis.com/stefans-modelle/sentiment.keras", "sentiment.keras")

print("Deserializing Keras model")
loaded_model = keras.saving.load_model("sentiment.keras")
print("Done preprocessing")

app = Flask(__name__)
api = Api(app, 
          version='0.1', 
          title='Sentiment Inference API', 
          description='API that demos how to infer the sentiment out of a movie rating')

ns = api.namespace('sentiment')
@ns.route('/<string:sentence>')
@ns.response(200, 'Inference was successful')
@ns.response(400, 'Invalid sentence provided')
@ns.param('sentence', f'The sentence (max {MAX_WORDS} words) for which the sentiment is determined as "positive" or "negative"')
class GenderInference(Resource):
    def get(self,sentence):
        """
            Inferes the sentiment based on a sentence
        """
        if not sentence or len(sentence.split(" ")) > MAX_WORDS:
            api.abort(400, "A sentence needs to be provided with a maximum of 200 words")

        sentence_input = "<START> " + sentence.lower()
        sentence_tokens = [word_to_id.get(i, word_to_id.get("<UNK>")) for i in sentence_input.split(" ")]
        sentence_tokens

        if None in sentence_tokens:
            print("ERROR")

        print(sentence_tokens)
        inference_data = pad_sequences([sentence_tokens], maxlen=MAX_WORDS)
        res = loaded_model.predict(inference_data)

        if res:
            data = {"sentiment" : "negative", "sentence_tokens" : sentence_tokens}
            if res[0][0] > 0.5:
                data["sentiment"] = "positive"
        else:
            api.abort(500, "something went terribly wrong")

        return data     

if __name__ == '__main__':
    app.run(port=REST_API_PORT)
