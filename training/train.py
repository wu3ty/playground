from keras.datasets import imdb

# use a subset of the data, load it and give an overview
vocabulary_size = 50000

# y=0: negative rating, y=1: positive rating
(X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(num_words = vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train_raw), len(X_test_raw)))

# get word data
word2id = imdb.get_word_index()

# sample of raw training data
idx = 0
review_len = 10
print(f'IMDB review tokens: {X_train_raw[idx][:review_len]}..., label: {y_train[idx]}')

# sample of underlying data
id2word = {i: word for word, i in word2id.items()}
review = [id2word.get(i, ' ') for i in X_train_raw[idx][:review_len]]
print(f'IMDB review       : {review}..., label: {y_train[idx]}')

# lets check the size of reviews
print(f'Maximum review length: {len(max((X_train_raw + X_test_raw), key=len))}')
print(f'Minimum review length: {len(min((X_train_raw + X_test_raw), key=len))}')

from keras.utils import pad_sequences

max_words = 200
X_train = pad_sequences(X_train_raw, maxlen=max_words)
X_test = pad_sequences(X_test_raw, maxlen=max_words)
print(f'Maximum review length: {len(max((X_train + X_test), key=len))}')
print(f'Minimum review length: {len(min((X_train + X_test), key=len))}')

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SimpleRNN,Dense,Activation,Dropout

# simple RNN
model = Sequential()

model.add(Embedding(vocabulary_size,64,input_length =len(X_train[0])))
model.add(SimpleRNN(32,input_shape = (vocabulary_size,max_words), return_sequences=False,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(1)) #flatten
model.add(Activation("sigmoid")) # using sigmoid for binary classification

print(model.summary())

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

batch_size = 1000
num_epochs = 5
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
history = model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])

model.save("sentiment.keras")

