# requires data to be available
# curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# tar -xf aclImdb_v1.tar.gz
# rm -r aclImdb/train/unsup

import os
import pathlib

# Define a function to read the text data and return text and label pairs
def read_text_data(data_path):
    texts = []
    labels = []
    for label in ['pos', 'neg']:
        label_path = os.path.join(data_path, label)
        for text_file in os.listdir(label_path):
            with open(os.path.join(label_path, text_file), 'r', encoding='utf-8') as f:
                text = f.read()
            labels.append(1 if label == 'pos' else 0)
            texts.append(text)
    return texts, labels

# Path to the directory of the saved dataset
data_path = pathlib.Path("aclImdb")

# Read the text data and labels from the train directory
texts, labels = read_text_data(data_path/'train')

print(f'Successfully read {len(texts)} texts, and {len(labels)} labels from training dataset')

import pickle
filehandler = open("text.pkl","wb")
pickle.dump(texts,filehandler)
filehandler.close()

from torchtext.data.utils import get_tokenizer

# Define a tokenizer function to preprocess the text
tokenizer = get_tokenizer('basic_english')

from torchtext.vocab import build_vocab_from_iterator

# Build the vocabulary from the text data
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Define a function to numericalize the text
def numericalize_text(text):
    return [vocab[token] for token in tokenizer(text)]

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Define a custom dataset class for the text data
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, numericalize_text):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.numericalize_text = numericalize_text

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        numericalized_text = self.numericalize_text(text)
        return numericalized_text, label

    def __len__(self):
        return len(self.labels)
    
# Create train and validation datasets
dataset = CustomTextDataset(texts, labels, vocab, numericalize_text)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.utils.rnn import pad_sequence

# preprocess the data with a collate function, and pads the input sequences to the maximum length in the batch:
def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text)
        text_list.append(processed_text)
    padded_text = pad_sequence(text_list, batch_first=False, padding_value=1.0)
    return torch.tensor(label_list, dtype=torch.float64).to(device), padded_text.to(device)

# Create train and validation data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=False)


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
    
# Create an instance of the text classification model with the given vocabulary size, embedding dimension and output dimension

model = TextClassificationModel(vocab_size = len(vocab), embedding_dim = 100, output_dim = 1)

# Define a loss function based on binary cross entropy and sigmoid activation
criterion = nn.BCEWithLogitsLoss()
# Define an optimizer that updates the model parameters using Adam algorithm
optimizer = torch.optim.Adam(model.parameters())

# Move the model to the device (CPU or GPU) for computation
model = model.to(device)

for epoch in range(10):
  epoch_loss = 0
  epoch_acc = 0
  
  model.train()
  for label, text in train_loader:
      optimizer.zero_grad()
      predictions = model(text).squeeze(1)
      loss = criterion(predictions, label)
      
      rounded_preds = torch.round(
          torch.sigmoid(predictions))
      correct = (rounded_preds == label).float()
      acc = correct.sum() / len(correct)
      
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  print("Epoch %d Train: Loss: %.4f Acc: %.4f" % (epoch + 1, epoch_loss / len(train_loader), 
                                                  epoch_acc / len(train_loader)))

  epoch_loss = 0
  epoch_acc = 0
  model.eval()
  with torch.no_grad():
    for label, text in val_loader:
      predictions = model(text).squeeze(1)
      loss = criterion(predictions, label)
      
      rounded_preds = torch.round(torch.sigmoid(predictions))
      correct = (rounded_preds == label).float()
      acc = correct.sum() / len(correct)
      
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  print("Epoch %d Valid: Loss: %.4f Acc: %.4f" % (epoch + 1, epoch_loss / len(val_loader), 
                                                  epoch_acc / len(val_loader)))
  

  # Read the text data and labels from the test directory
test_labels, test_texts = read_text_data(data_path/'test')

# Create a custom text dataset object for the test data using the vocabulary and numericalize function
test_dataset = CustomTextDataset(test_labels, test_texts, vocab, numericalize_text)

# Create a data loader for the test dataset
test_loader = DataLoader(test_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=False)


test_loss = 0
test_acc = 0
model.eval()
with torch.no_grad():
  for label, text in test_loader:
    predictions = model(text).squeeze(1)
    loss = criterion(predictions, label)
    
    rounded_preds = torch.round(
        torch.sigmoid(predictions))
    correct = (rounded_preds == label).float()
    acc = correct.sum() / len(correct)

    test_loss += loss.item()
    test_acc += acc.item()

print("Test: Loss: %.4f Acc: %.4f" %
        (test_loss / len(test_loader), 
        test_acc / len(test_loader)))

torch.save(model.state_dict(), 'sentiment-model.pt')




# Define a text pipeline function that tokenizes and numericalizes a given sentence using the vocabulary
text_pipeline = lambda x: vocab(tokenizer(x))

# Define a function that predicts the sentiment of a given sentence using the model
def predict_sentiment(model, sentence):
    model.eval()
    text = torch.tensor(text_pipeline(sentence)).unsqueeze(1).to(device)
    prediction = model(text)
    return torch.sigmoid(prediction).item()

# small test
# load from saved state
loaded_model = TextClassificationModel(vocab_size = len(vocab), embedding_dim = 100, output_dim = 1)
loaded_model.load_state_dict(torch.load('sentiment-model.pt'))

sentiment = predict_sentiment(model, "Very bad movie")
if sentiment > 0.5:
    print("positive")
else:
    print("negativ")

sentiment = predict_sentiment(model, "This movie is awesome")
if sentiment > 0.5:
    print("positive")
else:
    print("negativ")