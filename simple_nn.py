import torch
from torch import nn
from torcheval.metrics.functional import multiclass_f1_score
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from functions import clean_data
# nltk.download('wordnet')
MAX_VOCAB = 5000


class Encoder:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=MAX_VOCAB)
        self.vocab = None
        return

    def build_vocab(self, X):
        X['text'] = X['text'].apply(lambda x: [self.lemmatizer.lemmatize(word.lower()) for word in x.split()])
        vectors = self.vectorizer.fit_transform(np.array(X['text'].apply(lambda x: ' '.join(x))))
        self.vocab = self.vectorizer.vocabulary_
        X = vectors.todense()
        return X


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(MAX_VOCAB, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_nn(X, y):
    NUM_ITER = 30
    encode = Encoder()
    X = encode.build_vocab(X)
    y = np.array(y['target'])
    y = torch.from_numpy(y)
    model = NeuralNetwork().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    CrossEntropy = torch.nn.CrossEntropyLoss()
    X = torch.from_numpy(X).float()
    history, f1 = [], []
    prob = None
    for epoch in range(NUM_ITER):
        optimizer.zero_grad()
        logits = model(X)
        prob = nn.Softmax(dim=1)(logits)
        f1.append(multiclass_f1_score(prob, y).item())
        loss = CrossEntropy(prob, y)
        history.append(loss.item())
        loss.backward()
        optimizer.step()
    y_pred = prob.argmax(1)
    acc = torch.sum(y_pred == y) / len(y)
    print('Accuracy is: ', acc.item(), '\nF1 Score is: ', f1[-1])
    return model, history, f1, encode


def plot_f1(history, f1):
    fig, axes = plt.subplots(2, 1)
    fig.tight_layout(pad=2.5)
    epochs = list(range(len(history)))
    title = ['Training Loss', 'F1-Score']
    data = [history, f1]
    for i in range(2):
        axes[i].plot(epochs, data[i])
        axes[i].set_title(title[i])
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(title[i])
    fig.savefig('./plots/plot_f1.png')


def submit(model, encode):
    test = pd.read_csv('test.csv')
    idx = test['id']
    test = clean_data(test)
    test = torch.from_numpy(encode.vectorizer.transform(test['text']).todense()).float()
    output = nn.Softmax(dim=1)(model(test)).argmax(1).detach().numpy()
    submissions = pd.DataFrame({'id': idx, 'target': output}).to_csv('submission.csv', index=False)