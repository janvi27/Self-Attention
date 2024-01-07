import torch
from torch import nn
from torcheval.metrics.functional import binary_f1_score, binary_accuracy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
MAX_SEQ_LEN = 27


def positional(seq_len, d, n=10000):
    pos_enc = np.zeros((seq_len, d))
    for pos in range(seq_len):
        for i in np.arange(d // 2):
            denom = n ** (2 * i / d)
            pos_enc[pos, 2 * i] = np.sin(pos / denom)
            pos_enc[pos, 2 * i + 1] = np.cos(pos / denom)
    return torch.from_numpy(pos_enc).float()


class Encoding:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.word2id = None
        self.id2word = None
        self.embed_dim = 0

    def word_to_vector(self, X):
        X['text'] = X['text'].apply(lambda x: " ".join([self.lemmatizer.lemmatize(word.lower()) for word in x.split()]))
        # words = list(set(" ".join(X.text.tolist()).split()))
        word_list = X.text.str.split(expand=True).stack().value_counts().reset_index()[:5000]
        self.word2id = {w: i + 1 for i, w in enumerate(word_list)}
        self.id2word = {i + 1: w for i, w in enumerate(word_list)}
        self.id2word[0], self.word2id['<pad>'] = '<pad>', 0
        self.id2word[-1], self.word2id['<UNK>'] = '<UNK>', len(self.id2word)
        self.embed_dim = len(self.id2word)

    def encoding(self, seq):
        seq = seq.split()
        if len(seq) < MAX_SEQ_LEN:
            seq += ['<pad>' for _ in range(MAX_SEQ_LEN - len(seq))]
        encode = np.array([self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in seq])
        mask = np.array([1 if i < len(seq) else 0 for i in range(MAX_SEQ_LEN)])
        mask = torch.from_numpy(mask)
        return torch.from_numpy(encode), mask


class Transformer(nn.Module):
    def __init__(self, dq, embed_dim, dv, word_dim=32):
        super().__init__()
        self.embed_dim, self.word_dim = embed_dim, word_dim
        self.dq, self.dv = dq, dv
        self.Embedding = nn.Embedding(self.embed_dim, self.word_dim)
        self.Q, self.K, self.V = nn.Linear(self.word_dim, dq, bias=False), \
            nn.Linear(self.word_dim, dq, bias=False), \
            nn.Linear(self.word_dim, dv, bias=False)
        nn.init.kaiming_normal_(self.Q.weight)
        nn.init.kaiming_normal_(self.K.weight)
        nn.init.kaiming_normal_(self.V.weight)
        self.Softmax1 = nn.Softmax(dim=2)
        self.FFN = nn.Sequential(
            nn.Linear(self.word_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.word_dim),
        )
        self.LayerNorm1 = nn.LayerNorm(64)
        self.LayerNorm2 = nn.LayerNorm(64)
        self.Layer2 = nn.Linear(MAX_SEQ_LEN * self.word_dim, 1)
        self.Classifier = nn.Sequential(
            nn.Linear(MAX_SEQ_LEN, 32),
            nn.PReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x, mask=None):
        data = self.Embedding(x)  # data.shape = 7613 x MAX_SEQ_LEN x word_dim
        data = data + positional(MAX_SEQ_LEN, self.word_dim).unsqueeze(0).repeat(data.shape[0], 1, 1)
        query, key = self.Q(data), self.K(data)
        query, key = query * mask.unsqueeze(-1), key * mask.unsqueeze(-1)
        attn = self.Softmax1(torch.matmul(query, torch.permute(key, (0, 2, 1))) / torch.sqrt(torch.tensor(self.dq)))
        value = self.V(data)
        attn = torch.matmul(attn, value)
        attn = self.LayerNorm1(data + attn)
        out = self.FFN(attn)
        out = self.LayerNorm2(out + attn)
        out = out.sum(dim=2) / out.shape[2]
        # out = out.view(-1, MAX_SEQ_LEN * out.shape[-1])
        out = self.Classifier(out)
        return out


def training(x, y):
    NUM_EPOCHS = 1
    y = torch.from_numpy(np.array(y['target']))
    encoder = Encoding()
    # seq_len = np.array(x['text'].apply(lambda tweet: len(tweet.split())))
    encoder.word_to_vector(x)
    data = torch.zeros(len(x), MAX_SEQ_LEN, dtype=torch.long)
    mask = torch.ones(len(x), MAX_SEQ_LEN)
    for i, row in x.iterrows():
        data[i, :], mask[i] = encoder.encoding(row['text'])  # len(x) * MAX_SEQ_LEN
    transformer = Transformer(dq=64, embed_dim=encoder.embed_dim, dv=64, word_dim=64).to("cpu")
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    history = {'loss': [], 'accuracy': [], 'f1': []}
    # Training
    batch_size = 64
    for epoch in range(NUM_EPOCHS):
        print('Epoch: ', epoch)
        for i in range(0, len(x) - batch_size, batch_size):
            start = i
            end = i + batch_size if i + batch_size < len(x) else len(x)
            batch, target = data[start:end], y[start:end]
            logits = transformer(batch, mask[start:end]).view(target.shape[0], 2)
            prob = nn.Softmax(dim=1)(logits)
            y_pred = prob.argmax(1)
            acc, f1 = binary_accuracy(y_pred, target), binary_f1_score(y_pred, target)
            cost = loss(logits, target)
            print('Loss: ', cost.item(), '; Accuracy: ', acc.item(), '; F1 Score: ', f1.item())
            history['loss'].append(cost.item())
            history['accuracy'].append(acc.item())
            history['f1'].append(f1.item())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
