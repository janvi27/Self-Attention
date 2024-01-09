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
        word_list = X.text.str.split(expand=True).stack().value_counts().reset_index()['index'].tolist()[:6000]
        self.word2id = {w: i + 1 for i, w in enumerate(word_list)}
        self.id2word = {i + 1: w for i, w in enumerate(word_list)}
        self.id2word[0], self.word2id['<pad>'] = '<pad>', 0
        self.id2word[-1], self.word2id['<UNK>'] = '<UNK>', len(self.id2word)
        self.embed_dim = len(self.id2word)

    def encoding(self, seq):
        seq = seq.split()
        mask = np.array([1 if i < len(seq) else 0 for i in range(MAX_SEQ_LEN)])
        if len(seq) < MAX_SEQ_LEN:
            seq += ['<pad>' for _ in range(MAX_SEQ_LEN - len(seq))]
        encode = np.array([self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in seq])
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
            nn.Dropout(0.5)
        )
        self.LayerNorm1 = nn.LayerNorm(64)
        self.LayerNorm2 = nn.LayerNorm(64)
        self.Layer2 = nn.Linear(MAX_SEQ_LEN * self.word_dim, 1)
        self.Classifier = nn.Sequential(
            nn.Linear(MAX_SEQ_LEN, 32),
            nn.PReLU(),
            nn.Dropout(0.4),
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


def split_train_val(x, y):
    train = x.sample(frac=0.8)
    val = x.drop(train.index)
    y_train, y_val = y.loc[train.index].reset_index(), y.loc[val.index].reset_index()
    return train.reset_index(), y_train, val.reset_index(), y_val


def validation(x, mask, y, transformer, loss):
    transformer.eval()
    with torch.no_grad():
        logits = transformer(x, mask)
        y_pred = nn.Softmax(dim=1)(logits).argmax(1)
        cost = loss(logits, y)
        acc, f1 = binary_accuracy(y_pred, y), binary_f1_score(y_pred, y)
    return cost, acc, f1


def encode_data(x, encoder=None):
    initialize = True if encoder is None else False
    if encoder is None:
        encoder = Encoding()
        encoder.word_to_vector(x)
    data = torch.zeros(len(x), MAX_SEQ_LEN, dtype=torch.long)
    mask = torch.ones(len(x), MAX_SEQ_LEN)
    for i, row in x.iterrows():
        data[i, :], mask[i] = encoder.encoding(row['text'])  # len(x) * MAX_SEQ_LEN
    if initialize:
        return data, mask, encoder
    else:
        return data, mask


def training(x, y, x_val, y_val):
    NUM_EPOCHS = 80
    y = torch.from_numpy(np.array(y['target']))
    y_val = torch.from_numpy(np.array(y_val['target']))
    data, mask, encoder = encode_data(x)
    data_val, mask_val = encode_data(x_val, encoder)

    transformer = Transformer(dq=64, embed_dim=encoder.embed_dim, dv=64, word_dim=64).to("cpu")
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.01)
    loss = torch.nn.CrossEntropyLoss()
    history = {'loss': [], 'accuracy': [], 'f1': []}
    val_history = {'loss': [], 'accuracy': [], 'f1': []}
    # print(data)
    # Training
    batch_size = 128
    for epoch in range(NUM_EPOCHS):
        if epoch % 10 == 0:
            print('Epoch: ', epoch)
        transformer.train()
        cost, acc, f1 = 0, 0, 0
        for i in range(0, len(x) - batch_size, batch_size):
            start = i
            end = i + batch_size if i + batch_size < len(x) else len(x)
            batch, target = data[start:end], y[start:end]
            logits = transformer(batch, mask[start:end]).view(target.shape[0], 2)
            prob = nn.Softmax(dim=1)(logits)
            y_pred = prob.argmax(1)
            acc, f1 = binary_accuracy(y_pred, target), binary_f1_score(y_pred, target)
            cost = loss(logits, target)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # print('Loss: ', cost.item(), '; Accuracy: ', acc.item(), '; F1 Score: ', f1.item())
        history['loss'].append(cost.item())
        history['accuracy'].append(acc.item())
        history['f1'].append(f1.item())
        val_cost, val_acc, val_f1 = validation(data_val, mask_val, y_val, transformer, loss)
        val_history['loss'].append(val_cost.item())
        val_history['accuracy'].append(val_acc.item())
        val_history['f1'].append(val_f1.item())

    transformer.eval()
    y_pred = transformer(data, mask).view(y.shape[0], 2)
    y_pred = nn.Softmax(dim=1)(y_pred).argmax(1)
    acc, f1 = binary_accuracy(y_pred, y), binary_f1_score(y_pred, y)
    print("Final Accuracy: ", acc, "; Final F1: ", f1)
    torch.save(transformer.state_dict(), "Attention.pth")
    return history, val_history, encoder


def plot_history(history, val_history):
    fig, axes = plt.subplots(3, 1, sharex=True)
    num_epochs = len(history['loss'])
    epochs = np.arange(1, num_epochs + 1)
    for ax, key in zip(axes, list(history.keys())):
        ax.plot(epochs, history[key], label='Training')
        ax.plot(epochs, val_history[key], label='Validation')
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel(key)
    fig.tight_layout(pad=2.5)
    fig.suptitle("Training vs Validation", y=1.01)
    fig.savefig("./plots/transformer_validation.png", bbox_inches='tight')



