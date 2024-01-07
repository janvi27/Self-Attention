import string

from functions import *
from model import *
from transformer import *

X, y = read_data()
X = clean_data(X)
# clf = Model(tf_idf, y['target'])
# model, history, f1, encode = train_nn(X, y)
# plot_f1(history, f1)
# submit(model, encode)

training(X, y)
