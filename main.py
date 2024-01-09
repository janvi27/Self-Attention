import string

from functions import *
from simple_nn import *
from transformer import *

X, y = read_data()
X = clean_data(X)

train, y_train, val, y_val = split_train_val(X, y)
# clf = Model(tf_idf, y['target'])
# model, history, f1, encode = train_nn(X, y)
# plot_f1(history, f1)
# submit(model, encode)


history, val_history, encoder = training(train, y_train, val, y_val)
plot_history(history, val_history)
