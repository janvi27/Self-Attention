# Self-Attention and other Neural Network Models
## Introduction
This repository is aimed at using different neural network models for Natural Language Processing tasks. In this case, the task is to detect whether a given tweet actually 
refers to a disaster or not. <br />
The main aspect of this repository is the coding of the self-attention mechanism, outlined in the paper 
"[Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)" from researchers at Google. Self-Attention is a method used to evaluate the relationships between words and their importance in the overall task. This will be described in further detail in the later sections.

## Dataset
The dataset was obtained from the Kaggle competition "[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/submissions)". It consists of 4 columns, and each row of the dataset is assigned 1 or 0 depending on whether or not the tweet refers to an actual disaster. The labels were approximately equally distributed. <br />

### Pre-Processing
Considering only the text column, the library `nltk` was used to remove stop words and punctuation from the text and tokenize it. Words in each row were lemmatized so that the size of the vocabulary could be reduced while retaining the main information in each tweet. Once this was done, I chose the 5000 most frequently occurring words in the dataset and created a dictionary for the vocabulary which would map the words to an ID. In addition, I chose the length of the longest sentence as the `MAX_SEN_LEN` and padded all rows with the string '<pad>', which got its own ID. To account for any word not in the vocabulary, the string '<UNK>' was added. <br />
This encoding function was then wrapped in the Encoding class, which was used to encode the text from the dataset. Each word was mapped to its corresponding index, so that it could be fed into the Pytorch `Embedding` layer. <br />
