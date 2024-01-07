# Self-Attention
## Introduction
This repository is aimed at using different neural network models for Natural Language Processing tasks. In this case, the task is to detect whether a given tweet actually 
refers to a disaster or not. <br />
The main aspect of this repository is the coding of the self-attention mechanism, outlined in the paper 
"[Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)" from researchers at Google. 
Self-Attention is a method used to evaluate the relationships between words and their importance in the overall task. This will be described in further detail in the later sections.

## Dataset
The dataset was obtained from the Kaggle competition "[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/submissions)". 
It consists of 4 columns, and each row of the dataset is assigned 1 or 0 depending on whether or not the tweet refers to an actual disaster. The labels were approximately equally distributed. <br />

### Pre-Processing
Considering only the text column, the library `nltk` was used to remove stop words and punctuation from the text and 
tokenize it. Words in each row were lemmatized so that the size of the vocabulary could be reduced while retaining the 
main information in each tweet. Once this was done, I chose the 5000 most frequently occurring words in the dataset and 
created a dictionary for the vocabulary which would map the words to an ID. In addition, I chose the length of the 
longest sentence as the `MAX_SEN_LEN` and padded all rows with the string '<pad>', which got its own ID. To account for 
any word not in the vocabulary, the string '<UNK>' was added. <br />
This encoding function was then wrapped in the Encoding class, which was used to encode the text from the dataset. 
Each word was mapped to its corresponding index, so that it could be fed into the Pytorch `Embedding` layer. <br />

In this code, I only used the "text" column of the dataset. However, the dataset also contains "location" and "keyword" 
columns indicating the location referred to by the tweet and the keyword (disaster). However, both of these columns had 
rows with null values, so for the time being, I used only the text from the tweets.

## Self-Attention
<figure>
  <img
  src="/images/attention.png"
  alt="Attention Mechanism">
  <figcaption>Source: Vaswani et. al</figcaption>
</figure>

The Self-Attention Mechanism works by using a set of matrices known as the Query (Q), Key (K) and Value (V). Each of 
these are learnt during training. The idea is to obtain an "attention score" which represents the importance of a 
particular word with respect to other words in the sentence. This method trumps previously used RNNs and LSTM networks 
which had increased complexity in order to store the context of the sentence. <br />
The model is structured as follows:
1. Embed the encoded text using the `Embedding` layer.
2. Add a positional embedding (Vaswani et. al) to store the order of words using sines and cosines
3. Now feed this data to the Q, K and V matrices to generate queries, keys and values
4. Take the dot product of queries and keys and scale by square root of the embedding dimension
5. Take the `Softmax` of the above matrix and multiply these obtained attention scores with the V matrix.
6. Add the original output from Step 2 to the output from Step 5 to avoid "vanishing gradients" and retain important
information, and apply `Layer Normalization`.
7. Feed this through a `Feedforward` network, add to output from Step 6 and apply `Layer Normalization` again
8. Finally, run this through a simply classifier neural network to obtain the final logits


