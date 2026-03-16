# 🧠 Word2Vec with CBOW From Scratch

This project implements a small Word2Vec-style Continuous Bag of Words (CBOW) model in NumPy and uses parquet datasets for training.

The implementation is intentionally simple and educational, so it does not use any deep learning frameworks or advanced optimizations. The goal is to clearly demonstrate the core CBOW algorithm and training process.

The code follows the classic CBOW idea: average context word embeddings and predict the center word. For inference, the model uses a full softmax over the vocabulary. For training, the current implementation uses negative sampling to avoid the full-softmax bottleneck.

## 📁 Project Structure

```text
.
├── cbow/
│   └── word2vec_cbow.py
├── dataset/
│   ├── dataset.py
│   ├── train/
│   ├── test/
│   └── validate/
├── word2vec.ipynb
└── pyproject.toml
```

- the CBOW model lives in `cbow/word2vec_cbow.py`
- an end-to-end training and evaluation workflow lives in `word2vec.ipynb`
- data loading and preprocessing lives in `dataset/dataset.py`

## 💡 What CBOW Does

CBOW predicts a target word from the words around it.

If the sentence is:

```text
the quick brown fox jumps over the lazy dog
```

and the target is `fox`, then with window size `2` the context is:

```text
[quick, brown, jumps, over]
```

In this project, that example is created by `get_cbow_example(...)`.

## ⚙️ Model Parameters

The model has two learnable matrices:

1. Embedding matrix $W \in \mathbb{R}^{V \times d}$ where
each row is the input embedding of one vocabulary item.

2. Context matrix $W' \in \mathbb{R}^{d \times V}$
where each column is an output vector used to score a vocabulary item.

Where:

- $V$ is the vocabulary size
- $d$ is the embedding dimension

In code:

```python
self.embedding_matrix  # shape: (vocab_size, embed_dim)
self.context_matrix    # shape: (embed_dim, vocab_size)
```

## 🔄 Data Pipeline

`DatasetLoader` in `dataset/dataset.py`:

- loads a parquet file
- validates the requested text column
- lowercases text
- removes URLs and non-letter noise
- keeps apostrophes
- filters out sentences with fewer than two tokens

So the training pipeline is:

1. Load parquet text
2. Clean and tokenize
3. Build vocabulary
4. Generate CBOW `(context, target)` examples
5. Train the model

## ➡️ Forward Pass

Given context words $c_1, c_2, \dots, c_C$, CBOW first averages their embeddings.

Let $w_{c_i}$ denote the embedding vector of context word $c_i$. Then:

$$
\bar{h} = \frac{1}{C} \sum_{i=1}^{C} w_{c_i}
$$

This is implemented by:

- `context_word_average(...)` for string tokens
- `context_word_average_by_ids(...)` for integer token ids

Then the model computes raw scores for all words in the vocabulary:

$$
z = W'^T \bar{h}
$$

where $z \in \mathbb{R}^{V}$.

This is implemented by `score_target_words(...)`.

Finally, the model turns scores into probabilities using softmax:

$$
p_j = P(w_j \mid c_1, \dots, c_C) = \frac{e^{z_j}}{\sum_{k=1}^{V} e^{z_k}}
$$

In vector form:

$$
p = \text{softmax}(z)
$$

This is implemented by `softmax(...)` and used in `forward(...)` and `forward_ids(...)`.

## 📉 Full-Softmax Loss

For a target word index $t$, the full-softmax cross-entropy loss is:

$$
L = -\log p_t
$$

Expanding softmax gives:

$$
L = -\log \frac{e^{w_t'^T \bar{h}}}{\sum_{k=1}^{V} e^{w_k'^T \bar{h}}} = -\log {e^{w_t'^T \bar{h}}} + \log {\sum_{k=1}^{V} e^{w_k'^T \bar{h}}} =-w_t'^T \bar{h} + \log \sum_{k=1}^{V} e^{w_k'^T \bar{h}}
$$

which gives:

$$
L = -w_t'^T \bar{h} + \log \sum_{k=1}^{V} e^{w_k'^T \bar{h}}
$$

In this codebase:

- `compute_loss(...)` computes $-\log p_t$
- `cross_entropy_loss(...)` expresses the same idea in expanded form

## 🔁 Full-Softmax Gradients

Although main training method uses negative sampling, the class still contains the standard full-softmax gradient implementation in:

- `compute_gradients(...)`
- `backward(...)`

### Gradient for the context matrix

Given the target word index $t$, the error term for the output layer with respect to the j-th row in context matrix is:

$$
\frac{\partial L}{\partial w_j'} = \frac{\partial}{\partial w_j'} (-w_t'^T \bar{h} + \log \sum_{k=1}^{V} e^{w_k'^T \bar{h}}) = -1_{j=t} \cdot \bar{h} + \frac{1}{\sum_{k=1}^{V} e^{w_k'^T \bar{h}}} \cdot \frac{\partial}{\partial w_j'}(\sum_{k=1}^{V} e^{w_k'^T \bar{h}}) =
$$

$$
= -1_{j=t} \cdot \bar{h} + \frac{1}{\sum_{k=1}^{V} e^{w_k'^T \bar{h}}} \cdot  e^{w_j'^T \bar{h}} \cdot \frac{\partial}{\partial w_j'} (w_j'^T \bar{h}) = -1_{j=t} \cdot \bar{h} + \frac{1}{\sum_{k=1}^{V} e^{w_k'^T \bar{h}}} \cdot  \cdot \bar{h} = (p_j - 1_{j=t}) \bar{h}
$$

$$
-1_{j=t} \cdot \bar{h} + \frac{e^{w_j'^T \bar{h}}}{\sum_{k=1}^{V} e^{w_k'^T \bar{h}}} \bar{h} = (p_j - 1_{j=t}) \bar{h}
$$

where:

$$
1_{j=t} = 1 \space \text{if } j = t \space 0 \space \text{otherwise}
$$

$$
p_j \text{ is the probability of word } j:
$$

$$
p_j = P(w_j \mid c_1, \dots, c_C) = \frac{e^{z_j}}{\sum_{k=1}^{V} e^{z_k}}
$$

In matrix form:

$$
\frac{\partial L}{\partial W'} = \bar{h} \cdot (p - y)^T
$$

This is implemented as:

```python
grad_context_matrix = h_hat @ grad_scores.T
```

where `grad_scores = probs.copy(); grad_scores[target_idx] -= 1`.

### Gradient for the averaged hidden vector

$$
\frac{\partial L}{\partial \bar{h}} = W'(p - y)
$$

Derivation from the full-softmax loss:

$$
L = -w_t'^T\bar{h} + \log\sum_{k=1}^{V} e^{w_k'^T\bar{h}}
$$

Take derivatives term by term with respect to $\bar{h}$:

$$
\frac{\partial}{\partial \bar{h}}\left(-w_t'^T\bar{h}\right) = -w_t'
$$

Let:

$$
S = \sum_{k=1}^{V} e^{w_k'^T\bar{h}}
$$

Then:

$$
\frac{\partial}{\partial \bar{h}}\log S
= \frac{1}{S}\frac{\partial S}{\partial \bar{h}}
= \frac{1}{S}\sum_{k=1}^{V} e^{w_k'^T\bar{h}} w_k'
= \sum_{k=1}^{V} p_k w_k'
$$

So:

$$
\frac{\partial L}{\partial \bar{h}} = -w_t' + \sum_{k=1}^{V} p_k w_k'
$$

Using one-hot target vector $y$ ($y_t=1$, otherwise $0$):

$$
\frac{\partial L}{\partial \bar{h}} = \sum_{k=1}^{V} (p_k - y_k) w_k'
$$

Since columns of $W'$ are $w_k'$, this is exactly:

$$
\frac{\partial L}{\partial \bar{h}} = W'(p-y)
$$



This is implemented as:

```python
grad_h_hat = self.context_matrix @ grad_scores
```

### Gradient for each context embedding

Because the hidden vector is an average,

$$
\bar{h} = \frac{1}{C} \sum_{i=1}^{C} w_{c_i}
$$

each context embedding gets an equal share of the hidden gradient:

$$
\frac{\partial L}{\partial w_{c_i}} = \frac{1}{C} \frac{\partial L}{\partial \bar{h}}
$$

This is implemented as:

```python
grad_word_embeddings = grad_h_hat / len(context_words)
```

This division by $C$ is one of the defining properties of CBOW: each context word receives only a fraction of the total gradient.

## ⚡ Why Training Uses Negative Sampling

A full softmax requires scoring every vocabulary item for every training example:

$$
z = W'^T \bar{h}
$$

That makes each update cost $O(V)$, which becomes expensive as the vocabulary grows.

To speed training up, this project uses negative sampling in `train_example_negative_sampling(...)`.

Instead of optimizing probabilities over the entire vocabulary, the model treats training as a small binary classification problem:

- one positive pair: the true target word
- a few negative pairs: randomly sampled incorrect words

## 🎯 Negative-Sampling Objective

Let:

- $u_t$ be the output vector of the true target word
- $u_{n_1}, \dots, u_{n_K}$ be output vectors of sampled negative words
- $\bar{h}$ be the averaged context embedding

Then the negative-sampling loss for one example is:

$$
L_{NS} = -\log \sigma(u_t^T \bar{h}) - \sum_{i=1}^{K} \log \sigma(-u_{n_i}^T \bar{h})
$$

where:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

This logic is implemented in `train_example_negative_sampling(...)`:

```python
positive_score = float(h_hat @ positive_output)
positive_prob = float(sigmoid(positive_score))

negative_scores = h_hat @ negative_outputs
negative_probs = sigmoid(negative_scores)

loss = -np.log(positive_prob + 1e-10)
loss -= np.sum(np.log(1.0 - negative_probs + 1e-10))
```

## 🔀 Negative-Sampling Gradients

For the positive example, the error term is:

$$
e_{pos} = \sigma(u_t^T \bar{h}) - 1
$$

For each negative sample:

$$
e_{neg,i} = \sigma(u_{n_i}^T \bar{h})
$$

Then the gradient with respect to the averaged hidden vector is:

$$
\frac{\partial L_{NS}}{\partial \bar{h}} = e_{pos} u_t + \sum_{i=1}^{K} e_{neg,i} u_{n_i}
$$

This is implemented as:

```python
grad_h_hat = positive_error * positive_output
grad_h_hat += negative_outputs @ negative_errors
```

The positive output vector update is:

$$
u_t \leftarrow u_t - \alpha \space e_{pos} \bar{h}
$$

The negative output vectors update is:

$$
u_{n_i} \leftarrow u_{n_i} - \alpha \space e_{neg,i} \bar{h}
$$

And each context embedding still receives an equal share of the hidden gradient:

$$
w_{c_i} \leftarrow w_{c_i} - \alpha \space \frac{1}{C} \frac{\partial L_{NS}}{\partial \bar{h}}
$$

In code:

```python
context_gradient = grad_h_hat / len(context_indices)
np.add.at(self.embedding_matrix, context_indices, -learning_rate * context_gradient)
```

## 📝 Important Implementation Note

This project currently mixes two views of CBOW, which presents two possible approaches to CBOW training.

1. Inference and explanation use full softmax.
   - `forward(...)` and `predict(...)` remain easy to understand
2. Training uses negative sampling for speed.
   - `train(...)` is much faster than naive full-softmax training

## 🏋️ Training Overview

`train(...)` performs:

1. normalization of contexts and targets to integer ids
2. repeated negative-sampling updates
3. epoch-level loss averaging
4. early stopping when loss stops improving
5. timeout-based termination

## 🚀 Running The Project

Install dependencies:

```bash
uv sync
```

and open the notebook in your favourite IDE.

The notebook demonstrates:

1. loading parquet data
2. preprocessing text
3. building CBOW examples
4. training with negative sampling
5. saving and loading the model with `joblib`
6. qualitative prediction checks on custom sentences

## ⚠️ Limitations Of This Implementation

This project is intentionally simple, so several limitations are present:

- CBOW treats context as an unordered bag of words
- word order is ignored
- notebook uses only the subset of data, so semantic quality is limited
- the current negative sampling helper samples uniformly, not from the usual smoothed unigram distribution

## 🔭 Possible Next Improvements

Good next steps would be:

1. sample negatives from the standard unigram distribution raised to the 0.75 power
2. add cosine-similarity nearest-neighbor evaluation for learned embeddings
3. add a small intrinsic evaluation set for analogy or similarity testing
4. support weighted context averaging by distance from the target word

## 📚 Reference

Brenndoerfer, Michael. "CBOW Model: Learning Word Embeddings by Predicting Center Words." 2025.

https://mbrenndoerfer.com/writing/cbow-model-word2vec-word-embeddings
