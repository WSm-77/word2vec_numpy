import numpy as np
from typing import Dict, List, Sequence, Tuple

sentences: List[str] = ["the quick brown fox jumps over the lazy dog",
             "i love machine learning and neural networks",
             "numpy is great for deep learning implementations"]

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Compute the sigmoid activation for a scalar or NumPy array."""
    return 1 / (1 + np.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities from raw scores."""
    e_x: np.ndarray = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

class Word2VecCBOW:
    """Minimal NumPy-based CBOW helper class for Word2Vec experimentation."""

    def __init__(self, sentences: Sequence[str], window_size: int = 2, embed_dim: int = 50) -> None:
        """Initialize corpus data, vocabulary mappings, and CBOW parameter matrices."""
        self.sentences: Sequence[str] = sentences
        self.sentences_tokens: List[List[str]] = [s.split() for s in sentences]
        self.window_size: int = window_size
        self.vocab, self.word2idx, self.idx2word = self.get_vocab(sentences)
        self.embedding_matrix: np.ndarray = self.initialize_embeddings_matrix(len(self.vocab), embed_dim)
        self.context_matrix: np.ndarray = self.initialize_context_matrix(len(self.vocab), embed_dim)

    def get_vocab(self, sentences: Sequence[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        """Build sorted vocabulary and index mappings from the input sentences."""
        words = " ".join(sentences).split()
        vocab = sorted(list(set(words)))
        word2idx = {w: i for i, w in enumerate(vocab)}
        idx2word = {i: w for w, i in word2idx.items()}
        return vocab, word2idx, idx2word

    def initialize_embeddings_matrix(self, vocab_size: int, embed_dim: int) -> np.ndarray:
        """Initialize input embedding matrix W with uniform random values."""
        return np.random.randn(vocab_size, embed_dim) * 0.01

    def initialize_context_matrix(self, vocab_size: int, embed_dim: int) -> np.ndarray:
        """Initialize output/context matrix W' with uniform random values."""
        return np.random.randn(embed_dim, vocab_size) * 0.01

    def get_cbow_example(self, sentence_words: List[str], target_word_idx: int) -> Tuple[List[str], str]:
        """Generate a CBOW training example: (context_words, target_word)."""
        target = sentence_words[target_word_idx]
        context: List[str] = []

        for offset in range(-self.window_size, self.window_size + 1):
            if offset == 0:  # Skip the target word itself
                continue

            context_idx = target_word_idx + offset
            if 0 <= context_idx < len(sentence_words):
                context.append(sentence_words[context_idx])

        return context, target

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding vector for a given word.

        Returns:
            np.ndarray: Embedding vector of shape (embed_dim, 1) for the given word.

        Raises:
            ValueError: If the word is not in the vocabulary.
        """
        word_idx = self.word2idx.get(word, None)

        if word_idx is None:
            raise ValueError(f"Word '{word}' not in vocabulary.")

        return self.embedding_matrix[word_idx].reshape(-1, 1)

    def context_word_average(self, context_words: Sequence[str]) -> np.ndarray:
        """Average context word embeddings into a single CBOW hidden representation."""
        embeddings = np.array([self.get_embedding(word) for word in context_words])
        return np.mean(embeddings, axis=0)

    def cross_entropy_loss(self, h_hat: np.ndarray, target_word_context: np.ndarray) -> float:
        """Compute the cross-entropy loss for a single training example."""
        return -target_word_context.T @ h_hat + np.log(np.sum(np.exp(self.context_matrix.T @ h_hat)))



    def forward_pass(self, context_words: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run CBOW forward pass to produce vocabulary scores from context words."""
        # Average of context embeddings
        h_hat = self.context_word_average(context_words)

        # Scores for all target words
        predicted_scores = self.context_matrix.T @ h_hat  # Shape: (vocab_size, 1)

        # Probabilities
        probs = softmax(predicted_scores)  # Shape: (vocab_size, 1)

        return h_hat, predicted_scores, probs

    def predict(self, context_words: Sequence[str], top_k: int = 3) -> str:
        """Predict the target word given context words."""
        _, _, probs = self.forward_pass(context_words)
        print(probs.flatten())
        top_k_indices = np.argsort(probs.flatten())[::-1][:top_k]
        print(top_k_indices)
        return [{self.idx2word[idx] : probs[idx].item()} for idx in top_k_indices]

# Training Hyperparameters
learning_rate: float = 0.05
epochs: int = 150

# Model hyperparameters
window_size: int = 2
embed_dim: int = 50  # Dimension of the embedding vectors

word2vec = Word2VecCBOW(sentences, window_size=window_size, embed_dim=embed_dim)
vocab, word2idx, idx2word = word2vec.vocab, word2vec.word2idx, word2vec.idx2word
vocab_size: int = len(vocab)
print(f"Vocabulary Size: {vocab_size}")
print(f"Sample vocab: {vocab[:10]}")
context, target = word2vec.get_cbow_example(word2vec.sentences_tokens[0], 3)
print(f"CBOW Example (Context, Target):\n{context},\n{target}")
print(f"Context embeddings:")
for context_word in context:
    embedding = word2vec.get_embedding(context_word)
    print(f"Embedding for '{context_word}':\n{embedding[:5]}...")

embed_avg = word2vec.context_word_average(context)
print(f"Average embedding for context {context}:\n{embed_avg[:5]}...")  # Print first 5 dimensions for brevity

h_hat, predicted_scores, probs = word2vec.forward_pass(context)
print(f"Predicted scores for all target words:\n{predicted_scores[:5]}...")  # Print first 5 scores for brevity
print(f"Predicted probabilities for all target words:\n{probs[:5]}...")  # Print first 5 probabilities for brevity

pred = word2vec.predict(context)
print(f"Predicted target word for context {context}: '{pred}'")

# for epoch in range(epochs):
#     total_loss = 0
#     for sentence in sentences:
#         tokens = sentence.split()
#         for i, target_word in enumerate(tokens):
#             t_idx = word2idx[target_word]

#             # --- CONTEXT AGGREGATION ---
#             context_indices = []
#             for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
#                 if i != j:
#                     context_indices.append(word2idx[tokens[j]])

#             if not context_indices: continue

#             # Average of context vectors (The "Hidden" layer in CBOW)
#             v_hat = np.mean(embedding_matrix[context_indices], axis=0)

#             # --- FORWARD PASS (Negative Sampling) ---
#             # Positive sample: Target word
#             z_pos = np.dot(v_hat, context_matrix[t_idx])
#             p_pos = sigmoid(z_pos)

#             # Negative samples
#             neg_indices = np.random.choice(vocab_size, n_negs, replace=False)
#             z_neg = np.dot(context_matrix[neg_indices], v_hat)
#             p_neg = sigmoid(z_neg)

#             # Loss
#             loss = -np.log(p_pos + 1e-10) - np.sum(np.log(1 - p_neg + 1e-10))
#             total_loss += loss

#             # --- GRADIENT COMPUTATION ---
#             err_pos = p_pos - 1
#             err_neg = p_neg - 0

#             # Gradient for context_matrix (Target/Negative vectors)
#             grad_context_matrix_pos = err_pos * v_hat
#             grad_context_matrix_neg = np.outer(err_neg, v_hat)

#             # Gradient for v_hat (The mean vector)
#             grad_v_hat = err_pos * context_matrix[t_idx] + np.dot(err_neg, context_matrix[neg_indices])

#             # --- PARAMETER UPDATES ---
#             # Update context_matrix
#             context_matrix[t_idx] -= learning_rate * grad_context_matrix_pos
#             context_matrix[neg_indices] -= learning_rate * grad_context_matrix_neg

#             # Update embedding_matrix: The gradient is distributed equally across all context words
#             # because of the mean operation (grad_v_hat / len(context_indices))
#             grad_embedding_matrix = grad_v_hat / len(context_indices)
#             for idx in context_indices:
#                 embedding_matrix[idx] -= learning_rate * grad_embedding_matrix

#     if epoch % 30 == 0:
#         print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
