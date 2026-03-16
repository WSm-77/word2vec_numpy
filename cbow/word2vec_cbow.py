import numpy as np
import time
from typing import Dict, List, Sequence, Tuple
from numbers import Number
from sortedcontainers import SortedSet

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
        """
        Initialize corpus data, vocabulary mappings, and CBOW parameter matrices.

        Args:
            sentences: Input corpus represented as a sequence of raw sentences.
            window_size: Number of context words to include on each side of a target word.
            embed_dim: Dimensionality of learned word embeddings.
        """
        self.sentences: Sequence[str] = sentences
        self.sentences_words: List[List[str]] = [s.split() for s in sentences]
        self.window_size: int = window_size
        self.vocab, self.word2idx, self.idx2word = self.get_vocab(sentences)
        self.sentences_ids: List[List[int]] = [self.words_to_ids(words) for words in self.sentences_words]
        self.embedding_matrix: np.ndarray = self.initialize_embeddings_matrix(len(self.vocab), embed_dim)
        self.context_matrix: np.ndarray = self.initialize_context_matrix(len(self.vocab), embed_dim)

    def get_vocab(self, sentences: Sequence[str]) -> Tuple[SortedSet[str], Dict[str, int], Dict[int, str]]:
        """
        Build sorted vocabulary and index mappings from the input sentences.

        Args:
            sentences: Input corpus represented as a sequence of raw sentences.

        Returns:
            Tuple[SortedSet[str], Dict[str, int], Dict[int, str]]: Sorted vocabulary,
            word-to-index mapping, and index-to-word mapping.
        """
        words = " ".join(sentences).split()
        vocab = SortedSet(words)
        word2idx = {w: i for i, w in enumerate(vocab)}
        idx2word = {i: w for w, i in word2idx.items()}
        return vocab, word2idx, idx2word

    def words_to_ids(self, words: Sequence[str]) -> List[int]:
        """
        Convert a sequence of words into vocabulary indices.

        Args:
            words: Token sequence to convert.

        Returns:
            List[int]: Vocabulary indices for the provided words.

        Raises:
            ValueError: If any word is not present in the vocabulary.
        """
        word_ids: List[int] = []

        for word in words:
            word_idx = self.word2idx.get(word)
            if word_idx is None:
                raise ValueError(f"Word '{word}' not in vocabulary.")
            word_ids.append(word_idx)

        return word_ids

    def initialize_embeddings_matrix(self, vocab_size: int, embed_dim: int) -> np.ndarray:
        """
        Initialize the input embedding matrix.

        Args:
            vocab_size: Number of unique words in the vocabulary.
            embed_dim: Dimensionality of each embedding vector.

        Returns:
            np.ndarray: Randomly initialized embedding matrix of shape
            (vocab_size, embed_dim).
        """
        return np.random.randn(vocab_size, embed_dim) * 0.01

    def initialize_context_matrix(self, vocab_size: int, embed_dim: int) -> np.ndarray:
        """
        Initialize the output context matrix.

        Args:
            vocab_size: Number of unique words in the vocabulary.
            embed_dim: Dimensionality of each embedding vector.

        Returns:
            np.ndarray: Randomly initialized context matrix of shape
            (embed_dim, vocab_size).
        """
        return np.random.randn(embed_dim, vocab_size) * 0.01

    def get_cbow_example(self, sentence_words: List[str], target_word_idx: int) -> Tuple[List[str], str]:
        """
        Generate a single CBOW training example.

        Args:
            sentence_words: Tokenized sentence from the training corpus.
            target_word_idx: Index of the target word within the sentence.

        Returns:
            Tuple[List[str], str]: Context words surrounding the target word and
            the target word itself.
        """
        target = sentence_words[target_word_idx]
        context: List[str] = []

        for offset in range(-self.window_size, self.window_size + 1):
            if offset == 0:  # Skip the target word itself
                continue

            context_idx = target_word_idx + offset
            if 0 <= context_idx < len(sentence_words):
                context.append(sentence_words[context_idx])

        return context, target

    def get_cbow_example_ids(self, sentence_word_ids: Sequence[int], target_word_idx: int) -> Tuple[List[int], int]:
        """
        Generate a single CBOW training example using token ids.

        Args:
            sentence_word_ids: Tokenized sentence represented by vocabulary indices.
            target_word_idx: Index of the target token within the sentence.

        Returns:
            Tuple[List[int], int]: Context token ids and the target token id.
        """
        target_id = sentence_word_ids[target_word_idx]
        context_ids: List[int] = []

        for offset in range(-self.window_size, self.window_size + 1):
            if offset == 0:
                continue

            context_idx = target_word_idx + offset
            if 0 <= context_idx < len(sentence_word_ids):
                context_ids.append(sentence_word_ids[context_idx])

        return context_ids, target_id

    def get_cbow_examples(self) -> Tuple[List[List[str]], List[str]]:
        """
        Generate all CBOW training examples from the corpus.

        Returns:
            Tuple[List[List[str]], List[str]]: Parallel lists containing context
            word sequences and their corresponding target words.
        """
        contexts: List[List[str]] = []
        targets: List[str] = []

        for sentence_words in self.sentences_words:
            for i in range(len(sentence_words)):
                context, target = self.get_cbow_example(sentence_words, i)

                if context:
                    contexts.append(context)
                    targets.append(target)

        return contexts, targets

    def get_cbow_examples_ids(self) -> Tuple[List[List[int]], List[int]]:
        """
        Generate all CBOW training examples from the corpus using token ids.

        Returns:
            Tuple[List[List[int]], List[int]]: Parallel lists containing context
            token-id sequences and their corresponding target token ids.
        """
        contexts: List[List[int]] = []
        targets: List[int] = []

        for sentence_word_ids in self.sentences_ids:
            for i in range(len(sentence_word_ids)):
                context_ids, target_id = self.get_cbow_example_ids(sentence_word_ids, i)

                if context_ids:
                    contexts.append(context_ids)
                    targets.append(target_id)

        return contexts, targets

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

    def get_embedding_by_id(self, word_idx: int) -> np.ndarray:
        """
        Retrieve the embedding vector for a given vocabulary index.

        Args:
            word_idx: Vocabulary index of the target word.

        Returns:
            np.ndarray: Embedding vector of shape (embed_dim, 1).
        """
        return self.embedding_matrix[word_idx].reshape(-1, 1)

    def context_word_average(self, context_words: Sequence[str]) -> np.ndarray:
        """
        Convert context words into a single averaged embedding.

        Args:
            context_words: Context words surrounding a target word.

        Returns:
            np.ndarray: Averaged context embedding of shape (embed_dim, 1).
        """
        embeddings = np.array([self.get_embedding(word) for word in context_words])
        return np.mean(embeddings, axis=0)

    def context_word_average_by_ids(self, context_word_ids: Sequence[int]) -> np.ndarray:
        """
        Convert context token ids into a single averaged embedding.

        Args:
            context_word_ids: Context words represented by vocabulary indices.

        Returns:
            np.ndarray: Averaged context embedding of shape (embed_dim, 1).
        """
        context_indices = np.asarray(context_word_ids, dtype=np.int64)
        embeddings = self.embedding_matrix[context_indices]
        return np.mean(embeddings, axis=0, keepdims=True).T

    def cross_entropy_loss(self, h_hat: np.ndarray, target_word_context: np.ndarray) -> float:
        """
        Compute the cross-entropy loss for a single training example.

        Args:
            h_hat: Averaged context embedding of shape (embed_dim, 1).
            target_word_context: One-hot target representation of shape
            (vocab_size, 1).

        Returns:
            float: Scalar cross-entropy loss value.
        """
        return -target_word_context.T @ h_hat + np.log(np.sum(np.exp(self.context_matrix.T @ h_hat)))

    def score_target_words(self, h_hat: np.ndarray) -> np.ndarray:
        """
        Compute raw vocabulary scores for a context embedding.

        Args:
            h_hat: Averaged context embedding of shape (embed_dim, 1).

        Returns:
            np.ndarray: Raw scores for each target word in the vocabulary of shape (vocab_size, 1).
        """
        return self.context_matrix.T @ h_hat

    def forward(self, context_words: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the CBOW forward pass for a context window.

        Args:
            context_words: Context words surrounding a target word.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Averaged context embedding and softmax
            probabilities.
        """
        # Average of context embeddings
        h_hat = self.context_word_average(context_words)

        # Scores for all target words
        predicted_scores = self.score_target_words(h_hat)  # Shape: (vocab_size, 1)

        # Probabilities
        probs = softmax(predicted_scores)  # Shape: (vocab_size, 1)

        return h_hat, probs

    def forward_ids(self, context_word_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the CBOW forward pass for context token ids.

        Args:
            context_word_ids: Context words represented by vocabulary indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Averaged context embedding and softmax
            probabilities over the vocabulary.
        """
        h_hat = self.context_word_average_by_ids(context_word_ids)
        predicted_scores = self.score_target_words(h_hat)
        probs = softmax(predicted_scores)
        return h_hat, probs

    def predict(self, context_words: Sequence[str], top_k: int = 3) -> List[Dict[str, float]]:
        """
        Predict the k most likely target words for a given context.

        Args:
            context_words: Context words surrounding a missing target word.
            top_k: Number of highest-probability predictions to return.

        Returns:
            List[Dict[str, float]]: List of dictionaries mapping predicted words
            to their probabilities.
        """
        _, probs = self.forward(context_words)
        top_k_indices = np.argsort(probs.flatten())[::-1][:top_k]

        return [{self.idx2word[idx] : probs[idx].item()} for idx in top_k_indices]

    def compute_loss(self, probs: np.ndarray, target_word: str) -> float:
        """
        Compute cross-entropy loss for one training example.

        Args:
            probs: Softmax probabilities over the vocabulary.
            target_word: Ground-truth target word.

        Returns:
            float: Scalar loss value for the example.
        """
        target_idx = self.word2idx[target_word]
        loss = -np.log(float(probs[target_idx, 0]) + 1e-10)

        return loss

    def compute_loss_by_id(self, probs: np.ndarray, target_idx: int) -> float:
        """
        Compute cross-entropy loss for one training example using a target id.

        Args:
            probs: Softmax probabilities over the vocabulary.
            target_idx: Vocabulary index of the ground-truth target word.

        Returns:
            float: Scalar loss value for the example.
        """
        return -np.log(float(probs[target_idx, 0]) + 1e-10)

    def sample_negative_indices(self, target_word_idx: int, n_samp: int = 32) -> np.ndarray:
        """
        Sample negative target indices for negative sampling training using a vectorized approach.

        Args:
            target_word_idx: Vocabulary index of the positive target word.
            n_samp: Number of negative samples to draw.

        Returns:
            np.ndarray: Sampled negative vocabulary indices.
        """
        raw_samps = np.random.randint(0, len(self.vocab), size=n_samp)
        neg_inds = raw_samps[raw_samps != target_word_idx]
        return neg_inds

    def train_example_negative_sampling(
        self,
        context_word_ids: Sequence[int],
        target_idx: int,
        learning_rate: float = 0.01,
        num_negative_samples: int = 5,
    ) -> float:
        """
        Train the model on a single CBOW example using negative sampling.

        Args:
            context_word_ids: Context words represented by vocabulary indices.
            target_idx: Vocabulary index of the ground-truth target word.
            learning_rate: Gradient descent learning rate.
            num_negative_samples: Number of negative samples to use.

        Returns:
            float: Negative-sampling loss value for the training example.
        """
        if not context_word_ids:
            return 0.0

        context_indices = np.asarray(context_word_ids, dtype=np.int64)
        h_hat = self.embedding_matrix[context_indices].mean(axis=0)

        positive_output = self.context_matrix[:, target_idx].copy()
        positive_score = float(h_hat @ positive_output)
        positive_prob = float(sigmoid(positive_score))

        negative_indices = self.sample_negative_indices(target_idx, num_negative_samples)
        negative_outputs = self.context_matrix[:, negative_indices].copy() if negative_indices.size else np.empty((h_hat.shape[0], 0))
        negative_scores = h_hat @ negative_outputs if negative_indices.size else np.empty(0, dtype=np.float64)
        negative_probs = sigmoid(negative_scores) if negative_indices.size else np.empty(0, dtype=np.float64)
        negative_probs = np.asarray(negative_probs, dtype=np.float64)

        loss = -np.log(positive_prob + 1e-10)
        if negative_indices.size:
            loss -= np.sum(np.log(1.0 - negative_probs + 1e-10))

        positive_error = positive_prob - 1.0
        negative_errors = negative_probs

        grad_h_hat = positive_error * positive_output
        if negative_indices.size:
            grad_h_hat += negative_outputs @ negative_errors

        self.context_matrix[:, target_idx] -= learning_rate * positive_error * h_hat
        if negative_indices.size:
            negative_updates = np.outer(negative_errors, h_hat)
            np.add.at(self.context_matrix.T, negative_indices, -learning_rate * negative_updates)

        context_gradient = grad_h_hat / len(context_indices)
        np.add.at(self.embedding_matrix, context_indices, -learning_rate * context_gradient)

        return float(loss)

    def normalize_context_ids(self, context_words: Sequence[str] | Sequence[int]) -> List[int]:
        """
        Normalize context inputs into vocabulary indices.

        Args:
            context_words: Context words represented as strings or token ids.

        Returns:
            List[int]: Normalized context token ids.
        """
        if not context_words:
            return []

        first_item = context_words[0]
        if isinstance(first_item, str):
            return self.words_to_ids(context_words)

        return [int(word_idx) for word_idx in context_words]

    def normalize_target_idx(self, target_word: str | Number) -> int:
        """
        Normalize a target input into a vocabulary index.

        Args:
            target_word: Target word represented as a string or token id.

        Returns:
            int: Vocabulary index of the target word.
        """
        if isinstance(target_word, str):
            target_idx = self.word2idx.get(target_word)
            if target_idx is None:
                raise ValueError(f"Word '{target_word}' not in vocabulary.")
            return target_idx

        return int(target_word)

    def compute_gradients(
        self,
        context_words: Sequence[str],
        h_hat: np.ndarray,
        probs: np.ndarray,
        target_word: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for the embedding and context matrices.

        Args:
            context_words: Context words surrounding the target word.
            h_hat: Averaged context embedding of shape (embed_dim, 1).
            probs: Softmax probabilities over the vocabulary.
            target_word: Ground-truth target word.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Gradients for the context
            word embeddings, the averaged embedding, and the context matrix.
        """
        target_idx = self.word2idx[target_word]

        # Gradient of loss w.r.t. predicted scores
        grad_scores = probs.copy()
        grad_scores[target_idx] -= 1  # dL/dz

        # Gradient w.r.t. context_matrix (W')
        grad_context_matrix = h_hat @ grad_scores.T  # Shape: (embed_dim, vocab_size)

        # Gradient w.r.t. h_hat (the average embedding)
        grad_h_hat = self.context_matrix @ grad_scores  # Shape: (embed_dim, 1)

        # Gradient for each context embedding row (same vector distributed by mean op)
        grad_word_embeddings = grad_h_hat / len(context_words)

        return grad_word_embeddings, grad_h_hat, grad_context_matrix

    def backward(
        self,
        context_words: Sequence[str],
        target_word: str,
        h_hat: np.ndarray,
        probs: np.ndarray,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Perform backpropagation and update model parameters.

        Args:
            context_words: Context words surrounding the target word.
            target_word: Ground-truth target word.
            h_hat: Averaged context embedding of shape (embed_dim, 1).
            probs: Softmax probabilities over the vocabulary.
            learning_rate: Gradient descent learning rate.
        """
        grad_word_embeddings, _, grad_context_matrix = self.compute_gradients(context_words, h_hat, probs, target_word)

        # Update context matrix
        self.context_matrix -= learning_rate * grad_context_matrix

        # Update each context embedding with the same distributed gradient vector.
        grad_context_vector = grad_word_embeddings.ravel()
        for word in context_words:
            word_idx = self.word2idx[word]
            self.embedding_matrix[word_idx] -= learning_rate * grad_context_vector

    def train_example(
        self,
        context_words: Sequence[str] | Sequence[int],
        target_word: str | Number,
        learning_rate: float = 0.01,
        num_negative_samples: int = 5,
    ) -> float:
        """
        Train the model on a single CBOW example.

        Args:
            context_words: Context words surrounding the target word.
            target_word: Ground-truth target word.
            learning_rate: Gradient descent learning rate.
            num_negative_samples: Number of negative samples to use during
            training.

        Returns:
            float: Loss value for the training example.
        """
        context_word_ids = self.normalize_context_ids(context_words)
        target_idx = self.normalize_target_idx(target_word)
        return self.train_example_negative_sampling(
            context_word_ids,
            target_idx,
            learning_rate=learning_rate,
            num_negative_samples=num_negative_samples,
        )

    def train(
        self,
        contexts: List[Sequence[str] | Sequence[int]],
        target_words: List[str | Number],
        epochs: int = 100,
        learning_rate: float = 0.01,
        max_epochs_without_loss_improvement: int = 10,
        timeout: float | None = None,
        num_negative_samples: int = 5,
    ) -> None:
        """
        Train the CBOW model across multiple epochs.

        Args:
            contexts: List of context-word sequences for training.
            target_words: List of target words aligned with the contexts.
            epochs: Maximum number of training epochs.
            learning_rate: Gradient descent learning rate.
            max_epochs_without_loss_improvement: Early stopping patience based on
            epoch loss.
            timeout: Maximum wall-clock time (in seconds) to spend training.
            If None or <= 0, timeout is disabled.
            num_negative_samples: Number of negative samples to draw per update.
        """
        losses = []
        epochs_without_loss_improvement = 0
        best_loss = float("inf")
        start_time = time.perf_counter()
        timed_out = False
        context_id_batches = [self.normalize_context_ids(context_words) for context_words in contexts]
        target_ids = [self.normalize_target_idx(target_word) for target_word in target_words]

        for epoch in range(epochs):
            total_loss = 0.0
            epoch_start_time = time.perf_counter()

            for context_word_ids, target_idx in zip(context_id_batches, target_ids):
                if timeout is not None and timeout > 0 and (time.perf_counter() - start_time) >= timeout:
                    timed_out = True
                    elapsed = time.perf_counter() - start_time
                    print(f"Timeout reached after {elapsed:.2f}s. Stopping training.")
                    break

                example_loss = self.train_example_negative_sampling(
                    context_word_ids,
                    target_idx,
                    learning_rate=learning_rate,
                    num_negative_samples=num_negative_samples,
                )
                total_loss += example_loss

            if timed_out:
                break

            epoch_loss = total_loss / len(contexts)
            losses.append(epoch_loss)

            epoch_end_time = time.perf_counter()
            print(f"Epoch {epoch + 1}/{epochs} Epoch loss: {epoch_loss} Time: {epoch_end_time - epoch_start_time:.2f}s")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_loss_improvement = 0
            else:
                epochs_without_loss_improvement += 1

            if epochs_without_loss_improvement >= max_epochs_without_loss_improvement:
                print("Early stopping triggered.")
                break
