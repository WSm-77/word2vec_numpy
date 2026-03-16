import re

import pandas as pd


class DatasetLoader:
    """Load and preprocess parquet text data for Word2Vec training."""

    def __init__(self, parquet_path: str = "", column_name: str = "text"):
        """
        Initialize the dataset loader.

        Args:
            parquet_path: Path to the parquet dataset file.
            column_name: Name of the text column to use.
        """
        self.parquet_path = parquet_path
        self.column_name = column_name
        self.df = self.load_data(parquet_path, column_name)

    def clean_text(self, text: str) -> str:
        """
        Normalize raw text for Word2Vec training.

        Args:
            text: Raw input text.

        Returns:
            str: Lowercased and cleaned text with URLs and non-letter noise removed.
        """
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_data(self, parquet_path: str, column_name: str = "text") -> pd.DataFrame:
        """
        Load a parquet dataset and validate the selected text column.

        Args:
            parquet_path: Path to the parquet dataset file.
            column_name: Name of the text column to validate.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            ValueError: If the requested column is missing or not string-like.
        """
        df = pd.read_parquet(parquet_path)

        if column_name not in df.columns:
            raise ValueError(f"Specified column \"{column_name}\" not found in the dataset")

        if not pd.api.types.is_string_dtype(df[column_name]):
            raise ValueError(f"Specified column \"{column_name}\" must be of type string/object")

        return df

    def preprocess_data(self, df: pd.DataFrame, column_name: str = "text") -> list[str]:
        """
        Clean a dataframe text column and filter out short sentences.

        Args:
            df: Dataset containing the raw text column.
            column_name: Name of the text column to preprocess.

        Returns:
            list[str]: Cleaned sentences with at least two tokens.
        """
        clean_text = df[column_name].dropna().map(self.clean_text)
        sentences = [s for s in clean_text.tolist() if len(s.split()) >= 2]

        print(f"Rows in parquet: {len(df):,}")
        print(f"Usable sentences: {len(sentences):,}")

        return sentences

    def load_and_preprocess_data(self, parquet_path: str, column_name: str = "text") -> list[str]:
        """
        Load text data from a parquet file and preprocess it for Word2Vec training.

        Args:
            parquet_path: Path to the parquet dataset file.
            column_name: Name of the text column to preprocess.

        Returns:
            list[str]: Cleaned sentences ready for training.
        """

        if self.df is None:
            self.parquet_path = parquet_path
            self.column_name = column_name
            self.df = self.load_data(self.parquet_path, self.column_name)

        sentences = self.preprocess_data(self.df, self.column_name)

        return sentences
