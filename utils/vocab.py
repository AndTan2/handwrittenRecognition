from config import config

class Vocabulary:
    def __init__(self):
        self.char2idx = {char: idx for idx, char in enumerate(config.VOCAB)}
        self.idx2char = {idx: char for idx, char in enumerate(config.VOCAB)}
        self.size = len(config.VOCAB)
        self.blank_char = len(config.VOCAB)  # CTC blank character index

    def text_to_indices(self, text):
        """Convert text string to list of character indices"""
        return [self.char2idx.get(char, self.blank_char) for char in text]

    def indices_to_text(self, indices):
        """Convert list of character indices to text string"""
        return ''.join([self.idx2char.get(idx, '') for idx in indices if idx in self.idx2char])

    def get_vocab(self):
        """Return the vocabulary string"""
        return config.VOCAB

# Global vocabulary instance
vocab = Vocabulary()