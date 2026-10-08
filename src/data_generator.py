"""
data_generator.py
------------------
The full set of (image, partial-caption -> next-word) training pairs for
Flickr8k is far too large to materialize in RAM at once (each of the ~8,000
images expands into dozens of teacher-forcing examples, one per caption
prefix length). This module streams batches instead, via a Keras
`Sequence` subclass, so training only ever holds one batch in memory.

Two-input format matches the model in model.py:
    X1 = image feature vectors, shape (batch, FEATURE_DIM)
    X2 = padded partial caption sequences, shape (batch, max_len)
    y  = one-hot next word, shape (batch, vocab_size)
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical


class CaptionDataGenerator(Sequence):
    """
    Yields (X1, X2), y batches for teacher-forced training.

    Parameters
    ----------
    image_ids : list[str]
        Which images (already split into train/val/test upstream) to draw from.
    captions_map : dict[str, list[str]]
        image_id -> list of cleaned captions (already wrapped with startseq/endseq).
    features_map : dict[str, np.ndarray]
        image_id -> CNN feature vector.
    tokenizer : keras Tokenizer
    max_len : int
    vocab_size : int
    batch_size : int
    """

    def __init__(self, image_ids, captions_map, features_map, tokenizer,
                 max_len, vocab_size, batch_size=64, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.captions_map = captions_map
        self.features_map = features_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Flatten to one (image_id, caption) pair per row; only keep images
        # that actually have a cached feature vector.
        self.pairs = [
            (img_id, cap)
            for img_id in image_ids
            if img_id in self.features_map
            for cap in self.captions_map.get(img_id, [])
        ]
        self.indices = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.pairs) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X1, X2, y = [], [], []
        for i in batch_idx:
            img_id, caption = self.pairs[i]
            feature = self.features_map[img_id]
            seq = self.tokenizer.texts_to_sequences([caption])[0]

            # Teacher forcing: for a caption of length L, generate L-1
            # (prefix -> next word) training examples.
            for t in range(1, len(seq)):
                in_seq = seq[:t]
                out_word = seq[t]
                in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_word)

        X1 = np.array(X1)
        X2 = np.array(X2)
        y = to_categorical(y, num_classes=self.vocab_size)
        return (X1, X2), y
