from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch
import numpy as np

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    with open(infile) as file:
        text = file.read()  # Read the entire file

    with open(infile) as file:
        text = file.read()
    
    tokens: List[str] = tokenize(text)

    return tokens

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    word_counts: Counter = Counter(words)
    sorted_vocab: List[int] = sorted(word_counts, key=word_counts.get, reverse=True)
    
    int_to_vocab: Dict[int, str] = {pos: word for pos, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: pos for pos, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    int_words: List[int] = [vocab_to_int[word] for word in words] # I don't exactly know where to apply this
    
    freqs: Dict[str, float] = {word:0 for word in words}
    for word in words:
        freqs[word] += 1/len(words)
        
                    
    train_words: List[str] = []
    for word, freq in freqs.items():
        p_drop = np.clip(1 - np.sqrt(threshold / freq), 0, 1) # maintain the probability
        
        if np.random.rand() > p_drop:
            train_words.append(vocab_to_int[word])
            
    return train_words, freqs

def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    target_words: List[str] = []
    random_window_size = np.random.randint(1, window_size + 1)
    
    for i in range(idx - random_window_size, idx + random_window_size + 1):
        if i != idx and 0 <= i < len(words):
            target_words.append(words[i])

    return target_words

def get_batches(words: List[int], batch_size: int, window_size: int = 5) -> Tuple[List[int], List[int]]:

    for batch_start in range(0, len(words), batch_size):
        inputs, targets = [], []
        
        for center_idx in range(batch_start, min(batch_start + batch_size, len(words))):
            random_window_size = np.random.randint(1, window_size + 1)
            start = max(0, center_idx - random_window_size)
            
            for left_idx in range(start, center_idx):
                inputs.append(center_idx)
                targets.append(left_idx)
            
            end = min(len(words), center_idx + random_window_size + 1)
            
            for right_idx in range(center_idx + 1, end):
                inputs.append(center_idx)
                targets.append(right_idx)
                
        yield inputs, targets

def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):

    embedding = embedding.to(device)
    embedding_weights = embedding.weight
    
    valid_examples: torch.Tensor = torch.tensor(np.array(np.random.choice(valid_window, valid_size, replace=False)), device=device)
    similarities: torch.Tensor = torch.zeros(valid_size, len(embedding_weights))
    
    for i, valid_word in enumerate(valid_examples):
        expand_embed = embedding_weights[valid_word].view(1, -1)
        similarities[i] = torch.mm(expand_embed, embedding_weights.T) / (torch.norm(expand_embed) * torch.norm(embedding_weights, dim=1))

    return valid_examples, similarities