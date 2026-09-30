import torch
import numpy as np
from typing import Dict, List


def bigrams_count_to_probabilities(
    bigram_counts: torch.Tensor, smooth_factor: int = 0
) -> torch.Tensor:
    
    for i in range(bigram_counts.shape[0]):
        
        counts_sum:int = sum(bigram_counts[i, k] for k in range(bigram_counts.shape[0]))
        
        for j in range(bigram_counts.shape[0]):
            bigram_counts[i, j] =  (bigram_counts[i, j] + smooth_factor) / (counts_sum + bigram_counts.shape[0]*smooth_factor)

    return bigram_counts


def calculate_neg_mean_log_likelihood(
    words: List[str],
    bigram_probabilities: torch.tensor,
    char_to_index: Dict[str, int],
    start_token: str = "<S>",
    end_token: str = "<E>",
) -> float:
    
    total_log_likelihood: torch.tensor = torch.zeros(1, len(words))

    for i in range(len(words)):
        total_log_likelihood[0, i] = calculate_log_likelihood(words[i], bigram_probabilities, char_to_index, start_token, end_token)

    mean_log_likelihood: float = torch.mean(total_log_likelihood).item() * -1

    return mean_log_likelihood


def sample_next_character(
    current_char_index: int,
    probability_distribution: torch.Tensor,
    idx_to_char: Dict[int, str],
) -> str:
    current_probs: torch.Tensor[float] = probability_distribution[current_char_index, :]

    next_char_index: int = torch.multinomial(current_probs, 1).item()

    next_char: str = idx_to_char[next_char_index]
    
    return next_char


def generate_name(
    start_token: str,
    end_token: str,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    bigram_probabilities: torch.Tensor,
    max_length: int = 15,
) -> str:
    current_char: str = start_token
    generated_name: str = current_char

    for i in range(max_length):
        
        current_char: str = generated_name[i]
        
        if current_char != end_token:
            generated_name += sample_next_character(char_to_idx[current_char], bigram_probabilities, idx_to_char)
            
        else:
            break
        
    if generated_name[-1] != end_token:
        generated_name += end_token
        
    return generated_name

def calculate_log_likelihood(
    word: str,
    bigram_probabilities: torch.Tensor,
    char_to_index: Dict[str, int],
    start_token: str = "<S>",
    end_token: str = "<E>",
) -> torch.Tensor:
    processed_word: str = start_token + word + end_token

    log_likelihood: torch.tensor = torch.zeros(1, len(processed_word) - 1)

    for i in range(len(processed_word) - 1):
        
        likelihood = bigram_probabilities[char_to_index[processed_word[i]], char_to_index[processed_word[i+1]]]
        
        if likelihood > 0: 
            log_likelihood[0, i] = np.log(likelihood)
            
        else:
            log_likelihood[0, i] = 0

    log_likelihood: torch.tensor = torch.sum(log_likelihood)
    
    return log_likelihood

if __name__ == "__main__":
    pass
