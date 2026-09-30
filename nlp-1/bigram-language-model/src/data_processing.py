from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def load_and_preprocess_data(
    filepath: str, start_token: str = "!", end_token: str = "."
) -> List[Tuple[str, str]]:
    with open(filepath, "r") as file:
        lines: List[str] = file.read().splitlines()

    bigrams: List[Tuple[str, str]] = []
    
    for line in lines:
        words = line.split(" ")[:-2]
        word = " ".join(words)
        word = (start_token + word + end_token).lower()
        
        for i in range(len(word)-1):
            bigrams.append(tuple(word[i] + word[i+1]))
        
    return bigrams


def char_to_index(alphabet: str, start_token: str, end_token: str) -> Dict[str, int]:
    char_to_idx: Dict[str, int] = {}
    
    complete_alphabet = start_token + alphabet + end_token
    
    for i, char in enumerate(complete_alphabet):
        char_to_idx[char] = i

    return char_to_idx


def index_to_char(char_to_index: Dict[str, int]) -> Dict[int, str]:
    idx_to_char: Dict[int, str] = {}
    
    for key, val in char_to_index.items():
        idx_to_char[val] = key

    return idx_to_char


def count_bigrams(
    bigrams: List[Tuple[str, str]], char_to_idx: Dict[str, int]
) -> torch.Tensor:

    bigram_counts: torch.Tensor = torch.zeros(len(char_to_idx), len(char_to_idx))

    for bigram in bigrams:
        if bigram[0] in char_to_idx.keys() and bigram[1] in char_to_idx.keys():
            bigram_counts[char_to_idx[bigram[0]], char_to_idx[bigram[1]]] += 1

    return bigram_counts


def plot_bigram_counts(bigram_counts: torch.Tensor, idx_to_char: Dict):
    plt.figure(figsize=(16, 16))
    plt.imshow(bigram_counts, cmap="Blues")

    for i in range(bigram_counts.shape[0]):
        for j in range(bigram_counts.shape[1]):
            char_str: str = idx_to_char[i] + idx_to_char[j]
            plt.text(j, i, char_str, ha="center", va="bottom", color="gray")
            plt.text(
                j, i, bigram_counts[i, j].item(), ha="center", va="top", color="gray"
            )

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    file_path: str = "data/nombres_raw.txt"

    alphabet: str = "abcdefghijklmnopqrstuvwxyz "

    start_token: str = "-"
    end_token: str = "."

    char_to_idx: Dict[str, int] = char_to_index(alphabet, start_token, end_token)

    bigrams: List[Tuple[str, str]] = load_and_preprocess_data(
        file_path, start_token=start_token, end_token=end_token
    )

    bigram_counts: torch.Tensor = count_bigrams(bigrams, char_to_idx)

    idx_to_char: Dict[int, str] = index_to_char(char_to_idx)

    plot_bigram_counts(bigram_counts, idx_to_char)
