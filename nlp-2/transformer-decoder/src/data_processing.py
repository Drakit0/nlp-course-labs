
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


def load_and_preprocess_data(
    filepath: str, alphabet: str, start_token: str = "-", end_token: str = "."
) -> List[str]:
    with open(filepath, "r") as file:
        lines: List[str] = file.read().splitlines()

    names: List[str] = []
    for line in lines:
        parts: List[str] = line.strip().split()
        if len(parts) < 3:
            continue  # Skip lines that don't have enough parts
        word: str = " ".join(parts[:-2]).lower()  # Joining all parts except the last two

        word = "".join([char for char in word if char in alphabet])

        name = start_token + word + end_token
        names.append(name)

    return names


class CharTokenizer:

    def __init__(self, alphabet: str, start_token: str = "-", end_token: str = "."):
        self.alphabet = sorted(list(set(alphabet)))
        self.char2idx = {char: idx + 3 for idx, char in enumerate(self.alphabet)}
        self.char2idx[start_token] = 1
        self.char2idx[end_token] = 2
        self.idx2char = {idx + 3: char for idx, char in enumerate(self.alphabet)}
        self.idx2char[1] = start_token
        self.idx2char[2] = end_token
        self.vocab_size = len(self.alphabet) + 3

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[char] for char in text]

    def decode(self, indices: List[int]) -> str:
        return ''.join([self.idx2char[idx] for idx in indices])


class NameDataset(Dataset):

    def __init__(self, encoded_names: List[List[int]]):
        self.encoded_names = encoded_names

    def __len__(self) -> int:
        return len(self.encoded_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_name = self.encoded_names[idx]
        input_ids = torch.tensor(encoded_name[:-1], dtype=torch.long)
        target_ids = torch.tensor(encoded_name[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    target_ids_padded = torch.nn.utils.rnn.pad_sequence(
        target_ids, batch_first=True, padding_value=0
    )

    return input_ids_padded, target_ids_padded