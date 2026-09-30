from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    examples: List[SentimentExample] = []
    with open(infile, "r") as file:
        raw_file = file.read()
        
    cleaned_file = raw_file.replace(".", "").replace(",", "").replace(":", "").replace(";", "")
    lines = cleaned_file.split("\n")
    processed_lines = [line for line in lines if len(line) > 0]
    
    examples = [SentimentExample(line.split("\t")[0].split(" "), int(line.split("\t")[-1])) for line in processed_lines]
    
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    
    index = 0
    
    for example in examples:
        for word in example.words:
            if word not in vocab.keys():
                
                vocab[word] = index
                
                index += 1

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    bow: torch.Tensor = torch.zeros(len(vocab.keys()))
    
    for word in text:
        if word in vocab.keys():
            
            if  not binary:
                bow[vocab[word]] += 1
                
            else:
                bow[vocab[word]] = 1

    return bow
