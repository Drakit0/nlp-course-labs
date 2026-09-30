from typing import List
from collections import Counter
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import os

def tokenize(text: str) -> List[str]:

    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    word_counts: Counter = Counter(words)
    trimmed_words: List[str] = [word for word in words if word_counts[word] > 5]

    return trimmed_words

def plot_embeddings(model, int_to_vocab, viz_words=400, figsize=(16, 16)):
    embeddings = model.in_embed.weight.to('cpu').data.numpy()
    
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])
    
    fig, ax = plt.subplots(figsize=figsize)
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    
    plt.show()

def save_model(model, model_path="skipgram_model.pth"):
    directory = os.path.dirname(model_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    return model_path
