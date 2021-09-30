import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from typing import List
from mosestokenizer import MosesTokenizer
import sentencepiece as spm

# https://github.com/kakaobrain/kortok/tokenizer

class KlueBertTokenizer():
    def __init__(self, MODEL_NAME = "klue/bert-base"):
        self.MODEL_NAME = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    def tokenize(self, text):
        return self.tokenizer(text)

    def detokenize(self, tokens):
        text = " ".join(tokens).strip()
        return text

class SentencePieceTokenizer():
    def __init__(self, model_path: str, reverse: bool = False):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.reverse = reverse

    def tokenize(self, text: str) -> List[str]:
        if self.reverse:
            tokenized = self.sp.EncodeAsPieces(text[::-1].strip())
            tokenized = [s[::-1] for s in tokenized][::-1]
        else:
            tokenized = self.sp.EncodeAsPieces(text.strip())

        return tokenized

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", " ").strip()
        return text

class WordTokenizer():
    def __init__(self):
        self.tokenizer = MosesTokenizer()

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text.strip())

    def detokenize(self, tokens: List[str]) -> str:
        text = " ".join(tokens).strip()
        return text

    def close(self):
        self.tokenizer.close()

def compare_tokenizer(sentence):
    #BERT pretrained Tokenizer
    tokenizer = KlueBertTokenizer().tokenizer
    tokened_text = tokenizer(sentence)
    tokened_text = tokenizer.convert_ids_to_tokens((tokened_text['input_ids']))
    print()

    #Word Tokenzier
    tokenizer = WordTokenizer()
    tokened_text = tokenizer.tokenize(sentence)
    print(tokened_text)

    #SentencePiece Tokenizer
    tokenizer = SentencePieceTokenizer()
    tokened_text = tokenizer.tokenize(sentence)
    print(tokened_text)
    

def main():
    text = "나랑 쇼핑하자."
    compare_tokenizer(text)

if __name__ == "__main__":
    main()