
import torch
import torch.nn as nn
from transformers import pipeline
import re
import numpy as np
from huggingface_hub import PyTorchModelHubMixin

class EmotionPredictor(nn.Module,PyTorchModelHubMixin):
    def __init__(self):
        super(EmotionPredictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli",device=self.device)
        self.tokenizer = self.classifier.tokenizer
    
    def forward(self, payload):
        length_sentences = []
        sentences = []
        sorted_tensors = []
        tokens = self.tokenizer.encode(payload, return_tensors="pt", return_overflowing_tokens=True, stride=10, max_length=1096, truncation=True, padding=True)
        for i in range(len(tokens)):
            tokens_list = self.tokenizer.convert_ids_to_tokens(tokens[i])
            tokens_string = self.tokenizer.convert_tokens_to_string([token for token in tokens_list if token not in ['<s>', '</s>', self.tokenizer.pad_token]])            
            length_sentences.append(len(tokens_string.split()))
            sentences.append(tokens_string)
            
        length_sentences = torch.tensor(length_sentences)
        weights = length_sentences/length_sentences.sum()
        weights.to(self.device)
        del length_sentences,tokens
        emotions = ['anger', 'disgust', 'fear', 'inspiration', 'joy', 'love', 'neutral', 'sadness', 'suprise']
        predictions = self.classifier(sentences, emotions, multi_label=True)
        print(predictions)
        emotions.sort()
        for prediction in predictions:
            item = dict(zip(prediction['labels'],prediction['scores']))
            sorted_scores = [item[label] for label in emotions]
            sorted_tensors.append(sorted_scores)
        sorted_tensors = torch.tensor(sorted_tensors)
        sorted_tensors.to(self.device)
        weighted_scores = torch.mul(weights.unsqueeze(1),sorted_tensors).to(self.device)
        weighted_scores = weighted_scores.sum(dim=0)
        return weighted_scores.cpu().numpy()
