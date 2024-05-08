from typing import Any, Dict
from transformers import Pipeline
from transformers import AutoTokenizer,AutoModel
from transformers.utils import ModelOutput
import numpy as np 
import unicodedata
import re 
import torch

class Preprocess_Text:
    @staticmethod
    def remove_tags(sentence):
        return re.sub('<.*?>', ' ', sentence)

    @staticmethod
    def remove_accents(sentence):
        return unicodedata.normalize('NFD', sentence).encode('ascii', 'ignore').decode("utf-8")

    @staticmethod
    def remove_punctuation(sentence):
        sentence = re.sub(r'[?|!|\'|"|#]', '', sentence)
        sentence = re.sub(r'[.,;:(){}[\]\\/<>|-]', ' ', sentence)
        return sentence.replace("\n", " ")

    @staticmethod
    def keep_alpha(sentence):
        return re.sub('[^a-z A-Z]+', ' ', sentence)

    @staticmethod
    def lower_case(sentence):
        return sentence.lower()
    
    def __call__(self, text):
        text = self.remove_tags(text)
        text = self.remove_accents(text)
        text = self.remove_punctuation(text)
        text = self.keep_alpha(text)
        text = self.lower_case(text)
        return text
    
class GenrePredictionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "text" in kwargs:
            preprocess_kwargs['text'] = kwargs['text']
        return preprocess_kwargs,{},{}
    
    def preprocess(self,text,**kwargs):
        self.model = AutoModel.from_pretrained("Stanford-TH/GenrePrediction", trust_remote_code=True)
        text_preprocessing_obj = Preprocess_Text()
        processed_description = text_preprocessing_obj(text)
        
        try:
            if type(processed_description) == str:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                encoded_text = tokenizer.encode_plus(
                    processed_description, None, add_special_tokens=True, max_length=512,
                    padding='max_length', return_token_type_ids=True, truncation=True,
                    return_tensors='pt', return_overflowing_tokens=True )
                
                maximum_overflowed_samples = len(encoded_text.pop('overflow_to_sample_mapping'))
                
                try:
                    numbers = [[x for x in encoded_text.word_ids(batch_index=i) if x is not None][-1] 
                            for i in range(maximum_overflowed_samples)]
                except IndexError:
                    return None,torch.zeros(17,dtype='float32')
                
                sequence_length = numbers[-1]
                weights = [numbers[0]] + [numbers[i] - numbers[i-1] for i in range(1, len(numbers))]
                weights = (torch.tensor(weights) / sequence_length).to(self.device)  # Normalize weights
                return {"model_inputs":encoded_text,"weights":weights,"max_length":sequence_length}
            else:
                raise AttributeError()
        except Exception as error:
            print("Wrong format {}".format(str(error)))
            return -1
        
    def _forward(self,model_inputs):
        weights,max_length = model_inputs.pop('weights'),model_inputs.pop('max_length')
        with torch.no_grad():
            outputs = self.model(**model_inputs['model_inputs'])
            
        return {"model_outputs":outputs,"weights":weights,"max_length":max_length}        
    
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        # Apply sigmoid activation and calculate weighted logits
        print(model_outputs,postprocess_parameters)
        logits = torch.sigmoid(model_outputs.pop('model_outputs'))
        probabilities = logits * model_outputs.pop('weights').unsqueeze(1)
        
        probabilities = probabilities.sum(dim=0)
        
        top_scores, top_indices = torch.topk(probabilities, 3)  # Get the top 3 scores and their indices
        
        print(top_scores,top_indices)
    
        top_genres = [self.model.config.id2label[str(idx.item())] for idx in top_indices.squeeze()]
        top_scores = top_scores.detach().cpu().numpy()  
        
        genre_scores = {genre: score for genre, score in zip(top_genres, top_scores.squeeze())}
        
        return genre_scores