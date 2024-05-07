from transformers import pipeline
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel

class BERTClass(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = transformers.AutoModel.from_pretrained('bert-base-uncased')
            self.l2 = torch.nn.Dropout(0.3)
            self.l3 = torch.nn.Linear(768, 17)
        
        def forward(self, input_ids, attention_mask, token_type_ids):
            output_1 = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['pooler_output']
            output_2 = self.l2(output_1)
            output = self.l3(output_2)
            return output

class CustomBERTPipeline:
    def __init__(self):
        # Instantiate your BERT model
        self.model = BERTClass()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text):
        # Tokenization
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def predict(self, text):
        # Preprocess the text
        inputs = self.preprocess_text(text)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply softmax to logits
        probabilities = F.softmax(outputs, dim=-1)
        return probabilities

# Example usage:
#custom_pipeline = CustomBERTPipeline()
#text = "My name is Khan"
#predictions = custom_pipeline.predict(text)
#print(predictions)