from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import transformers

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
    def __init__(self, model_path):
        # Load your BERT model from the provided model_path
        self.model = self.load_model(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_model(model_path):
        # Load the model from the provided checkpoint file
        model = BERTClass()
        optimizer = torch.optim.Adam(params=model.parameters())
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model

    def preprocess_text(self, text):
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def predict(self, text):
        inputs = self.preprocess_text(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = F.softmax(outputs, dim=-1)
        return probabilities

# Example usage:
#model_path = '/kaggle/input/model/current_checkpoint.pt'
#custom_pipeline = CustomBERTPipeline(model_path)
#text = "My name is Khan"
#predictions = custom_pipeline.predict(text)
#print(predictions)
