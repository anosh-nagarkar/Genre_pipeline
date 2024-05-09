import torch
import torch.nn as nn
from transformers import DistilBertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import gc

class BertNER(nn.Module):
    """
        A custom PyTorch Module for Named Entity Recognition (NER) using DistilBertForTokenClassification.
    """
    def __init__(self,token_dims): 
        """
        Initializes the BertNER model.
        
        Parameters:
        token_dims (int): The number of unique tokens/labels in the NER task.
        """
        super(BertNER,self).__init__()
        if type(token_dims) !=  int:
            raise TypeError("Token Dimensions should be an integer")
        if token_dims <= 0:
            raise ValueError("Dimension should atleast be more than 1")
        
        self.pretrained_model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased',num_labels=token_dims)
        
    def forward(self,input_ids,attention_mask,labels=None):
        """
        Forward pass of the model.
        
        Parameters:
        input_ids (torch.Tensor): Tensor of token ids to be fed to DistilBERT.
        attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to by the model.
        labels (torch.Tensor, optional): Tensor of actual labels for computing loss. If None, the model returns logits.
        
        Returns:
        The model's output, which varies depending on whether labels are provided.
        """
        if labels == None:
            out = self.pretrained_model(input_ids=input_ids,attention_mask=attention_mask)
        
        out = self.pretrained_model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        
        return out 

class SentenceDataset(TensorDataset):
    """
    Custom Dataset class for sentences, handling tokenization and preparing inputs for the NER model.
    """
    def __init__(self, sentences, tokenizer, max_length=256):
        """
        Initializes the SentenceDataset.
        
        Parameters:
        sentences (list of str): The list of sentences to be processed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for converting sentences to model inputs.
        max_length (int): Maximum length of the tokenized output.
        """
        self.sentences = [sentence.split() for sentence in sentences]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text = self.tokenizer(sentences, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt",is_split_into_words=True)

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset by index.
        
        Parameters:
        idx (int): Index of the item to retrieve.
        
        Returns:
        A dictionary containing input_ids, attention_mask, word_ids, and the original sentences.
        """
        sentence = self.sentences[idx]
        encoded_sentence = self.tokenizer(sentence, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt", is_split_into_words=True)
        #During __getitem__ call the tokenized_sentence ('encoded_sentence') does not consider it to be tokenized by fast tokenizer, hence word_ids will not be given when accessed through data loader
        return {"input_ids":encoded_sentence.input_ids.squeeze(0),"attention_mask":encoded_sentence.attention_mask.squeeze(0),'word_ids':[-1 if x is None else x for x in encoded_sentence.word_ids()],"sentences":self.sentences}
    
class NERWrapper:
    """
    A wrapper class for the Named Entity Recognition (NER) model, simplifying the process of model loading,
    prediction, and utility functions.
    """
    def __init__(self, model_path, idx2tag_path, tokenizer_path='distilbert-base-uncased', token_dims=17):
        """
        Initializes the NERWrapper.
        
        Parameters:
        model_path (str): Path to the pre-trained NER model.
        idx2tag_path (str): Path to the index-to-tag mapping file, for decoding model predictions.
        tokenizer_path (str): Path or identifier for the tokenizer to be used.
        token_dims (int): The number of unique tokens/labels in the NER task.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)
        self.model = BertNER(token_dims=token_dims)
        self.idx2tag = self.load_idx2tag(idx2tag_path)
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """
        Loads the model from a specified path.
        
        Parameters:
        model_path (str): Path to the pre-trained NER model.
        """
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path,map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def load_idx2tag(self, idx2tag_path):
        """
        Loads the index-to-tag mapping from a specified path.
        
        Parameters:
        idx2tag_path (str): Path to the index-to-tag mapping file.
        
        Returns:
        dict: A dictionary mapping indices to tags.
        """
        with open(idx2tag_path, 'r') as file:
            idx2tag = json.load(file)
        def _jsonKeys2int(x):
            if isinstance(x, dict):
                return {int(k):v for k,v in x.items()}
            return x
        return _jsonKeys2int(idx2tag)
    
    def align_word_ids(self,texts, input_tensor,label_all_tokens=False):
        """
        Aligns word IDs with their corresponding labels, useful for creating a consistent format for model inputs.
        
        Parameters:
        texts (list of str): The original texts used for prediction.
        input_tensor (torch.Tensor): Tensor containing word IDs.
        label_all_tokens (bool): Whether to label all tokens or only the first token of each word.
        
        Returns:
        torch.Tensor: Tensor of aligned label IDs.
        """
        # Initialize an empty tensor for all_label_ids with the same shape and type as input_tensor but empty
        all_label_ids = []

        # Iterate through each row in the input_tensor
        for i, word_ids in enumerate(input_tensor):
            previous_word_idx = None
            label_ids = []
            # Iterate through each word_idx in the word_ids tensor
            for word_idx in word_ids:
                # Convert tensor to Python int for comparison
                word_idx = word_idx.item()
                if word_idx == -1:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(1)
                else:
                    label_ids.append(1 if label_all_tokens else -100)
                previous_word_idx = word_idx

            # Convert label_ids list to a tensor and assign it to the corresponding row in all_label_ids
            all_label_ids.append(label_ids)
        return all_label_ids

    def evaluate_text(self, sentences):
        """
        Evaluates texts using the NER model, returning the prediction results.
        
        Parameters:
        sentences (list of str): List of sentences to evaluate.
        
        Returns:
        list of str: The modified sentences with identified entities replaced with special tokens (e.g., <PER>).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        dataset = SentenceDataset(sentences,self.tokenizer)
        dataloader = DataLoader(dataset,batch_size=32,shuffle=False)
        predictions = [] 
        for data in dataloader:
            #Load the attention mask and the input ids 
            mask = data['attention_mask'].to(device)
            input_id = data['input_ids'].to(device)
            # Creates a tensor of word IDs for aligning model predictions with words.
            concatenated_tensor = torch.stack((data['word_ids'])).t()
            label_ids = torch.Tensor(self.align_word_ids(data['sentences'][0],concatenated_tensor)).to(device)
            output = self.model(input_id, mask, None)
            logits = output.logits
            for i in range(logits.shape[0]):
                 # Filters logits for each item in the batch, removing those not associated with actual words.
                logits_clean = logits[i][label_ids[i] != -100]
                # Determines the most likely label for each token and stores the result.
                predictions.append(logits_clean.argmax(dim=1).tolist())
            del mask,input_id,label_ids
            word_ids = []
            gc.collect()
            torch.cuda.empty_cache()
            prediction_label = [[self.idx2tag[i] for i in prediction] for prediction in predictions]
            
        return self.replace_sentence_with_tokens([sentence.split() for sentence in sentences],prediction_label)
    
    def replace_sentence_with_tokens(self,sentences,prediction_labels):
        """
        Replaces identified entities in sentences with special tokens based on the model's predictions.
        
        Parameters:
        sentences (list of list of str): Tokenized sentences.
        prediction_labels (list of list of str): Labels predicted by the model for each token.
        
        Returns:
        list of str: Modified sentences with entities replaced by special tokens.
        """
        modified_sentences = []
        for sentence, tags in zip(sentences, prediction_labels):
            words = sentence  # Split the sentence into words
            modified_sentence = [] # Initializes an empty list for the current modified sentence.
            skip_next = False  # A flag used to indicate whether to skip the next word (used for entities spanning multiple tokens).
            for i,(word,tag) in enumerate(zip(words,tags)):
                if skip_next:
                    skip_next = False
                    continue #Skip the current word
                if tag == 'B-per':
                    modified_sentence.append('<PER>')  
                     # Checks if the next word is part of the same entity (continuation of a person's name).
                    if i + 1 < len(tags) and tags[i + 1] == 'I-per':
                        skip_next = True  # Skip the next word if it's part of the same entity
                elif tag == 'I-per':
                    pass
                elif tag != 'I-per':
                    modified_sentence.append(word)
                    
            modified_sentences.append(" ".join(modified_sentence))
        
        return modified_sentences

class NextPassNERWrapper:
    """
    This class wraps around a pretrained BERT model for Named Entity Recognition (NER) tasks,
    simplifying the process of sentence processing, entity recognition, and sentence reconstruction
    with entity tags.
    """
    def __init__(self):
        """
        Initializes the wrapper by loading a pretrained tokenizer and model from Hugging Face's
        transformers library specifically designed for NER. It also sets up the device for model
        computation (GPU if available, otherwise CPU) and establishes a mapping from model output
        indices to entity types.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.entity_map = {
            0: "O",
            1: "B-MISC",
            2: "I-MISC",
            3: "B-PER",
            4: "I-PER",
            5: "B-ORG",
            6: "I-ORG",
            7: "B-LOC",
            8: "I-LOC",
        }

    def process_sentences(self, sentences):
        """
        Processes input sentences to identify named entities and reconstructs the sentences
        by tagging entities or modifying tokens based on the model's predictions. It leverages
        a custom dataset and DataLoader for efficient batch processing.
        
        Parameters:
        sentences (list of str): The sentences to be processed for named entity recognition.
        
        Returns:
        list of str: The list of processed sentences with entities tagged or tokens modified.
        """
        dataset = SentenceDataset(sentences,self.tokenizer)
        dataloader = DataLoader(dataset,batch_size=32,shuffle=False)
        paragraph = []
        for data in dataloader:
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask).logits
                
                word_ids = torch.stack((data['word_ids'])).t()
                tokens = [self.tokenizer.convert_ids_to_tokens(X) for X in input_ids.cpu().numpy()]
                predictions = torch.argmax(outputs,dim=2).cpu().numpy()
                skip_next = False
                for word_id,tokens_single,prediction in zip(word_ids,tokens,predictions): 
                    reconstructed_tokens = []
                    for word_id_token, token, prediction_token in zip(word_id, tokens_single, prediction):
                        if word_id is None or token in ["[CLS]", "[SEP]", "[PAD]"] or skip_next:
                            skip_next = False
                            continue

                        entity = self.entity_map[prediction_token]

                        if entity in ["B-PER", "I-PER"] and (reconstructed_tokens[-1] != "<PER>" if reconstructed_tokens else True):
                            reconstructed_tokens.append("<PER>")
                        elif entity not in ["B-PER", "I-PER"]:
                            if token.startswith("##"):
                                if(len(reconstructed_tokens) > 1 and reconstructed_tokens[-2] == '<'):
                                    reconstructed_tokens[-1] = '<' + reconstructed_tokens[-1] + token[2:] + '>'
                                    reconstructed_tokens.pop(-2)
                                    skip_next = True
                                else:
                                    reconstructed_tokens[-1] = reconstructed_tokens[-1] + token[2:]
                            else:
                                reconstructed_tokens.append(token.strip())

                    detokenized_sentence = " ".join(reconstructed_tokens)
                    paragraph.append(detokenized_sentence)
        return paragraph