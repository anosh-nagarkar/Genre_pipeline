import gc
from Executors_.FeatureExtractors.NER_Wrapper.NameExtractors import NERWrapper,NextPassNERWrapper

class FullNERPipeline:
    def __init__(self):
        """
        Initializes the FullNERPipeline with paths to the model and idx2tag configuration.
    
        
        Parameters:
        model_path (str): Path to the pre-trained NER model.
        idx2tag_path (str): Path to the index-to-tag mapping file.
        """
        # Initialize the NERWrapper with the provided model and idx2tag path.
        self.ner_wrapper = NERWrapper('models/NER_Models/torch_model.pth','models/NER_Models/idx2tag.json')
        
        # Initialize the NextPassNERWrapper which uses a different pre-trained model.
        self.next_ner_wrapper = NextPassNERWrapper()

    def process_text(self, text):
        """
        Processes the input text through two stages of NER processing and returns processed sentences.
        
        Parameters:
        text (str): The input text to be processed for named entity recognition.
        
        Returns:
        list of str: The list of processed sentences with entities tagged or tokens modified.
        """
        # First, evaluate the text using the initial NER model.
        evaluated_text = self.ner_wrapper.evaluate_text(text.split('.'))

        # Next, process the sentences through the second NER pass.
        ner_text = self.next_ner_wrapper.process_sentences(evaluated_text)

        # Manually collect garbage to manage memory when dealing with large models or data.
        gc.collect()

        return " ".join(ner_text)