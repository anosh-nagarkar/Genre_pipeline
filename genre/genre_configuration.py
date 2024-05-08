from transformers import PretrainedConfig

class GenreConfig(PretrainedConfig):
    model_type = 'custom-bert-base-uncased'
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.id2label = {"0": "Action", "1": "Adventure", "2": "Biography", "3": "Comedy", "4": "Crime", "5": "Documentary", "6": "Drama", "7": "Family", "8": "Fantasy", "9": "History", "10": "Horror", "11": "Musical", "12": "Mystery", "13": "Romance", "14": "Sci-Fi", "15": "Sport", "16": "Thriller"}
        