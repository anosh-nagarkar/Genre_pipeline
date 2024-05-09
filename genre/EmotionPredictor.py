import pandas as pd
import numpy as np
from ast import literal_eval
import yake
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

class ScriptMatcher:
    def __init__(self, data_path = None, model_name='paraphrase-mpnet-base-v2',dataframe = None):
        """
        Initialize the SeriesMatcher object.

        Parameters:
        data_path (str): Path to the dataset file.
        model_name (str): Name of the sentence transformer model. Default is 'paraphrase-mpnet-base-v2'.
        """
        if data_path is not None:
            self.dataset = pd.read_csv(data_path)
        if dataframe is not None:
            self.dataset = dataframe
        self.model = SentenceTransformer(model_name)
        self.kw_extractor = yake.KeywordExtractor("en", n=1, dedupLim=0.9)
        self.k_dataset = pd.read_csv('models/Similarity_K_Dataset/K_Dataset.csv')
        self._ent_type = ["PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT","WORK","ART","LAW",
    "LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"]
        self.embeddings_synopsis_list = np.load("models/Similarity_K_Dataset/plot_embeddings.npy")
        self.plot_embedding_list = np.load("models/Similarity_K_Dataset/synopsis_embeddings.npy")

    def preprocess_dataset(self):
        """
        Preprocess the dataset by filling missing plot values, dropping rows with missing values in essential columns,
        and converting genre strings to lists.
        """
        # Fill NaN values in the 'Plot' column with 'Final_Synopsis'
        self.dataset["Plot"] = self.dataset["Plot"].fillna(self.dataset["Cleaned_Synopsis"])
        # Drop rows with missing values in 'Genres' and 'Final_Synopsis'
        self.dataset = self.dataset.dropna(subset=["genres", "Cleaned_Synopsis"]).reset_index(drop=True)
        # Convert 'Genres' column to list of literals
        self.dataset["genres"] = self.dataset["genres"].apply(literal_eval)

    def extract_keywords(self, text):
        """
        Extract keywords from a given text using the YAKE keyword extraction algorithm.

        Parameters:
        text (str): Text from which to extract keywords.

        Returns:
        str: A string of extracted keywords joined by spaces.
        """
        extracted_keywords = self.kw_extractor.extract_keywords(text)
        return " ".join([keywords[0] for keywords in extracted_keywords if keywords[0] not in self._ent_type])
    
    def preprocess_text(self, text):
        """
        Process a given text to replace named entities and extract keywords.

        Parameters:
        text (str): The text to process.

        Returns:
        str: Processed text with named entities replaced and keywords extracted.
        """
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy NLP model...")
            os.system(
                "pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl")
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)
        replaced_text = text
        for token in doc:
            if token.ent_type_ != "MISC" and token.ent_type_ != "":
                replaced_text = replaced_text.replace(token.text, f"<{token.ent_type_}>")
        
        return self.extract_keywords(replaced_text)
    
    def create_keyword_dataset(self):
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy NLP model...")
            print("This may take a few minutes and it's one time process...")
            os.system(
                "pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl")
            nlp = spacy.load("en_core_web_sm")
        NER_Plot_sentences = []
        for temp in self.dataset.Plot.to_list():
            doc = nlp(temp)
            replaced_text = temp
            for token in doc:
                if token.ent_type_ != "MISC" and token.ent_type_ != "":
                    replaced_text = replaced_text.replace(token.text, f"<{token.ent_type_}>")
            NER_Plot_sentences.append(replaced_text)
        self.dataset['NER_Plot'] = NER_Plot_sentences

        NER_Synopsis_sentences = []
        for temp in self.dataset.Synopsis.to_list():
            doc = nlp(temp)
            replaced_text = temp
            for token in doc:
                if token.ent_type_ != "MISC" and token.ent_type_ != "":
                    replaced_text = replaced_text.replace(token.text, f"<{token.ent_type_}>")
            NER_Synopsis_sentences.append(replaced_text)
        self.dataset['NER_Synopsis'] = NER_Synopsis_sentences

        """
        Create a keyword dataset from the existing dataset. This involves extracting keywords from both the plot
        and the synopsis of each series and then combining them with the genre information.

        """
        self.k_dataset = pd.DataFrame(columns=["IMDB_ID", "Series", "Keywords_Synopsis", "Keywords_Plot", "Synopsis",
                                              "Genre", "Plot"])
        for index, row in self.dataset.iterrows():
            extracted_keywords_synopsis = self.extract_keywords(row["NER_Synopsis"])
            extracted_keywords_plot = self.extract_keywords(row["NER_Plot"])
            synopsis_sentence = " ".join(row["genres"]) + " " + extracted_keywords_synopsis
            plot_sentence = " ".join(row["genres"]) + " " + extracted_keywords_plot
            self.k_dataset.loc[len(self.k_dataset)] = {"IMDB_ID": row["IMDB_ID"], "Series": row["Movie"],
                                                       "Keywords_Synopsis": synopsis_sentence,
                                                       "Keywords_Plot": plot_sentence,
                                                       "Synopsis": row["Cleaned_Synopsis"], "Genre": row["genres"], "Plot": row["Plot"]}

    def calculate_similarity_matrix(self):
        """
        Calculate the cosine similarity matrices for both plot and synopsis embeddings. A final similarity matrix is
        computed as a weighted sum of these two matrices.
        """
        embeddings_synopsis = self.model.encode(self.k_dataset["Synopsis"].tolist())
        embeddings_plot = self.model.encode(self.k_dataset["Plot"].tolist())

        cosine_similarity_matrix_1 = cosine_similarity(embeddings_synopsis, embeddings_synopsis)
        cosine_similarity_matrix_2 = cosine_similarity(embeddings_plot, embeddings_plot)

        self.final_cosine_similarity_matrix = 0.75 * cosine_similarity_matrix_1 + 0.25 * cosine_similarity_matrix_2
        
    def find_similar_series(self, new_synopsis, genres_keywords,k=5):
        """
        Find series similar to a new synopsis.

        Parameters:
        new_synopsis (str): The synopsis to compare.
        k (int): The number of similar series to return.

        Returns:
        pd.DataFrame: A dataframe of the closest series.
        """
        processed_synopsis = self.preprocess_text(new_synopsis)
        genre_keywords = " ".join(genres_keywords) 
        print(genre_keywords)
        synopsis_sentence = genre_keywords + self.extract_keywords(processed_synopsis)
        
        synopsis_embedding = self.model.encode([synopsis_sentence])
        
        cosine_similarity_matrix = 0.75 * cosine_similarity(synopsis_embedding, self.embeddings_synopsis_list) + 0.25 * cosine_similarity(synopsis_embedding,self.plot_embedding_list)

        top_k_indices = cosine_similarity_matrix.argsort()[0, -k:][::-1]
        closest_series = self.k_dataset.iloc[top_k_indices]
        
        # Add scores column
        closest_series["Score"] = cosine_similarity_matrix[0, top_k_indices]
        
        return closest_series[["Series", "Genre","Score"]].to_dict(orient='records')
            