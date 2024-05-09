from genre.EmotionPredictor import EmotionPredictor
from genre.NER_Wrapper import FullNERPipeline

def process_text(text):
    
    emotions = ['anger', 'disgust', 'fear', 'inspiration', 'joy', 'love', 'neutral', 'sadness', 'suprise']
    text = full_ner_pipeline.process_text(text)
    emotion_prediction = emotion_predictor(text)
    data = [{Y: X for X, Y in zip(emotion_prediction.tolist(), emotions)}]

    return {
        'emotion_prediction': data
    }

def get_models():
    return FullNERPipeline()

emotion_predictor,full_ner_pipeline = get_models()
