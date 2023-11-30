import re
import nltk

nltk.download(['stopwords','wordnet'])

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

class TextPreprocessor:
    def __init__(self):
        self._remove_punctuation = re.compile(r"[^\w\s]")
        custom_stopwords = set(stopwords.words("english"))
        position_words = ['above', 'below', 'up', 'down', 'between', 'on', 'under', 'against']
        for word in position_words:
            custom_stopwords.remove(word)
        self._stopwords = set(custom_stopwords)
        self._lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        text = self._remove_punctuation.sub(" ", text)
        text = text.lower()
        text = " ".join([word for word in text.split() if word not in self._stopwords])
        text = [self._lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(text)

if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    with open("descriptions.json","r") as file:
        descriptions = json.load(file)

    preprocessed_data = {}
    for image,captions in descriptions.items():
        processed_captions = [preprocessor.preprocess(text) for text in captions]
        preprocessed_data[image] = processed_captions

    with open("descriptions_cleaned.json","w") as file:
        json.dump(preprocessed_data,file)

    print("descriptions.json preprocessed and saved as descriptions_cleaned.json")