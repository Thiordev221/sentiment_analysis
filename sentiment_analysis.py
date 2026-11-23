import dotenv
import os
from transformers import pipeline

dotenv.load_dotenv()

class Analyzer:
    def __init__(self):
        # Initialise le client compatible avec l'API transformers.
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", top_k=None)

#Fonction pour l'analyse du sentiment
    def sentiment_analysis(self, text):
        labels = ["negative", "neutral", "positive"]
        result = self.classifier(text, candidate_labels=labels)

        return result


def main():
    analyzer = Analyzer()

    while True:
        text = input("Enter the text to analyze: ")
        sentiment = analyzer.sentiment_analysis(text)

        if text == "exit":
            break

        print(f"Sentiment: {sentiment}")


if __name__ == "__main__":
    main()
