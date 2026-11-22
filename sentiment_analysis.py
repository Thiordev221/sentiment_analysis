import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

class Analyzer:
    def __init__(self):
        # Initialise le client compatible avec l'API OpenAI, configuré ici pour Groq.
        self.llm = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        # Définit le message système qui donne son rôle et son style de réponse au bot.
        self.messages = [{
            'role' : 'system',
            'content' : """
            Classify the sentiment of the given text as positive, negative, or neutral.
            """,                                                        #Très long texte pour que le bot puisse comprendre plus de contexte et qu'on puisse tester la limite de tokens asssigné.
        }]

    def sentiment_analysis(self, text):

        response = self.llm.chat.completions.create(
            model='openai/gpt-oss-20b',
            messages=[
                *self.messages,
                {'role': 'user', 'content': text}
            ],
            temperature=0.1,
        )

        return response.choices[0].message.content



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
