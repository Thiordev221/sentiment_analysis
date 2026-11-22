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
        #Nous allons utiliser le prompt engineering avec les few-shot prompts

        self.messages = [
            {
                "role": "system",
                "content": "You are a precise sentiment classifier. Analyze the user's message and classify it strictly into one of the following three categories: negative, neutral, or positive. Respond only in the exact format: sentiment: ."
            },
            {
                "role": "user",
                "content": "the food is not good"
            },
            {
                "role": "assistant",
                "content": "sentiment: negative"
            },
            {
                "role": "user",
                "content": "The restaurant opens at 12:00 PM and is located on Main Street."
            },
            {
                "role": "assistant",
                "content": "sentiment: neutral"
            },
            {
                "role": "user",
                "content": "The service was super fast and the dessert was amazing!"
            },
            {
                "role": "assistant",
                "content": "sentiment: positive"
            }
        ]

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
