
import os

from dotenv import load_dotenv
from google import genai


def main():
    # Carrega a chave do arquivo .env
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY não encontrada no arquivo .env"
        )

    # Conecta à API do Gemini
    client = genai.Client(api_key=api_key)

    print("Enviando pergunta ao Gemini...")

    resposta = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=(
            "Explique em duas frases o que é "
            "Random Forest em Machine Learning."
        ),
    )

    print("\nResposta do Gemini:")
    print(resposta.text)


if __name__ == "__main__":
    main()
