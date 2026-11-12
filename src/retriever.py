
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "documentos.txt"


def carregar_documentos():
    texto = DATA_PATH.read_text(encoding="utf-8")

    documentos = [
        trecho.strip()
        for trecho in texto.split("\n\n")
        if trecho.strip()
    ]

    return documentos


def buscar_documentos(pergunta, documentos, top_k=3):
    if not pergunta.strip():
        raise ValueError("A pergunta não pode estar vazia")

    if not documentos:
        raise ValueError("Nenhum documento disponível")

    if top_k < 1:
        raise ValueError("top_k deve ser maior que zero")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode"
    )

    matriz = vectorizer.fit_transform(documentos)
    vetor_pergunta = vectorizer.transform([pergunta])

    similaridades = cosine_similarity(
        vetor_pergunta,
        matriz
    )[0]

    indices = similaridades.argsort()[::-1][:top_k]

    return [
        {
            "documento": documentos[i],
            "similaridade": round(float(similaridades[i]), 4)
        }
        for i in indices
        if similaridades[i] > 0
    ]


if __name__ == "__main__":
    documentos = carregar_documentos()

    pergunta = input("Digite sua pergunta: ")

    resultados = buscar_documentos(
        pergunta,
        documentos
    )

    if not resultados:
        print("Nenhum documento relevante encontrado.")

    for resultado in resultados:
        print("\nDocumento:", resultado["documento"])
        print("Similaridade:", resultado["similaridade"])
