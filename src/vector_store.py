
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retriever import carregar_documentos
from src.semantic_search import MODEL_NAME


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "vector_index"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
DOCUMENTOS_PATH = INDEX_DIR / "documentos.txt"


def criar_indice(documentos, modelo):
    if not documentos:
        raise ValueError("A lista de documentos não pode estar vazia.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = modelo.encode(
        documentos,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    DOCUMENTOS_PATH.write_text(
        "\n\n".join(documentos),
        encoding="utf-8"
    )

    print(f"Índice criado com {len(documentos)} documentos.")


def carregar_indice():
    if not EMBEDDINGS_PATH.exists() or not DOCUMENTOS_PATH.exists():
        raise FileNotFoundError(
            "Índice não encontrado. Execute a criação primeiro."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )

    texto = DOCUMENTOS_PATH.read_text(encoding="utf-8")

    documentos = [
        trecho.strip()
        for trecho in texto.split("\n\n")
        if trecho.strip()
    ]

    if len(documentos) != len(embeddings):
        raise ValueError("Índice inconsistente.")

    return documentos, embeddings


def buscar_no_indice(pergunta, modelo, top_k=3):
    if not pergunta.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    if top_k < 1:
        raise ValueError("top_k deve ser maior que zero.")

    documentos, embeddings = carregar_indice()

    embedding_pergunta = modelo.encode(
        pergunta,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similaridades = embeddings @ embedding_pergunta

    indices = np.argsort(similaridades)[::-1][:top_k]

    return [
        {
            "documento": documentos[int(i)],
            "similaridade": round(float(similaridades[int(i)]), 4)
        }
        for i in indices
    ]


if __name__ == "__main__":
    print("Carregando modelo...")
    modelo = SentenceTransformer(MODEL_NAME)

    if not EMBEDDINGS_PATH.exists() or not DOCUMENTOS_PATH.exists():
        print("Criando índice vetorial...")
        documentos = carregar_documentos()
        criar_indice(documentos, modelo)
    else:
        print("Índice vetorial encontrado!")

    pergunta = input("\nDigite sua pergunta: ")

    resultados = buscar_no_indice(pergunta, modelo)

    for resultado in resultados:
        print("\nDocumento:", resultado["documento"])
        print("Similaridade:", resultado["similaridade"])
