
from sentence_transformers import SentenceTransformer, util

from src.retriever import carregar_documentos


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def buscar_semanticamente(pergunta, documentos, modelo, top_k=3):

    if not pergunta.strip():
        raise ValueError("A pergunta não pode estar vazia")

    if not documentos:
        raise ValueError("Nenhum documento disponível")

    if top_k < 1:
        raise ValueError("top_k deve ser maior que zero")

    embeddings_documentos = modelo.encode(
        documentos,
        convert_to_tensor=True
    )

    embedding_pergunta = modelo.encode(
        pergunta,
        convert_to_tensor=True
    )

    similaridades = util.cos_sim(
        embedding_pergunta,
        embeddings_documentos
    )[0]

    indices = similaridades.argsort(descending=True)[:top_k]

    return [
        {
            "documento": documentos[int(i)],
            "similaridade": round(
                float(similaridades[int(i)]), 4
            )
        }
        for i in indices
    ]


if __name__ == "__main__":
    documentos = carregar_documentos()

    print("Carregando modelo de embeddings...")
    modelo = SentenceTransformer(MODEL_NAME)

    pergunta = input("\nDigite sua pergunta: ")

    resultados = buscar_semanticamente(
        pergunta,
        documentos,
        modelo
    )

    for resultado in resultados:
        print("\nDocumento:", resultado["documento"])
        print("Similaridade:", resultado["similaridade"])
