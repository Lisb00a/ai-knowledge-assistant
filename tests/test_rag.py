
import pytest

from src import rag


class ModeloFalso:
    pass


class RespostaFalsa:
    text = "Random Forest utiliza várias árvores. (Fonte 1)"


class ModelosFalsos:
    def generate_content(self, model, contents):
        return RespostaFalsa()


class ClienteFalso:
    def __init__(self, api_key):
        self.models = ModelosFalsos()


def preparar_ambiente(monkeypatch, documentos):
    monkeypatch.setenv("GEMINI_API_KEY", "chave_ficticia")

    monkeypatch.setattr(
        rag,
        "SentenceTransformer",
        lambda nome: ModeloFalso()
    )

    monkeypatch.setattr(
        rag,
        "buscar_no_indice",
        lambda pergunta, modelo, top_k: documentos
    )


def test_pergunta_vazia():
    with pytest.raises(ValueError):
        rag.responder_pergunta(" ")


def test_resposta_com_contexto(monkeypatch):
    documentos = [
        {
            "documento": "Random Forest usa várias árvores.",
            "similaridade": 0.85
        }
    ]

    preparar_ambiente(monkeypatch, documentos)

    monkeypatch.setattr(
        rag.genai,
        "Client",
        ClienteFalso
    )

    resultado = rag.responder_pergunta(
        "Como funciona Random Forest?"
    )

    assert "Random Forest" in resultado["resposta"]
    assert len(resultado["fontes"]) == 1


def test_sem_documentos_nao_chama_gemini(monkeypatch):
    preparar_ambiente(monkeypatch, [])

    def gemini_proibido(api_key):
        raise AssertionError(
            "Gemini não deveria ser chamado"
        )

    monkeypatch.setattr(
        rag.genai,
        "Client",
        gemini_proibido
    )

    resultado = rag.responder_pergunta(
        "Qual é a capital da Austrália?"
    )

    assert resultado["fontes"] == []
    assert "Não encontrei" in resultado["resposta"]


def test_filtro_remove_documentos_irrelevantes(monkeypatch):
    documentos = [
        {
            "documento": "Random Forest",
            "similaridade": 0.85
        },
        {
            "documento": "GitHub Actions",
            "similaridade": 0.20
        }
    ]

    preparar_ambiente(monkeypatch, documentos)

    monkeypatch.setattr(
        rag.genai,
        "Client",
        ClienteFalso
    )

    resultado = rag.responder_pergunta(
        "Explique Random Forest"
    )

    assert len(resultado["fontes"]) == 1
    assert resultado["fontes"][0]["documento"] == "Random Forest"
