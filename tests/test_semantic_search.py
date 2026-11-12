
import pytest

from src.semantic_search import buscar_semanticamente


def test_pergunta_vazia():
    with pytest.raises(ValueError):
        buscar_semanticamente(
            "",
            ["Documento de exemplo"],
            None
        )


def test_documentos_vazios():
    with pytest.raises(ValueError):
        buscar_semanticamente(
            "O que é RAG?",
            [],
            None
        )


def test_top_k_invalido():
    with pytest.raises(ValueError):
        buscar_semanticamente(
            "O que é RAG?",
            ["RAG combina busca e geração"],
            None,
            top_k=0
        )
