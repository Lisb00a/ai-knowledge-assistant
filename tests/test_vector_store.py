
import numpy as np
import pytest

from src import vector_store


class ModeloFalso:
    def encode(self, textos, **kwargs):
        if isinstance(textos, str):
            return np.array([1.0, 0.0])

        return np.array([
            [1.0, 0.0] for _ in textos
        ])


def test_criar_e_carregar_indice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "INDEX_DIR",
        tmp_path
    )
    monkeypatch.setattr(
        vector_store,
        "EMBEDDINGS_PATH",
        tmp_path / "embeddings.npy"
    )
    monkeypatch.setattr(
        vector_store,
        "DOCUMENTOS_PATH",
        tmp_path / "documentos.txt"
    )

    documentos = ["Documento 1", "Documento 2"]
    modelo = ModeloFalso()

    vector_store.criar_indice(documentos, modelo)

    documentos_salvos, embeddings = vector_store.carregar_indice()

    assert documentos_salvos == documentos
    assert embeddings.shape == (2, 2)


def test_pergunta_vazia():
    with pytest.raises(ValueError):
        vector_store.buscar_no_indice("", ModeloFalso())


def test_top_k_invalido():
    with pytest.raises(ValueError):
        vector_store.buscar_no_indice(
            "O que é IA?",
            ModeloFalso(),
            top_k=0
        )


def test_documentos_vazios():
    with pytest.raises(ValueError):
        vector_store.criar_indice([], ModeloFalso())
