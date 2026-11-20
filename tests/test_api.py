
from fastapi.testclient import TestClient
from src import api

client = TestClient(api.app)


def test_health():
    resposta = client.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_ask_sucesso(monkeypatch):
    def resposta_falsa(pergunta):
        return {
            "pergunta": pergunta,
            "resposta": "Random Forest usa várias árvores.",
            "fontes": []
        }

    monkeypatch.setattr(
        api, "responder_pergunta", resposta_falsa
    )

    resposta = client.post(
        "/ask",
        json={"pergunta": "Explique Random Forest"}
    )

    assert resposta.status_code == 200
    assert "Random Forest" in resposta.json()["resposta"]


def test_pergunta_vazia():
    resposta = client.post(
        "/ask", json={"pergunta": ""}
    )
    assert resposta.status_code == 422


def test_pergunta_espacos():
    resposta = client.post(
        "/ask", json={"pergunta": "   "}
    )
    assert resposta.status_code == 422


def test_sem_documentos(monkeypatch):
    def resposta_falsa(pergunta):
        return {
            "pergunta": pergunta,
            "resposta": "Não encontrei documentos para responder.",
            "fontes": []
        }

    monkeypatch.setattr(
        api, "responder_pergunta", resposta_falsa
    )

    resposta = client.post(
        "/ask",
        json={"pergunta": "Qual é a capital da Austrália?"}
    )

    assert resposta.status_code == 200
    assert resposta.json()["fontes"] == []


def test_erro_interno(monkeypatch):
    def erro_simulado(pergunta):
        raise RuntimeError("Erro de teste")

    monkeypatch.setattr(
        api, "responder_pergunta", erro_simulado
    )

    resposta = client.post(
        "/ask", json={"pergunta": "Teste"}
    )

    assert resposta.status_code == 500

def test_pergunta_muito_longa():
    resposta = client.post(
        "/ask",
        json={"pergunta": "A" * 2001}
    )

    assert resposta.status_code == 422
