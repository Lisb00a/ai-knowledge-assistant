
from fastapi.testclient import TestClient

from src import api
from src.rag import CitacaoInvalidaError


# ==========================================
# CONFIGURAÇÃO DOS TESTES
# ==========================================

client = TestClient(api.app)


# ==========================================
# TESTE 01 - HEALTH CHECK
# ==========================================

def test_health():
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"
    assert resposta.json()["service"] == "AI Knowledge Assistant"


# ==========================================
# TESTE 02 - PERGUNTA COM SUCESSO
# ==========================================

def test_ask_sucesso(monkeypatch):

    def resposta_falsa(pergunta):
        return {
            "pergunta": pergunta,
            "resposta": "Random Forest usa várias árvores.",
            "fontes": []
        }

    monkeypatch.setattr(
        api,
        "responder_pergunta",
        resposta_falsa
    )

    resposta = client.post(
        "/ask",
        json={"pergunta": "Explique Random Forest"}
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert "Random Forest" in dados["resposta"]
    assert dados["pergunta"] == "Explique Random Forest"
    assert dados["fontes"] == []


# ==========================================
# TESTE 03 - PERGUNTA VAZIA
# ==========================================

def test_pergunta_vazia():

    resposta = client.post(
        "/ask",
        json={"pergunta": ""}
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 04 - PERGUNTA SOMENTE COM ESPAÇOS
# ==========================================

def test_pergunta_espacos():

    resposta = client.post(
        "/ask",
        json={"pergunta": "   "}
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 05 - SEM DOCUMENTOS RELEVANTES
# ==========================================

def test_sem_documentos(monkeypatch):

    def resposta_falsa(pergunta):
        return {
            "pergunta": pergunta,
            "resposta": "Não encontrei documentos para responder.",
            "fontes": []
        }

    monkeypatch.setattr(
        api,
        "responder_pergunta",
        resposta_falsa
    )

    resposta = client.post(
        "/ask",
        json={
            "pergunta": "Qual é a capital da Austrália?"
        }
    )

    assert resposta.status_code == 200
    assert resposta.json()["fontes"] == []

    assert (
        "Não encontrei"
        in resposta.json()["resposta"]
    )


# ==========================================
# TESTE 06 - ERRO INTERNO
# ==========================================

def test_erro_interno(monkeypatch):

    def erro_simulado(pergunta):
        raise RuntimeError("Erro de teste")

    monkeypatch.setattr(
        api,
        "responder_pergunta",
        erro_simulado
    )

    resposta = client.post(
        "/ask",
        json={"pergunta": "Teste"}
    )

    assert resposta.status_code == 500

    assert (
        resposta.json()["detail"]
        == "Erro interno ao processar a pergunta."
    )


# ==========================================
# TESTE 07 - PERGUNTA MUITO LONGA
# ==========================================

def test_pergunta_muito_longa():

    resposta = client.post(
        "/ask",
        json={"pergunta": "A" * 2001}
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 08 - CITAÇÃO INVÁLIDA RETORNA 502
# ==========================================

def test_citacao_invalida_retorna_502(monkeypatch):

    def resposta_com_citacao_invalida(pergunta):
        raise CitacaoInvalidaError(
            "Referência inválida: Fonte 8"
        )

    monkeypatch.setattr(
        api,
        "responder_pergunta",
        resposta_com_citacao_invalida
    )

    resposta = client.post(
        "/ask",
        json={
            "pergunta": "Como funciona Random Forest?"
        }
    )

    assert resposta.status_code == 502

    assert (
        resposta.json()["detail"]
        == (
            "O serviço de IA retornou uma resposta "
            "com referências inválidas."
        )
    )
