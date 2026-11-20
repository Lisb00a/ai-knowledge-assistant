
import pytest

from src import rag


# ==========================================
# CONFIGURAÇÃO DOS TESTES
# ==========================================

@pytest.fixture(autouse=True)
def limpar_cache_modelo():
    """
    Evita que o cache do modelo de embeddings
    interfira entre os testes.
    """
    rag.carregar_modelo.cache_clear()
    yield
    rag.carregar_modelo.cache_clear()


# ==========================================
# OBJETOS FALSOS (MOCKS)
# ==========================================

class ModeloFalso:
    pass


class RespostaFalsa:
    text = "Random Forest utiliza várias árvores. (Fonte 1)"


class ModelosFalsos:
    def generate_content(self, model, contents):
        return RespostaFalsa()


class ClienteFalso:
    def __init__(self, api_key, http_options=None):
        self.models = ModelosFalsos()


# ==========================================
# PREPARAÇÃO DO AMBIENTE
# ==========================================

def preparar_ambiente(monkeypatch, documentos):
    """
    Simula a chave da API, o modelo de embeddings
    e a busca vetorial sem acessar serviços reais.
    """
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "chave_ficticia"
    )

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


# ==========================================
# TESTE 01 - PERGUNTA VAZIA
# ==========================================

def test_pergunta_vazia():
    with pytest.raises(
        ValueError,
        match="A pergunta não pode estar vazia"
    ):
        rag.responder_pergunta(" ")


# ==========================================
# TESTE 02 - RESPOSTA COM CONTEXTO
# ==========================================

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
    assert resultado["pergunta"] == "Como funciona Random Forest?"


# ==========================================
# TESTE 03 - SEM DOCUMENTOS
# ==========================================

def test_sem_documentos_nao_chama_gemini(monkeypatch):
    preparar_ambiente(monkeypatch, [])

    def gemini_proibido(api_key, http_options=None):
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


# ==========================================
# TESTE 04 - FILTRO DE SIMILARIDADE
# ==========================================

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


# ==========================================
# TESTE 05 - CONTEXTO ENVIADO AO GEMINI
# ==========================================

def test_gemini_recebe_contexto_correto(monkeypatch):
    documentos = [
        {
            "documento": "Random Forest combina várias árvores de decisão.",
            "similaridade": 0.92
        },
        {
            "documento": "GitHub Actions automatiza pipelines.",
            "similaridade": 0.15
        }
    ]

    preparar_ambiente(monkeypatch, documentos)

    chamadas = []

    class ModelosCaptura:
        def generate_content(self, model, contents):
            chamadas.append({
                "modelo": model,
                "prompt": contents
            })
            return RespostaFalsa()

    class ClienteCaptura:
        def __init__(self, api_key, http_options=None):
            self.models = ModelosCaptura()

    monkeypatch.setattr(
        rag.genai,
        "Client",
        ClienteCaptura
    )

    resultado = rag.responder_pergunta(
        "Como funciona Random Forest?"
    )

    assert len(chamadas) == 1

    prompt = chamadas[0]["prompt"]

    assert chamadas[0]["modelo"] == rag.GEMINI_MODEL

    assert (
        "Random Forest combina várias árvores de decisão."
        in prompt
    )

    assert (
        "GitHub Actions automatiza pipelines."
        not in prompt
    )

    assert "Como funciona Random Forest?" in prompt
    assert len(resultado["fontes"]) == 1


# ==========================================
# TESTE 06 - RESPOSTA VAZIA DO GEMINI
# ==========================================

def test_gemini_resposta_vazia(monkeypatch):
    documentos = [
        {
            "documento": "Random Forest utiliza várias árvores.",
            "similaridade": 0.90
        }
    ]

    preparar_ambiente(monkeypatch, documentos)

    class RespostaVazia:
        text = None

    class ModelosRespostaVazia:
        def generate_content(self, model, contents):
            return RespostaVazia()

    class ClienteRespostaVazia:
        def __init__(self, api_key, http_options=None):
            self.models = ModelosRespostaVazia()

    monkeypatch.setattr(
        rag.genai,
        "Client",
        ClienteRespostaVazia
    )

    with pytest.raises(
        RuntimeError,
        match="O Gemini não retornou uma resposta válida"
    ):
        rag.responder_pergunta(
            "Como funciona Random Forest?"
        )


# ==========================================
# TESTE 07 - TIMEOUT DO GEMINI
# ==========================================

def test_gemini_timeout_configurado(monkeypatch):
    documentos = [
        {
            "documento": "Random Forest utiliza várias árvores.",
            "similaridade": 0.90
        }
    ]

    preparar_ambiente(monkeypatch, documentos)

    configuracoes = []

    class ClienteTimeout:
        def __init__(self, api_key, http_options=None):
            configuracoes.append(http_options)
            self.models = ModelosFalsos()

    monkeypatch.setattr(
        rag.genai,
        "Client",
        ClienteTimeout
    )

    resultado = rag.responder_pergunta(
        "Como funciona Random Forest?"
    )

    assert len(resultado["fontes"]) == 1
    assert len(configuracoes) == 1

    assert configuracoes[0] is not None
    assert configuracoes[0].timeout == 30000
