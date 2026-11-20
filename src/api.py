
import logging
import time

from fastapi import FastAPI, HTTPException
from google.genai import errors
from pydantic import BaseModel, Field

from src.rag import responder_pergunta, CitacaoInvalidaError


# ==========================================
# CONFIGURAÇÃO DE LOGS
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURAÇÃO DA API
# ==========================================

app = FastAPI(
    title="AI Knowledge Assistant",
    description="API de perguntas e respostas com RAG e Gemini",
    version="1.2.0"
)


# ==========================================
# MODELO DE ENTRADA
# ==========================================

class PerguntaRequest(BaseModel):
    pergunta: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "Pergunta enviada ao assistente "
            "(máximo de 2000 caracteres)"
        )
    )


# ==========================================
# ENDPOINT - HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI Knowledge Assistant"
    }


# ==========================================
# ENDPOINT - PERGUNTAS E RESPOSTAS
# ==========================================

@app.post("/ask")
def ask(dados: PerguntaRequest):

    # 1. Limpeza da pergunta
    pergunta = dados.pergunta.strip()

    # 2. Validação de pergunta vazia
    if not pergunta:
        raise HTTPException(
            status_code=422,
            detail="A pergunta não pode estar vazia."
        )

    # 3. Início da medição de tempo
    inicio = time.perf_counter()

    logger.info(
        "Nova pergunta recebida | caracteres=%d",
        len(pergunta)
    )

    try:
        # 4. Execução do pipeline RAG
        resultado = responder_pergunta(pergunta)

        # 5. Cálculo do tempo de processamento
        duracao = time.perf_counter() - inicio

        logger.info(
            "Pergunta processada | fontes=%d | tempo=%.2fs",
            len(resultado["fontes"]),
            duracao
        )

        return resultado

    # ======================================
    # ERRO - CITAÇÃO INVÁLIDA DO GEMINI
    # ======================================

    except CitacaoInvalidaError as erro:
        logger.warning(
            "Resposta do Gemini com citação inválida | tipo=%s",
            type(erro).__name__
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "O serviço de IA retornou uma resposta "
                "com referências inválidas."
            )
        ) from erro

    # ======================================
    # ERRO - CONFIGURAÇÃO OU DADOS
    # ======================================

    except (ValueError, FileNotFoundError) as erro:
        logger.error(
            "Erro de configuração ou dados: %s",
            type(erro).__name__
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "O assistente não está configurado "
                "corretamente."
            )
        ) from erro

    # ======================================
    # ERRO - API DO GEMINI
    # ======================================

    except errors.APIError as erro:
        logger.warning(
            "Falha na API do Gemini | tipo=%s",
            type(erro).__name__
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "O serviço de IA está "
                "temporariamente indisponível."
            )
        ) from erro

    # ======================================
    # ERRO - FALHA INESPERADA
    # ======================================

    except Exception as erro:
        logger.exception(
            "Erro inesperado ao processar pergunta"
        )

        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar a pergunta."
        ) from erro
