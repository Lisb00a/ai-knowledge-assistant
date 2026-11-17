
import logging
import time

from fastapi import FastAPI, HTTPException
from google.genai import errors
from pydantic import BaseModel, Field

from src.rag import responder_pergunta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Knowledge Assistant",
    description="API de perguntas e respostas com RAG e Gemini",
    version="1.1.0"
)


class PerguntaRequest(BaseModel):
    pergunta: str = Field(
        ...,
        min_length=1,
        description="Pergunta enviada ao assistente"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI Knowledge Assistant"
    }


@app.post("/ask")
def ask(dados: PerguntaRequest):
    pergunta = dados.pergunta.strip()

    if not pergunta:
        raise HTTPException(
            status_code=422,
            detail="A pergunta não pode estar vazia."
        )

    inicio = time.perf_counter()

    logger.info(
        "Nova pergunta recebida | caracteres=%d",
        len(pergunta)
    )

    try:
        resultado = responder_pergunta(pergunta)

        duracao = time.perf_counter() - inicio

        logger.info(
            "Pergunta processada | fontes=%d | tempo=%.2fs",
            len(resultado["fontes"]),
            duracao
        )

        return resultado

    except (ValueError, FileNotFoundError) as erro:
        logger.error(
            "Erro de configuração ou dados: %s",
            type(erro).__name__
        )

        raise HTTPException(
            status_code=503,
            detail="O assistente não está configurado corretamente."
        ) from erro

    except errors.APIError as erro:
        logger.warning(
            "Falha na API do Gemini | tipo=%s",
            type(erro).__name__
        )

        raise HTTPException(
            status_code=503,
            detail="O serviço de IA está temporariamente indisponível."
        ) from erro

    except Exception as erro:
        logger.exception(
            "Erro inesperado ao processar pergunta"
        )

        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar a pergunta."
        ) from erro
