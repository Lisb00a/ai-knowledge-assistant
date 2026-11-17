
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag import responder_pergunta


app = FastAPI(
    title="AI Knowledge Assistant",
    description="API de perguntas e respostas com RAG e Gemini",
    version="1.0.0"
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

    try:
        resultado = responder_pergunta(pergunta)

        return resultado

    except (ValueError, FileNotFoundError) as erro:
        raise HTTPException(
            status_code=503,
            detail="O assistente não está configurado corretamente."
        ) from erro

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar a pergunta."
        ) from erro
