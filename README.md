## RAG com Gemini

O AI Knowledge Assistant é um assistente de inteligência artificial desenvolvido em Python que utiliza Retrieval-Augmented Generation (RAG) para responder perguntas com base em documentos.

### Tecnologias
- Python
- Sentence Transformers
- NumPy
- Google Gemini API
- Pytest

### Funcionamento
1. A pergunta é convertida em um embedding.
2. O sistema consulta um índice vetorial persistente.
3. Os documentos recuperados são filtrados por similaridade mínima de 0,40.
4. Os documentos relevantes são enviados ao Gemini 3.1 Flash Lite.
5. O assistente gera uma resposta e apresenta as fontes recuperadas.
6. Se nenhum documento atingir o limiar, o sistema retorna uma mensagem sem chamar o Gemini.

O limiar de 0,40 é experimental e deverá ser avaliado com um conjunto maior de perguntas.

### Como executar

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Crie um arquivo `.env` com a variável `GEMINI_API_KEY`, utilizando `.env.example` como referência.

Execute o assistente:

```powershell
python -m src.rag
```

### Testes automatizados

```powershell
python -m pytest -v
```

Na validação realizada durante o desenvolvimento, os 11 testes listados foram aprovados individualmente, cobrindo busca semântica, índice vetorial e lógica do RAG.

Os testes do RAG utilizam simulações para evitar chamadas reais à API do Gemini.

### Segurança

A chave da API é armazenada localmente no arquivo `.env`, que está incluído no `.gitignore`.

O projeto utiliza documentos fictícios e não contém dados confidenciais de clientes ou empresas.

### Limitações

O filtro de similaridade não garante que todos os documentos recuperados sejam relevantes. O modelo de linguagem também pode produzir respostas incorretas. Futuras melhorias incluem avaliação sistemática da recuperação, tratamento de falhas da API e testes adicionais de qualidade das respostas.

## API REST com FastAPI

O projeto disponibiliza uma API REST que permite consultar o assistente de inteligência artificial utilizando RAG e Gemini.

### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/health` | Verifica se a API está funcionando |
| POST | `/ask` | Recebe uma pergunta e retorna a resposta com as fontes |

### Executar a API

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

A documentação interativa estará disponível em:

`http://127.0.0.1:8000/docs`

### Exemplo de requisição

```json
{
  "pergunta": "Como funciona o algoritmo Random Forest?"
}
```

### Exemplo de resposta

```json
{
  "pergunta": "Como funciona o algoritmo Random Forest?",
  "resposta": "Random Forest utiliza várias árvores de decisão para realizar previsões (Fonte 1).",
  "fontes": [
    {
      "documento": "O Random Forest utiliza várias árvores de decisão para realizar previsões.",
      "similaridade": 0.7428
    }
  ]
}
```

### Testes

A API possui testes automatizados com `pytest` e `TestClient`, incluindo validação de perguntas, respostas simuladas e tratamento de erros.

```powershell
python -m pytest -v
```

Os testes automatizados não precisam realizar chamadas reais à API do Gemini.