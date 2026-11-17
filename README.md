# AI Knowledge Assistant — RAG com Gemini

Assistente de inteligência artificial desenvolvido em Python que utiliza **Retrieval-Augmented Generation (RAG)**, busca semântica, embeddings e Google Gemini para responder perguntas com base em documentos.

O projeto demonstra a construção de uma aplicação de IA generativa, da recuperação de informações à disponibilização de uma API REST com FastAPI.

## Objetivo

Desenvolver um assistente capaz de:

- Interpretar perguntas em linguagem natural.
- Localizar documentos relevantes por similaridade semântica.
- Utilizar os documentos recuperados como contexto para um modelo de linguagem.
- Gerar respostas fundamentadas nas informações disponíveis.
- Apresentar as fontes recuperadas.
- Evitar chamadas desnecessárias ao Gemini quando não houver documentos suficientemente relevantes.

O projeto integra um portfólio prático de Engenharia de Inteligência Artificial.

## Tecnologias

| Tecnologia | Aplicação |
|---|---|
| Python | Desenvolvimento da aplicação |
| Sentence Transformers | Geração de embeddings |
| NumPy | Operações vetoriais e cálculo de similaridade no índice |
| Scikit-learn | Ferramentas de similaridade na busca semântica |
| Google Gemini API (`google-genai`) | Geração de respostas |
| FastAPI | API REST |
| Pydantic | Validação de requisições |
| Pytest e TestClient | Testes automatizados |
| Git e GitHub | Versionamento |

## Arquitetura do sistema

```text
                 PERGUNTA DO USUÁRIO
                         |
                         v
                 Validação da entrada
                         |
                         v
                 Modelo de embeddings
                      (cache)
                         |
                         v
                 Busca no índice vetorial
                         |
                         v
              Filtro de similaridade >= 0.40
                         |
                +--------+--------+
                |                 |
          Sem documentos     Com documentos
                |                 |
                v                 v
          Resposta padrão    Montagem do contexto
                                  |
                                  v
                              Google Gemini
                                  |
                                  v
                            Resposta + fontes
```

### Etapas do processamento

1. O usuário envia uma pergunta.
2. O sistema valida a entrada.
3. O Sentence Transformers transforma a pergunta em um embedding.
4. O sistema consulta o índice vetorial persistente.
5. Os documentos recuperados são filtrados por similaridade mínima de `0.40`.
6. Os documentos relevantes são organizados em um contexto.
7. O Gemini recebe a pergunta e o contexto recuperado.
8. A aplicação retorna a resposta gerada e as fontes.

Se nenhum documento atingir o limiar, a aplicação retorna uma mensagem informativa **sem chamar o Gemini**.

> **Nota:** o limiar `0.40` é experimental; não garante relevância e ainda precisa ser calibrado com um conjunto maior de perguntas.

## Otimização de desempenho

O modelo de embeddings é carregado uma vez por processo e reutilizado nas chamadas seguintes com `lru_cache`:

```python
@lru_cache(maxsize=1)
def carregar_modelo():
    print("Carregando modelo de embeddings...")
    return SentenceTransformer(MODEL_NAME)
```

O cache foi verificado em execução local: duas chamadas retornaram o mesmo objeto (`Mesmo objeto: True`). Ao reiniciar o processo, o modelo precisa ser carregado novamente.

## Estrutura do projeto

```text
ai-knowledge-assistant/
├── data/
│   ├── documentos.txt
│   └── vector_index/          # Gerado localmente; não versionado
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── rag.py
│   ├── retriever.py
│   ├── semantic_search.py
│   ├── vector_store.py
│   └── test_gemini.py
├── tests/
│   ├── test_api.py
│   ├── test_rag.py
│   ├── test_semantic_search.py
│   └── test_vector_store.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Instalação e execução

### 1. Clonar o repositório

```powershell
git clone https://github.com/Lisb00a/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Criar e ativar o ambiente virtual (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```powershell
python -m pip install -r requirements.txt
```

### 4. Configurar o Gemini

Crie um arquivo `.env` na raiz do projeto, usando `.env.example` como referência:

```dotenv
GEMINI_API_KEY=SUA_CHAVE_AQUI
```

A chave pode ser criada no Google AI Studio. O uso da API depende das cotas e condições aplicáveis à conta e ao modelo. **Nunca publique a chave no GitHub.**

### 5. Preparar o índice vetorial

Antes da primeira consulta, gere o índice a partir de `data/documentos.txt`:

```powershell
python -m src.vector_store
```

Na primeira execução, o programa cria `data/vector_index/embeddings.npy` e `data/vector_index/documentos.txt` e, em seguida, solicita uma pergunta para testar a busca. Digite uma pergunta para concluir a execução.

Em execuções seguintes, o programa reutiliza o índice existente. **Se os documentos de origem forem alterados, recrie o índice:**

```powershell
Remove-Item -Recurse -Force .\data\vector_index
python -m src.vector_store
```

O diretório `data/vector_index/` é gerado localmente e está no `.gitignore`.

### 6. Executar o assistente no terminal

```powershell
python -m src.rag
```

Exemplo de pergunta:

```text
Como funciona o algoritmo Random Forest?
```

## API REST com FastAPI

Inicie o servidor:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Documentação interativa: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Se a porta `8000` já estiver em uso, substitua por `8001` no comando e no endereço do Swagger.

### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/health` | Verifica se a API está respondendo |
| `POST` | `/ask` | Recebe uma pergunta e retorna resposta e fontes |

### Exemplo de requisição

`POST /ask`

```json
{
  "pergunta": "Como funciona o algoritmo Random Forest?"
}
```

### Exemplo de resposta

```json
{
  "pergunta": "Como funciona o algoritmo Random Forest?",
  "resposta": "O algoritmo Random Forest funciona utilizando várias árvores de decisão para realizar previsões.",
  "fontes": [
    {
      "documento": "O Random Forest utiliza várias árvores de decisão para realizar previsões.",
      "similaridade": 0.7428
    }
  ]
}
```

O texto produzido pelo modelo pode variar entre execuções.

## Logs e tratamento de erros

A API registra recebimento de perguntas, tamanho da entrada, quantidade de fontes e tempo de processamento. Também trata erros de configuração, falhas do serviço Gemini e exceções inesperadas, retornando mensagens HTTP genéricas ao cliente para evitar expor detalhes internos.

Os logs não registram a chave da API nem o texto completo da pergunta. Em um ambiente de produção, seriam necessários controles adicionais de segurança e observabilidade.

## Testes automatizados

Execute:

```powershell
python -m pytest -v
```

**Resultado observado: 17 testes aprovados em 7,23 segundos.**

| Componente | Testes aprovados |
|---|---:|
| API REST | 6 |
| Pipeline RAG | 4 |
| Busca semântica | 3 |
| Índice vetorial | 4 |
| **Total** | **17** |

Os testes cobrem validação de entradas, respostas simuladas, persistência do índice, busca semântica, filtragem por relevância e tratamento de erros. Os testes automatizados simulam a integração com o Gemini, evitando chamadas reais à API.

Também foi realizada uma execução real do RAG: uma pergunta sobre Random Forest recuperou um documento com similaridade de `0.7428` e recebeu resposta do Gemini.

## Segurança

- A chave do Gemini é lida de variável de ambiente, configurada no arquivo local `.env`.
- O `.env` e o índice vetorial gerado não são versionados.
- Os documentos utilizados são fictícios, voltados à demonstração.
- O prompt orienta o Gemini a tratar os documentos como dados, não como instruções.
- A API valida entradas e trata exceções.

Essas medidas não eliminam riscos de prompt injection, respostas incorretas ou uso indevido da API.

## Limitações

- Base documental pequena e fictícia.
- Limiar de similaridade ainda não calibrado sistematicamente.
- Possibilidade de respostas incorretas ou incompletas.
- Dependência da disponibilidade e das cotas da API Gemini.
- API sem autenticação e sem controle de taxa de requisições.
- Ausência de avaliação sistemática de qualidade da recuperação e geração.

## Próximas melhorias

- Criar um conjunto de perguntas e respostas de referência para avaliação.
- Medir relevância dos documentos recuperados e qualidade das respostas.
- Separar métricas de latência da busca e da geração.
- Implementar autenticação e limitação de requisições.
- Adicionar suporte a PDF e outras fontes documentais.
- Preparar a aplicação para deploy.

## Conclusão

O **AI Knowledge Assistant** integra NLP, embeddings, busca vetorial, RAG, modelos de linguagem, APIs REST e testes automatizados em um protótipo funcional de assistente de conhecimento. O projeto foi desenvolvido com foco em aprendizado, reprodutibilidade e boas práticas de Engenharia de Inteligência Artificial.

**Repositório:** https://github.com/Lisb00a/ai-knowledge-assistant
