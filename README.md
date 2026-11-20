# AI Knowledge Assistant — RAG com Gemini

Assistente de inteligência artificial desenvolvido em Python que utiliza **Retrieval-Augmented Generation (RAG)**, busca semântica, embeddings e Google Gemini para responder perguntas com base em documentos.

O projeto demonstra a construção de uma aplicação de IA generativa, desde a recuperação de informações até a disponibilização de uma API REST com FastAPI, incluindo testes automatizados, validação de citações e tratamento de erros.

## Objetivo

Desenvolver um assistente capaz de:

- Interpretar perguntas em linguagem natural.
- Localizar documentos relevantes por similaridade semântica.
- Utilizar documentos recuperados como contexto para um modelo de linguagem.
- Gerar respostas fundamentadas nas informações disponíveis.
- Apresentar as fontes recuperadas.
- Identificar referências numeradas que não correspondem às fontes disponíveis.
- Evitar chamadas desnecessárias ao Gemini quando não houver documentos suficientemente relevantes.

Este projeto integra um portfólio prático de Engenharia de Inteligência Artificial.

## Tecnologias

| Tecnologia | Aplicação |
|---|---|
| Python | Desenvolvimento da aplicação |
| Sentence Transformers | Geração de embeddings |
| NumPy | Operações vetoriais |
| Scikit-learn | Similaridade semântica |
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
              Filtro de similaridade
                     >= 0.40
                        |
                +-------+-------+
                |               |
                v               v
         Sem documentos    Com documentos
                |               |
                v               v
         Resposta padrão   Montagem do contexto
                                |
                                v
                           Google Gemini
                                |
                                v
                      Validação das citações
                                |
                        +-------+-------+
                        |               |
                        v               v
                 Citação válida    Citação inválida
                        |               |
                        v               v
                 Resposta + fontes   HTTP 502 (API)
```

### Etapas do processamento

1. O usuário envia uma pergunta.
2. O sistema valida a entrada.
3. O Sentence Transformers transforma a pergunta em um embedding.
4. O sistema consulta o índice vetorial persistente.
5. Os documentos recuperados são filtrados por similaridade mínima de `0.40`.
6. Os documentos relevantes são organizados em um contexto.
7. O Gemini recebe a pergunta e o contexto recuperado.
8. O sistema verifica se as referências numeradas na resposta correspondem às fontes disponíveis.
9. A aplicação retorna a resposta e as fontes ou sinaliza um erro de citação.

Se nenhum documento atingir o limiar de similaridade, a aplicação retorna uma mensagem informativa **sem chamar o Gemini**.

**Nota:** o limiar `0.40` é experimental e ainda precisa ser calibrado com um conjunto maior de perguntas.

## Validação de citações

O sistema utiliza a função `validar_citacoes()` para verificar referências numeradas presentes nas respostas do Gemini.

Exemplo válido:

`Random Forest combina várias árvores de decisão. (Fonte 1)`

Exemplo inválido:

`Random Forest combina várias árvores de decisão. (Fonte 8)`

Se apenas duas fontes estiverem disponíveis, a referência `Fonte 8` é inválida.

Nesses casos, a aplicação gera uma exceção específica, `CitacaoInvalidaError`, que a API transforma em uma resposta HTTP `502`.

**Importante:** essa validação verifica a existência das referências numeradas, mas não comprova que as afirmações geradas pelo Gemini são sustentadas pelo conteúdo dos documentos. Também não exige que toda resposta contenha citações.

## Otimização de desempenho

O modelo de embeddings utiliza `lru_cache` para evitar carregamentos repetidos durante a execução do mesmo processo.

```python
@lru_cache(maxsize=1)
def carregar_modelo():
    print("Carregando modelo de embeddings...")
    return SentenceTransformer(MODEL_NAME)
```

O cache foi verificado em execução local: duas chamadas retornaram o mesmo objeto (`Mesmo objeto: True`).

Ao reiniciar o processo, o modelo precisa ser carregado novamente.

## Estrutura do projeto

```text
ai-knowledge-assistant/
├── data/
│   ├── documentos.txt
│   └── vector_index/        # Gerado localmente
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

### 2. Criar e ativar o ambiente virtual

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```powershell
python -m pip install -r requirements.txt
```

### 4. Configurar o Gemini

Crie um arquivo `.env` na raiz do projeto, utilizando `.env.example` como referência.

```dotenv
GEMINI_API_KEY=SUA_CHAVE_AQUI
```

A chave pode ser criada no Google AI Studio. O uso da API depende das cotas e condições aplicáveis à conta e ao modelo.

**Nunca publique a chave no GitHub.**

### 5. Preparar o índice vetorial

Antes da primeira consulta, gere o índice a partir de `data/documentos.txt`:

```powershell
python -m src.vector_store
```

Na primeira execução, o programa cria os arquivos:

- `data/vector_index/embeddings.npy`
- `data/vector_index/documentos.txt`

Em seguida, solicita uma pergunta para testar a busca.

Se os documentos de origem forem alterados, recrie o índice:

```powershell
Remove-Item -Recurse -Force .\data\vector_index
python -m src.vector_store
```

O diretório `data/vector_index/` é gerado localmente e não é versionado.

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

A documentação interativa estará disponível em:

http://127.0.0.1:8000/docs

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

### Exemplo ilustrativo de resposta

```json
{
  "pergunta": "Como funciona o algoritmo Random Forest?",
  "resposta": "O algoritmo Random Forest utiliza várias árvores de decisão para realizar previsões. (Fonte 1)",
  "fontes": [
    {
      "documento": "O Random Forest utiliza várias árvores de decisão para realizar previsões.",
      "similaridade": 0.7428
    }
  ]
}
```

O texto produzido pelo modelo pode variar entre execuções.

### Tratamento de erros HTTP

| Código | Situação |
|---|---|
| `200` | Requisição processada com sucesso |
| `422` | Entrada inválida ou pergunta acima do limite |
| `500` | Erro inesperado no servidor |
| `502` | Resposta do Gemini com referência numerada inválida |
| `503` | Falha de configuração ou indisponibilidade da API Gemini |

A API aceita perguntas com até **2.000 caracteres**.

## Logs e observabilidade

A API registra:

- Recebimento de perguntas.
- Quantidade de caracteres da entrada.
- Quantidade de fontes recuperadas.
- Tempo total de processamento.
- Categorias de falhas ocorridas.

Os logs não registram a chave da API nem o texto completo das perguntas.

O tempo é medido com `time.perf_counter()`.

O sistema também configura um timeout de **30 segundos** para as chamadas ao Gemini.

Essas funcionalidades oferecem observabilidade básica, mas não substituem monitoramento e segurança de produção.

## Testes automatizados

Execute a suíte completa:

```powershell
python -m pytest -v
```

### Resultado validado

**25 testes aprovados**, conforme execução local realizada em novembro de 2026.

| Componente | Testes aprovados |
|---|---:|
| API REST | 8 |
| Pipeline RAG | 10 |
| Busca semântica | 3 |
| Índice vetorial | 4 |
| **Total** | **25** |

### Cobertura funcional

Os testes verificam:

- Validação de perguntas vazias.
- Limite de caracteres da API.
- Respostas com documentos relevantes.
- Ausência de documentos relevantes.
- Filtragem por similaridade.
- Construção do contexto enviado ao Gemini.
- Tratamento de respostas vazias.
- Configuração de timeout.
- Validação de citações existentes e inexistentes.
- Retorno HTTP 502 para citações inválidas.
- Tratamento de erros internos.
- Persistência e consulta do índice vetorial.

Os testes automatizados utilizam mocks para simular o Gemini, evitando chamadas reais à API durante a suíte.

Também foi realizada anteriormente uma execução real do RAG: uma pergunta sobre Random Forest recuperou um documento com similaridade de `0.7428` e recebeu uma resposta do Gemini.

## Segurança

O projeto adota algumas medidas básicas:

- Chave do Gemini carregada por variável de ambiente.
- Arquivo `.env` excluído do versionamento.
- Índice vetorial gerado localmente.
- Documentos fictícios utilizados na demonstração.
- Prompt orientando o Gemini a tratar documentos como dados, não como instruções.
- Validação de entradas com Pydantic.
- Tratamento específico de erros e citações inválidas.
- Logs sem exposição do conteúdo completo das perguntas.

Essas medidas não eliminam riscos de prompt injection, respostas incorretas ou uso indevido da API.

## Limitações

- Base documental pequena e fictícia.
- Limiar de similaridade ainda não calibrado sistematicamente.
- Validação de citações limitada à existência das referências numeradas.
- Possibilidade de respostas incorretas ou incompletas.
- Dependência da disponibilidade e das cotas do Gemini.
- API sem autenticação e sem limitação de requisições.
- Ausência de avaliação sistemática da qualidade das respostas.
- Aplicação ainda não preparada para uso público em produção.

## Próximas melhorias

- Criar um conjunto de perguntas e respostas de referência para avaliação.
- Medir relevância dos documentos recuperados e qualidade das respostas.
- Verificar se as citações realmente sustentam as afirmações geradas.
- Separar métricas de latência da recuperação e da geração.
- Implementar autenticação e limitação de requisições.
- Adicionar suporte a PDF e outras fontes documentais.
- Preparar a aplicação para deploy.
- Configurar ou verificar execução automatizada dos testes no GitHub Actions.

## Conclusão

O **AI Knowledge Assistant** integra processamento de linguagem natural, embeddings, busca vetorial, RAG, modelos de linguagem, APIs REST e testes automatizados em um protótipo funcional de assistente de conhecimento.

O projeto foi desenvolvido com foco em aprendizado, reprodutibilidade, validação de respostas e boas práticas de Engenharia de Inteligência Artificial.

**Repositório:** https://github.com/Lisb00a/ai-knowledge-assistant
