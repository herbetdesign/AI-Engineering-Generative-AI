# FoodOps AI RAG

Assistente de operações de delivery que pesquisa uma base de conhecimento local, responde perguntas com contexto recuperado e apresenta referências às fontes. O projeto é um MVP didático para portfólio; os documentos fornecidos são fictícios e não representam políticas oficiais de nenhuma plataforma.

## Funcionalidades

- Carrega arquivos Markdown, TXT e PDFs com texto selecionável.
- Divide os documentos em chunks e mantém arquivo, índice e posição de origem.
- Gera embeddings locais e persiste o índice no ChromaDB.
- Recebe documentos pelo Swagger e os indexa em data/knowledge_base/uploads/.
- Responde perguntas em português pela OpenAI Responses API, com referências.
- Recusa responder quando a busca não encontra contexto.
- Guarda o histórico local em SQLite e permite continuar uma conversa usando seu ID.
- Mede a recuperação semântica com 40 perguntas de avaliação.
- Expõe a API FastAPI, com documentação interativa em /docs.

PDFs que contêm somente imagens precisam de OCR, que ainda não está incluído.

## Tecnologias

- Python >=3.11,<3.12
- FastAPI, Pydantic Settings e Uvicorn
- ChromaDB e embeddings locais
- OpenAI Python SDK / Responses API
- SQLite (biblioteca padrão do Python)
- Pytest, HTTPX e Ruff
- Docker e GitHub Actions

## Estrutura

    foodops-ai-rag/
    ├── .github/workflows/quality.yml
    ├── data/
    │   ├── evaluation/retrieval_questions.json
    │   ├── knowledge_base/              # exemplos .md; uploads locais são ignorados pelo Git
    │   └── conversations.sqlite3        # criado localmente e ignorado pelo Git
    ├── src/foodops_ai/
    │   ├── evaluation/                  # métricas de busca
    │   ├── knowledge/                   # loader, chunker e banco vetorial
    │   ├── answer_generator.py
    │   ├── api_models.py
    │   ├── conversation_store.py
    │   ├── evaluate_retrieval.py
    │   ├── index_knowledge.py
    │   └── main.py
    ├── tests/
    ├── .env.example
    ├── Dockerfile
    ├── compose.yaml
    └── pyproject.toml

## Instalação no Windows

Abra o PowerShell e execute:

    cd C:\projetos\foodops-ai-rag
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
    Copy-Item .env.example .env
    notepad .env

No arquivo .env, configure sua chave:

    OPENAI_API_KEY=sua-chave-da-api

O ambiente virtual foi criado com Python 3.11 e o projeto rejeita versões fora de >=3.11,<3.12. Com o ambiente ativo, o comando python usa o Python 3.11 desse ambiente. Não publique o arquivo .env.

## Usar localmente

### 1. Indexar os documentos existentes

Coloque arquivos .md, .txt ou .pdf em data/knowledge_base/ e rode:

    python -m foodops_ai.index_knowledge

O índice Chroma é salvo em data/chroma/. O primeiro uso pode baixar um modelo local de embeddings; isso não consome créditos da OpenAI.

### 2. Iniciar a API

    python -m uvicorn foodops_ai.main:app --reload

Acesse http://127.0.0.1:8000/docs.

### 3. Enviar um documento

No Swagger, expanda POST /documents/upload, clique em Try it out, selecione um arquivo .md, .txt ou .pdf e clique em Execute. O limite é 10 MB por arquivo. O arquivo é guardado em data/knowledge_base/uploads/ e indexado localmente.

O loader extrai texto selecionável de PDFs. Documentos digitalizados como imagem não são processados porque OCR não faz parte desta versão.

### 4. Fazer uma pergunta e continuar uma conversa

Em POST /ask, envie:

    {
      "question": "O que fazer quando um pedido está atrasado?",
      "top_k": 3
    }

A resposta inclui answer, sources e conversation_id. Copie o ID retornado e envie-o no próximo pedido para continuar o contexto da conversa:

    {
      "question": "O que devo registrar?",
      "top_k": 3,
      "conversation_id": "cole-aqui-o-id-retornado"
    }

Consulte as mensagens em GET /conversations/{conversation_id}. O histórico é salvo localmente em data/conversations.sqlite3.

Se a base não contiver trechos, a API retorna uma mensagem de insuficiência de contexto sem chamar a OpenAI.

## Avaliar a recuperação

Depois de indexar os documentos, execute:

    python -m foodops_ai.evaluate_retrieval

O programa avalia 40 perguntas e informa:

- Hit Rate@k: percentual em que o documento esperado aparece entre os primeiros k resultados.
- MRR: mede a posição do primeiro documento esperado; quanto mais próximo do topo, maior a pontuação.

O padrão é k=3. Para mudar:

    python -m foodops_ai.evaluate_retrieval --top-k 5

Essa avaliação mede somente a busca nos documentos fictícios. Ela não mede se a resposta do LLM está correta.

## Testes e qualidade

Com o ambiente virtual ativo:

    python -m pytest -q
    python -m ruff check .
    python -m ruff format --check .

O GitHub Actions executa essas verificações em cada push ou pull request usando Python 3.11.

## Executar com Docker

Crie .env a partir de .env.example e configure OPENAI_API_KEY. Depois rode:

    docker compose up --build

A API ficará em http://127.0.0.1:8000/docs. O Compose mantém documentos, índice vetorial e histórico no diretório data/.

## Preparar para publicar no GitHub

Antes de publicar, confirme que .env não aparece em git status e que nenhum documento operacional privado está no repositório. O .gitignore exclui chaves, uploads, índice Chroma e histórico SQLite.

Se ainda não houver um repositório Git local:

    git init
    git add .
    git status --short
    git commit -m "Build FoodOps AI RAG"

Depois, crie um repositório vazio no GitHub e siga os comandos que o próprio GitHub mostrar para conectar origin e enviar o código. A API não possui autenticação de usuários; use-a apenas localmente ou acrescente autenticação antes de expô-la publicamente.
