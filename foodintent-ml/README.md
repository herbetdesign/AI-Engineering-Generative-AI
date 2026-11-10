# FoodIntent ML

Classificador experimental de intenções em mensagens de atendimento de delivery, em português brasileiro. O projeto compara TF-IDF com Logistic Regression e Multinomial Naive Bayes, oferece API FastAPI e notebook Kaggle.

## Dados e limitações

O SID (licença MIT) é sintético e multilíngue. Filtramos português e usamos somente intenções de status, cancelamento, pagamento e reembolso. A intenção ampla de reclamação de entrega foi removida do treino por ser ambígua. Para representar pedidos incorretos/faltantes, incluímos um suplemento sintético escrito para este projeto em `data/curated/`.

Todos os dados são sintéticos; não são mensagens de clientes. A primeira chamada manual à API errou “Meu pedido está atrasado”, prevendo cancelamento com confiança alta. Por isso o projeto agora reporta um teste curado separado e mantém revisão humana obrigatória na API. Métricas não estimam desempenho em produção.

## Instalação (PowerShell)

Requer Python 3.11 (>=3.11,<3.12):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,publish]"
```

## Treinar e avaliar

```powershell
python -m foodintent_ml.train
```

A seleção de modelo usa macro F1 no conjunto curado de validação. O script reporta separadamente o teste do SID e o teste curado de delivery. O caso “Meu pedido está atrasado” está no teste curado, fora do treino. Os resultados são gravados em `reports/metrics.json` e `reports/curated_test_predictions.json`; o modelo vai para `models/foodintent_pipeline.joblib`.

## API

Depois do treinamento:

```powershell
uvicorn foodintent_ml.api:app --reload
```

- `GET /health`
- `POST /classify`: `{"message": "Meu pedido está atrasado"}`

A API marca todas as previsões para revisão humana até que haja avaliação representativa e calibração. A confiança não é garantia de acerto.

## Kaggle

Importe `notebooks/foodintent_nlp_classification.ipynb`, habilite Internet e execute todas as células. O notebook mostra os dois testes separados, matriz de confusão e erros. Use apenas métricas geradas pela execução.

## Publicar no Hugging Face

Após treinar, entre na sua conta com `hf auth login` e publique o modelo com:

```powershell
python -m foodintent_ml.publish seu-usuario/foodintent-ml
```

Não coloque tokens no repositório nem no chat. A publicação inclui métricas reais e limitações.

## Fonte

SID — https://huggingface.co/datasets/luigicfilho/sid (MIT). Os CSVs do dataset são baixados da fonte; o suplemento curado do projeto está incluído e identificado como sintético.
