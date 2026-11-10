# FoodIntent ML

Classificador experimental de intenções em mensagens de atendimento de delivery, em português brasileiro. O projeto compara TF-IDF com Logistic Regression e Multinomial Naive Bayes, fornece uma API FastAPI e inclui um notebook para Kaggle.

## Limitação dos dados

O projeto usa o [SID — Synthetic Intent Dataset](https://huggingface.co/datasets/luigicfilho/sid), licença MIT. Ele contém exemplos sintéticos multilíngues, não mensagens reais de clientes de delivery. Os resultados são uma avaliação exploratória do benchmark e não representam desempenho em produção.

O pipeline filtra português e agrupa intenções da fonte em cinco classes amplas: status do pedido, cancelamento, problema de pagamento, reembolso e problema na entrega. O mapeamento e suas ambiguidades estão documentados no notebook e no artigo.

## Instalação (Windows PowerShell)

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Treino e avaliação

```powershell
python -m foodintent_ml.train
```

O modelo é selecionado pela macro F1 no conjunto de validação. As métricas finais são calculadas no teste e salvas em `reports/metrics.json`; o pipeline fica em `models/foodintent_pipeline.joblib`. Os resultados só são conhecidos depois que o treinamento é executado.

## API

Depois do treino:

```powershell
uvicorn foodintent_ml.api:app --reload
```

- `GET /health`
- `POST /classify` com `{"message": "Meu pedido está atrasado"}`

A confiança reportada não é calibrada. O endpoint sinaliza revisão humana abaixo de um limiar exploratório; não automatize decisões com impacto ao cliente.

## Notebook Kaggle

Importe `notebooks/foodintent_nlp_classification.ipynb`, habilite Internet e execute todas as células. Ele inclui distribuição de classes, comparação de modelos, métricas, matriz de confusão e inspeção de erros.

## Hugging Face

Após treinar, autentique-se localmente com `hf auth login` e publique usando:

```powershell
python -m foodintent_ml.publish seu-usuario/foodintent-ml
```

O script usa a sessão já autenticada; não armazene tokens no projeto. A publicação depende da conta Hugging Face do autor.

## Fonte e licença

SID — Synthetic Intent Dataset: https://huggingface.co/datasets/luigicfilho/sid (MIT). O projeto baixa os CSVs da fonte, sem redistribuí-los.
