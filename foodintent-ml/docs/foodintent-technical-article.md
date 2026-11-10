# Classificação de intenções em mensagens de delivery: um baseline reproduzível

> **Rascunho técnico.** Execute o notebook ou `python -m foodintent_ml.train` antes de completar a seção de resultados. Não há métricas presumidas.

## Problema

Mensagens de suporte precisam ser encaminhadas à equipe correta. Este experimento avalia um classificador baseline para cinco intenções amplas: status do pedido, cancelamento, problema de pagamento, reembolso e problema na entrega.

## Dados e preparação

Usamos o [SID — Synthetic Intent Dataset](https://huggingface.co/datasets/luigicfilho/sid), publicado sob licença MIT. Filtramos as linhas marcadas como português e mapeamos sete intenções de origem para cinco rótulos do projeto.

O conjunto SID é sintético, criado a partir de templates e aumentado automaticamente. Não é uma amostra representativa de conversas reais de delivery. Além disso, `report_delivery_issue` é uma classe abrangente e não significa apenas “pedido incorreto”; `track_order` e `check_order_status` são agrupadas como `order_status`, sem garantir que o pedido esteja atrasado. Essas limitações restringem as conclusões.

## Método

Comparamos dois pipelines TF-IDF:

1. Logistic Regression com balanceamento de classes.
2. Multinomial Naive Bayes.

O TF-IDF é ajustado apenas no conjunto de treino. A macro F1 do conjunto de validação escolhe o candidato; depois, medimos accuracy, macro F1, métricas por classe e matriz de confusão no teste reservado.

## Resultados

Preencha após executar o treinamento; os valores são gravados em `reports/metrics.json`:

- Modelo escolhido pela validação: **[resultado]**
- Accuracy no teste: **[resultado]**
- Macro F1 no teste: **[resultado]**
- Precisão, recall e F1 por classe: **[resultado]**
- Principais confusões observadas: **[análise dos erros]**

Os resultados são do benchmark sintético filtrado. Não devem ser apresentados como desempenho em usuários reais.

## API e uso responsável

O serviço FastAPI oferece `GET /health` e `POST /classify`. A API retorna uma confiança não calibrada e sinaliza revisão humana abaixo de um limiar exploratório de 0,55. É uma demonstração educacional; não deve decidir automaticamente reembolsos ou ações que afetem clientes.

## Próximos experimentos

- Criar um conjunto pt-BR de mensagens de delivery revisado por pessoas e licenciado.
- Separar treino e teste por template/origem para testar generalização.
- Investigar erros por classe e calibrar probabilidades com dados apropriados.
- Comparar TF-IDF de palavras com n-gramas de caracteres para mensagens com erros de digitação.

## Reprodutibilidade

O projeto fixa Python em `>=3.11,<3.12`. Instale com `pip install -e ".[dev]"`, execute `python -m foodintent_ml.train` e rode todas as células do notebook no Kaggle com Internet habilitada.
