---
license: mit
tags:
- text-classification
- scikit-learn
- portuguese
library_name: sklearn
---

# FoodIntent ML

Classificador experimental para mensagens de atendimento de delivery em português. Toda previsão requer revisão humana.

## Avaliação

As métricas são geradas pelo treinamento e publicadas a partir de `reports/metrics.json`. O cartão separa teste SID e teste curado. Não publique valores antes de executar a versão atual do treinamento.

## Dados e limitações

O SID é sintético e multilíngue. O suplemento de delivery também é sintético e escrito para este projeto. O teste curado é pequeno e não é independente. Métricas não estimam desempenho em clientes reais. As probabilidades não são calibradas.

Dataset SID: https://huggingface.co/datasets/luigicfilho/sid
