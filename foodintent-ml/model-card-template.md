---
license: mit
tags:
- text-classification
- scikit-learn
- portuguese
library_name: sklearn
---

# FoodIntent ML

Classificador experimental TF-IDF para mensagens de suporte em português.

## Avaliação

Este cartão é gerado pelo comando de publicação a partir de `reports/metrics.json`. Métricas não devem ser informadas antes da execução do treinamento.

## Dados e limitações

Treinado no subconjunto em português do SID, um dataset sintético multilíngue. Não representa clientes reais de delivery. As classes de status e problema de entrega são aproximações amplas. As probabilidades não são calibradas. Não use para decisões automatizadas com impacto ao cliente.

Dataset: https://huggingface.co/datasets/luigicfilho/sid
