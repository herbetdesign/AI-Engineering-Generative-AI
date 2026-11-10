# Classificação de intenções em delivery: o que um erro revelou sobre o benchmark

## Resumo

Este projeto avalia um baseline TF-IDF para classificar mensagens de atendimento em cinco intenções: status do pedido, cancelamento, pagamento, reembolso e problema na entrega. Uma chamada real de teste mostrou uma falha importante: “Meu pedido está atrasado” foi classificada como cancelamento, com confiança próxima de 1. O erro levou à revisão do mapeamento dos dados e à criação de um teste separado com frases de delivery escritas para o projeto.

## Problema e classes

A ideia é encaminhar mensagens para a fila operacional adequada. Status e atraso compartilham vocabulário com cancelamento, pagamento e reclamações. O modelo precisa distinguir a intenção, não apenas identificar palavras como “pedido” ou “entrega”.

## Dados e preparação

A fonte externa é o [SID — Synthetic Intent Dataset](https://huggingface.co/datasets/luigicfilho/sid), licença MIT. A fonte é sintética, multilíngue e baseada em templates; filtramos português. O rótulo de reclamação de entrega na fonte é amplo, então foi removido do mapeamento automático para evitar tratá-lo como “item incorreto”.

Para acrescentar linguagem de delivery, o projeto inclui um pequeno suplemento escrito manualmente, também sintético. Ele tem splits separados de treino, validação e teste. Os exemplos do suplemento recebem peso maior no treino e a validação curada seleciona o modelo. A frase observada no erro (“Meu pedido está atrasado”) fica somente no teste curado.

Esse teste serve como diagnóstico de regressão, mas ainda é pequeno e criado pela mesma equipe. Não é validação independente nem evidência de desempenho em clientes.

## Método

Comparamos TF-IDF de palavras com Logistic Regression e Multinomial Naive Bayes. Selecionamos pelo macro F1 da validação curada. Reportamos métricas em dois conjuntos separados: o teste do SID e o teste curado específico de delivery. Isso deixa visível quando o resultado de um benchmark sintético mais amplo não se transfere ao vocabulário operacional pretendido.

## Resultados

Execute novamente `python -m foodintent_ml.train` após atualizar o código. Preencha com os resultados reais gerados em `reports/metrics.json`:

- Modelo selecionado: **[resultado]**
- SID test: accuracy **[resultado]**, macro F1 **[resultado]**
- Teste curado: accuracy **[resultado]**, macro F1 **[resultado]**
- “Meu pedido está atrasado”: **[intenção prevista e confiança]**
- Erros por classe e matriz de confusão: **[análise dos arquivos gerados]**

A métrica anterior de 0,9811 de accuracy e 0,9722 de macro F1 veio do teste SID antes da inclusão do suplemento; ela não mede a falha observada em linguagem natural de delivery e não deve ser divulgada como resultado final do projeto.

## API e segurança operacional

A API FastAPI expõe `GET /health` e `POST /classify`. Como as probabilidades não foram calibradas e o benchmark não é representativo, toda previsão recomenda revisão humana. Não automatize cancelamentos, reembolsos ou ações com impacto para clientes.

## Próximos experimentos

- Obter uma amostra pt-BR de delivery revisada por anotadores independentes e com licença clara.
- Aumentar o teste curado sem reutilizar frases do treino.
- Inspecionar falsas previsões de cancelamento e itens incorretos.
- Calibrar as probabilidades somente com dados adequados.
