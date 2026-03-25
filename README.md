# Maintenance Request Classification

## PT-BR

Projeto de classificação supervisionada para **rotear solicitações de manutenção** a partir de texto curto e contexto operacional, inspirado em cenários em que equipes precisam decidir rapidamente qual tipo de intervenção deve ser encaminhado com base em colunas como descrição, localização e status do chamado.

### Base pública utilizada

O projeto usa uma **amostra pública real** do dataset **NYC 311 Service Requests**, acessado pela API aberta da cidade de Nova York.

Fonte:
- [NYC 311 Service Requests](https://catalog.data.gov/dataset/311-service-requests-from-2010-to-present)

O recorte selecionado inclui chamados mais próximos de manutenção urbana e infraestrutura, como:
- `Street Condition`
- `Street Light Condition`
- `Sidewalk Condition`
- `Water System`
- `Sewer`
- `Root/Sewer/Sidewalk Condition`
- `Missed Collection`
- `Damaged Tree`

### Objetivo analítico

O alvo do modelo não é o `complaint_type` original em si, mas um **grupo operacional de manutenção** derivado dele, para simular o roteamento interno de chamados:

- `pavement_surface`
- `pedestrian_infrastructure`
- `lighting`
- `water_network`
- `sanitation`
- `urban_forestry`

Assim, o projeto se aproxima de um problema corporativo real: **dado um texto descritivo e o contexto da ocorrência, prever que tipo de manutenção deve ser acionado**.

### Variáveis usadas

As features foram montadas a partir de:
- `descriptor`
- `borough`
- `location_type`
- `street_name`
- `agency`
- `status`

Essas colunas são concatenadas em um campo textual único para representar o chamado de forma compacta e próxima da triagem operacional.

### Técnicas usadas

- limpeza e preparação tabular com `pandas`
- engenharia de features textuais concatenando descrição e contexto
- vetorização com `TF-IDF`
- classificação supervisionada multiclasse
- comparação entre modelos
- avaliação com `accuracy`, `macro F1` e `weighted F1`
- matriz de confusão para leitura dos erros por classe

### Modelos comparados

- `Logistic Regression`
- `Linear SVC`
- `Multinomial Naive Bayes`

### Bibliotecas e frameworks

- `pandas`
  Para ingestão, limpeza e transformação da base pública.
- `requests`
  Para download reproduzível da amostra via API pública.
- `scikit-learn`
  Para `TF-IDF`, split estratificado, pipelines e classificadores.
- `matplotlib`
  Para salvar a matriz de confusão.
- `streamlit`
  Para o painel interativo.
- `plotly`
  Para gráficos comparativos e exploração da base.

### Interface

O `Streamlit` foi desenhado para funcionar como uma bancada de validação:
- simular um novo chamado e prever o grupo de manutenção
- comparar métricas dos modelos
- inspecionar a distribuição da base processada

### Estrutura

- [main.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/maintenance-request-classification/main.py)
- [app.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/maintenance-request-classification/app.py)
- [src/data_pipeline.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/maintenance-request-classification/src/data_pipeline.py)
- [src/modeling.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/maintenance-request-classification/src/modeling.py)
- [scripts/download_dataset.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/maintenance-request-classification/scripts/download_dataset.py)

### Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
streamlit run app.py
```

---

## EN

Supervised classification project for **routing maintenance requests** from short text and operational context, inspired by scenarios where teams need to decide what kind of maintenance action should be triggered based on fields such as description, location, and request status.

### Public dataset

This project uses a **real public sample** from the **NYC 311 Service Requests** dataset, accessed through New York City's open API.

Source:
- [NYC 311 Service Requests](https://catalog.data.gov/dataset/311-service-requests-from-2010-to-present)

The selected subset focuses on urban infrastructure and maintenance-related requests such as:
- `Street Condition`
- `Street Light Condition`
- `Sidewalk Condition`
- `Water System`
- `Sewer`
- `Root/Sewer/Sidewalk Condition`
- `Missed Collection`
- `Damaged Tree`

### Analytical goal

The target is not the original `complaint_type` itself, but an **operational maintenance group** derived from it, in order to simulate internal routing:

- `pavement_surface`
- `pedestrian_infrastructure`
- `lighting`
- `water_network`
- `sanitation`
- `urban_forestry`

### Features used

The model builds its signal from:
- `descriptor`
- `borough`
- `location_type`
- `street_name`
- `agency`
- `status`

These columns are concatenated into a compact textual representation of the request.

### Techniques

- tabular preprocessing with `pandas`
- text feature engineering by concatenating description and context
- `TF-IDF` vectorization
- multiclass supervised classification
- model benchmarking
- evaluation with `accuracy`, `macro F1`, and `weighted F1`
- confusion matrix for error analysis

### Models compared

- `Logistic Regression`
- `Linear SVC`
- `Multinomial Naive Bayes`

### Libraries and frameworks

- `pandas`
- `requests`
- `scikit-learn`
- `matplotlib`
- `streamlit`
- `plotly`
