#  **Churn Prediction Project**

Este projeto tem como objetivo desenvolver um **modelo preditivo para identificar clientes com maior propensão ao churn (_Churned = 1_)**, aplicando técnicas de Machine Learning avançadas. O foco principal é maximizar o **recall** para garantir que a maior parte dos clientes propensos ao churn sejam identificados, reduzindo o impacto financeiro da perda de clientes.

Aplicamos **Logistic Regression** e **Random Forest** como principais algoritmos, e foram exploradas técnicas como **Conformal Predictions**, **threshold ótimo**, **balanceamento de classes**, **log-loss** e **curvas de aprendizado** para avaliar a robustez dos modelos.

---

##  **Objetivos do Projeto**
1. **Prever clientes propensos ao churn** para auxiliar na retenção proativa.
2. **Reduzir o impacto financeiro** do churn, priorizando a minimização dos falsos negativos.
3. **Avaliar a incerteza nas previsões** utilizando intervalos de conformidade via Conformal Predictions.
4. **Aplicar técnicas avançadas de validação e regularização**, como threshold ótimo, OOB score e análise de log-loss.

---

##  **Pipeline do Projeto**

O pipeline do projeto foi dividido em etapas que garantem um fluxo contínuo de análise e modelagem:

1. **Análise Exploratória dos Dados (EDA)**:
   - Estatísticas descritivas e visualizações.
   - Verificação de **distribuições e valores ausentes**.
   - Criação de gráficos de correlação e análise de outliers.

2. **Engenharia de Features**:
   - Imputação de dados ausentes utilizando **KNN Imputer**.
   - Criação de novas features relacionadas ao uso da plataforma, como `time_satisfaction` e `usage_stability`.
   - Normalização dos dados com **MinMaxScaler**.
   - Codificação de variáveis categóricas.

3. **Balanceamento de Classes**:
   - aplicou-se técnicas de **undersampling com NearMiss** para equilibrar a proporção de classes (churn = 1 / não churn = 0) em **50:50**.

4. **Modelagem**:
   - Foram testados dois modelos principais:
     - **Logistic Regression** com regularização ElasticNet.
     - **Random Forest** com ajuste de hiperparâmetros.
   - Aplicação de **threshold ótimo** para maximizar o recall.
   - Avaliação de métricas como **log-loss**, **OOB Score** (Out-of-Bag), e **AUC-ROC**.

5. **Avaliação e Interpretação dos Resultados**:
   - Matriz de confusão, curva ROC, curva Precision-Recall.
   - Aplicação de **Conformal Predictions** para gerar intervalos de conformidade.

---

##  **Modelos Utilizados**

| **Modelo**            | **Hiperparâmetros Ajustados**                          | **Técnicas Aplicadas**                  |
|-----------------------|--------------------------------------------------------|-----------------------------------------|
| Logistic Regression    | `C=0.1`, `penalty='elasticnet'`, `l1_ratio=0.1`       | Threshold ótimo, Log-Loss, Conformal Predictions |
| Random Forest          | `n_estimators=100`, `max_depth=5`, `class_weight='balanced'` | OOB Score, Conformal Predictions       |

---

##  **Conformal Predictions Aplicados**
Utilizados os seguintes métodos de conformal predictions para adicionar intervalos de incerteza nas previsões:

- **`score`**: Baseado nas pontuações preditivas do modelo.
- **`lac`**: Conformal adjustment com base em conformal scores.

### **Resultados dos Intervalos de Conformidade:**

| **Modelo**            | **Método** | **Acurácia** | **Tamanho Médio do Intervalo** |
|-----------------------|------------|--------------|---------------------------------|
| Logistic Regression    | score      | 78,1%        | 0.610                           |
| Logistic Regression    | lac        | 78,1%        | 0.609                           |
| Random Forest          | score      | 78,7%        | 0.674                           |
| Random Forest          | lac        | 78,7%        | 0.676                           |

---

##  **Principais Métricas Avaliadas**

### 1. **Threshold Ótimo Ajustado**

O threshold foi ajustado para priorizar **recall** em ambos os modelos, utilizando as seguintes métricas:

| **Modelo**            | **Threshold Padrão** | **Threshold Ótimo** | **Recall** | **Precision** | **Log-Loss** |
|-----------------------|----------------------|---------------------|------------|---------------|--------------|
| Logistic Regression    | 0.50                 | 0.347               | 82%        | 61%           | 0.562        |
| Random Forest          | 0.50                 | 0.367               | 75%        | 68%           | 0.499        |

### 2. **Log-Loss (Perda Logarítmica)**

A perda logarítmica foi utilizada para avaliar a qualidade das probabilidades previstas pelos modelos.

| **Modelo**            | **Log-Loss (Treino)** | **Log-Loss (Teste)** |
|-----------------------|-----------------------|----------------------|
| Logistic Regression    | 0.421                 | 0.562                |
| Random Forest          | 0.398                 | 0.499                |

### 3. **OOB Score (Out-of-Bag Score) - Random Forest**

A métrica OOB foi utilizada para avaliar o desempenho do modelo Random Forest sem a necessidade de uma validação cruzada explícita:

- **OOB Score**: **77.9%**

##  **Insights e Conclusões**

1. **Logistic Regression com threshold ajustado** apresentou **o menor custo total**, devido ao maior recall e menor log-loss.
2. **Random Forest** apresentou um **OOB Score consistente** e maior robustez em relação à incerteza nas previsões.
3. **Conformal Predictions** ajudaram a adicionar uma camada de confiança nas previsões, permitindo decisões mais informadas.

---

##  **Próximos Passos**

1. Explorar novos métodos de conformal predictions, como `beta` e `raps`.
2. Implementar monitoramento contínuo para identificar possíveis **data drifts**.
3. Avaliar o impacto de outras técnicas de balanceamento de classes.

---

##  **Reproduzir o Projeto**

Clone o repositório e execute o pipeline:

```bash
git clone https://github.com/RodrigoCamposDS/churn-prediction.git
cd churn-prediction
