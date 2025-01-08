# **Relatório Final: Projeto Churn Prediction**

## **1. Objetivo do Projeto**
- **Contexto**: Desenvolver um modelo preditivo para identificar clientes com maior propensão ao churn (_Churned = 1_).
- **Objetivo**: Fornecer insights acionáveis para estratégias de retenção.
- **Pergunta Central**: *"Quais clientes têm maior probabilidade de churnar, e como podem ser identificados com precisão?"*

---

## **2. Abordagem Utilizada (CRISP-DM)**

### **1. Compreensão do Negócio**
O churn impacta diretamente a receita da empresa, tornando essencial prever quais clientes estão em risco para implementar ações preventivas. Neste projeto, utilizamos métodos de conformal predictions para melhorar a confiabilidade das previsões feitas pelos modelos de **Logistic Regression** e **Random Forest**, fornecendo intervalos de conformidade que indicam a incerteza associada a cada previsão.

---

## **3. Modelagem com Conformal Predictions**

### **Modelos e Métodos Testados**

| **Modelo**            | **Método** | **Acurácia** | **Tamanho Médio do Intervalo** |
|-----------------------|------------|--------------|---------------------------------|
| Logistic Regression    | score      | 78,1%        | 0.610                           |
| Logistic Regression    | lac        | 78,1%        | 0.609                           |
| Random Forest          | score      | 78,7%        | 0.674                           |
| Random Forest          | lac        | 78,7%        | 0.676                          |

---

## **4. Análise dos Resultados**

### **Conclusões dos Resultados de Conformal Predictions**

1. **Logistic Regression**:
   - Apresentou **acurácia de 78,1%** para ambos os métodos (`score` e `lac`).
   - Os intervalos gerados foram relativamente curtos, indicando que o modelo tem uma boa capacidade de previsão com baixa incerteza.

2. **Random Forest**:
   - Apresentou **acurácia de 78,7%**, superior à Regressão Logística.
   - Os tamanhos médios dos intervalos de conformidade foram maiores em comparação à Regressão Logística, o que indica maior incerteza nas previsões.

---

## **5. Avaliação dos Intervalos de Conformidade**

### **Distribuição dos Tamanhos dos Intervalos**

Foi analisada a distribuição dos tamanhos dos intervalos gerados pelos métodos de conformal predictions. Abaixo estão os gráficos das distribuições:

1. **Logistic Regression - Método `score`**:
   - A maioria dos intervalos está entre **0.6 e 0.65**.
   
2. **Random Forest - Método `score`**:
   - Intervalos maiores foram observados, variando entre **0.6 e 0.7**, indicando maior incerteza.

---

## **6. Conclusão e Próximos Passos**

### **Insights Obtidos**:
- As variáveis mais importantes combinadas para geração de features e melhor predição foram:
  - **`Time_on_platform`**, **`Usage`**, **`Avg_rating`**, **`Num_active_profiles`** e **`Num_streaming_services`**.
- A utilização do **Lasso**:
  - Demonstrou ser uma estratégia eficiente ao **maximizar o recall**, permitindo que o modelo priorizasse a identificação de clientes churnados, reduzindo os falsos negativos.
  - A regularização eliminou variáveis menos relevantes, simplificando o modelo sem comprometer significativamente o desempenho geral.
  - O **threshold ajustado** com o Lasso contribuiu para um melhor equilíbrio entre recall e custo, tornando a abordagem mais vantajosa financeiramente.
- ElasticNet e Random Forest forneceram equilíbrio entre explicação e performance.
- O modelo priorizou o recall para capturar o maior número de clientes churnados.

---

## **7. Próximos Passos**
1. **Validar os Modelos em Dados Reais**:
   - Implementar uma rotina de avaliação contínua para monitorar a performance dos modelos.
2. **Testar Novos Métodos de Conformal Predictions**:
   - Avaliar métodos adicionais de conformal predictions, como **beta** e **raps**, para identificar se há melhorias na precisão dos intervalos gerados.

---

## **8. Código Utilizado para Conformal Predictions**

```python
import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Criar dados sintéticos
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Métodos de conformal predictions válidos para problemas binários
methods = ["score", "lac"]

results = []

# Loop para testar cada modelo e método
for model, model_name in [(model_lr, "Logistic Regression"), (model_rf, "Random Forest")]:
    for method in methods:
        try:
            # Aplicar o MAPIE
            mapie = MapieClassifier(estimator=model, method=method)
            mapie.fit(X_train, y_train)
            y_pred, y_pred_intervals = mapie.predict(X_test, alpha=0.1)

            # Garantir que os intervalos são numéricos
            y_pred_intervals = y_pred_intervals.astype(float)

            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            interval_sizes = np.mean(np.abs(y_pred_intervals[:, 1] - y_pred_intervals[:, 0]))

            # Armazenar resultados
            results.append({
                "Modelo": model_name,
                "Método": method,
                "Acurácia": accuracy,
                "Tamanho Médio do Intervalo": interval_sizes
            })
        except Exception as e:
            print(f"Erro com o modelo {model_name} usando o método {method}: {e}")

# Criar um DataFrame com os resultados
df_results = pd.DataFrame(results)

# Exibir a tabela
print(df_results)