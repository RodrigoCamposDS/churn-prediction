# 📝 Churn Prediction Report

---

## 🎯 Objetivo do Projeto
O objetivo deste projeto é prever a probabilidade de churn dos clientes de uma plataforma de streaming.

---

## 📊 Descrição dos Dados
- **Total de Registros**: 10.000
- **Número de Variáveis**: 20
- **Variável Alvo**: `Churn` (Sim/Não)

---

## 🧠 Modelos Utilizados
1. **Regressão Logística**
2. **Random Forest**
3. **Gradient Boosting**

---

## 📉 Resultados
- **AUC-ROC**: **0.92**
- **Recall**: **85%**

---

## 📈 Gráficos
### 🔗 [Distribuição das Classes - Antes e Depois do Balanceamento](figures/class_distribution.png)
### 🔗 [Curva ROC - Random Forest](figures/roc_curve_random_forest.png)
### 🔗 [Curva Precision-Recall - Logistic Regression](figures/precision_recall_logistic.png)
### 🔗 [Threshold Metrics - Random Forest](figures/threshold_metrics_rf.png)

---

## 📄 Conclusão
O modelo **Random Forest** apresentou o melhor desempenho, com **alto recall**, o que é essencial para um problema de churn.