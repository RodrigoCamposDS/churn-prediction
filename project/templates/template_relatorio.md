










## **1. Objetivo do Projeto**
- **Contexto**: Desenvolver um modelo preditivo para identificar clientes com maior propensão ao churn (_Churned = 1_).
- **Objetivo**: Fornecer insights acionáveis para estratégias de retenção.
- **Pergunta Central**: *"Quais clientes têm maior probabilidade de churnar, e como podem ser identificados com precisão?"*

---

## **2. Abordagem Utilizada (CRISP-DM)**

### **1. Compreensão do Negócio**
- O churn impacta diretamente a receita da empresa, tornando essencial prever quais clientes estão em risco para implementar ações preventivas;
- **Foco do modelo**: Maximizar o **recall**, garantindo a identificação correta da maioria dos clientes propensos ao churn.

---

### **2. Compreensão dos Dados**
- **Dados Originais**:

  - Registros:

    - Código de identificação do cliente;
    - Idade do cliente;
    - Gênero do cliente;
    - Tipo de conta;
    - Avaliação média dos conteúdos da plataforma;
    - Tempo on-line na plataforma;
    - Número de perfis ativos na plataforma;
    - Quantidade de serviços de _streaming_ que o cliente possui;
    - Quantidade de dispositivos conectados à conta;
    - Se o cliente cancelou a conta ou não - _CHURNED_;

- **Problemas Identificados**:
  - Esparsidade alta com 88% em `Num_streaming_services`, 32% em `Devices_connected` e `Subscription_type`, levando em conta os campos vazios;
  - Desbalanceamento da classe _Churned_ (80% "No" e 20% "Yes"), antes da interpolação e (77% "No" e 22% "Yes"), após interpolação dos dados.
  - Quantidade de 12707 linhas duplicadas encontradas;

---

### **3. Preparação dos Dados**
#### **Etapas Realizadas**:
1. **Limpeza de Dados**:
   - Dropado a coluna `User_id`;
   - Remoção de linhas totalmente duplicadas;
   - Tratamento de valores ausentes com imputação via _K-Nearest Neighbors (KNN)_ e preenchimento via _Random Forest Classifier_;
   - Transformação dos valores Churned 0 E 1 por 'No' e 'Yes';
   - Valores _floats_ transformados em inteiros;
   
2. **Criação de Novas Features**:
   - `Time_satisfaction = Avg_rating / Time_on_platform`
   - `Usage = Time_on_platform / Age`   
   - `Usage_stability = Time_on_platform / Num_active_profiles`

3. **Codificação e Escalonamento**:
 - Normalização dos dados com o método Min-Max;
 - Variáveis categóricas foram codificadas Churned 0 E 1 por 'No' e 'Yes';

4. **Balanceamento das Classes**:
   - Aplicado **NearMiss** para equilibrar as classes (proporção 50:50).


---

### **4. Modelagem**
#### **Modelos Testados**:
- **Regressão Logística**:
  - **Hiperparâmetros**:
    - penalty = 'elasticnet',
    - solver = 'saga',
    - max_iter = 000000,
    - class_weight = None,
    - l1_ratio = 0.5,
    - C = 10

  - **Desempenho**:
    - Recall médio: **0.65**
    - Precision médio: **0.72**
  - **Desempenho com ponto de corte ótimo**:
    - Recall médio: **0.82**
    - Precision médio: **0.61**


- **Random Forest**:
  - **Hiperparâmetros**:
    - n_estimators = 100,
    - max_depth = 5,
    - min_samples_split = 2,
    - min_samples_leaf = 5,
    - max_features = 'sqrt',
    - oob_score = True,
    - bootstrap = True,
    - random_state = 42,
    -class_weight = None

  - **Desempenho**:
    - Recall médio: **0.63**
    - Precision médio: **0.78**
  - **Desempenho com ponto de corte ótimo**:
    - Recall médio: **0.75**
    - Precision médio: **0.68**
---

### **5. Avaliação**
#### **Métricas-Chave**:
- **Recall**: Foco principal, para capturar a classe _Churned_ = 1.
- **Precision**: Avaliação da precisão das predições positivas.
- **AUC-ROC e AUC-PR**: Avaliar o equilíbrio geral do modelo.

###  **Tabela de Resultados**


<h3><b>Resultados Finais</b></h3>
<table border="1" style="border-collapse: collapse; width: 100%; text-align: center;">
    <thead>
        <tr>
            <th>Modelo</th>
            <th>Threshold</th>
            <th>Falsos Positivos (FP)</th>
            <th>Falsos Negativos (FN)</th>
            <th>Custo Total (R$)</th>
        </tr>
    </thead>
    <tbody>
        {%- for resultado in resultados %}
        <tr>
            <td>{{ resultado['Modelo'] }}</td>
            <td>{{ resultado['Threshold'] }}</td>
            <td>{{ resultado['FP'] }}</td>
            <td>{{ resultado['FN'] }}</td>
            <td>{{ resultado['Custo Total (R$)'] }}</td>
        </tr>
        {%- endfor %}
    </tbody>
</table>

#### **Visualizações**:
1. **Curva ROC e Precision-Recall** para cada modelo.
2. **Distribuição das Probabilidades Previstadas**.
3. **Importância das Features (Random Forest)**.

---

### **6. Conclusão e Próximos Passos**

#### **Insights Obtidos**:
- As variáveis mais importantes combinadas para geração de features e melhor predição foram:
  - **`Time_on_platform`**, **`Usage`**, **`Avg_rating`**, **`Num_active_profiles`** e **`Num_streaming_services`**.
- A utilização do **Lasso**:
  - Demonstrou ser uma estratégia eficiente ao **maximizar o recall**, permitindo que o modelo priorizasse a identificação de clientes churnados, reduzindo os falsos negativos.
  - A regularização eliminou variáveis menos relevantes, simplificando o modelo sem comprometer significativamente o desempenho geral.
  - O **threshold ajustado** com o Lasso contribuiu para um melhor equilíbrio entre recall e custo, tornando a abordagem mais vantajosa financeiramente.
- ElasticNet e Random Forest forneceram equilíbrio entre explicação e performance.
- O modelo priorizou o recall para capturar o maior número de clientes churnados.

#### **Próximos Passos**:
1. Avaliar o modelo em novos dados para validação contínua, garantindo que esses dados sejam consistentes com os utilizados no treinamento e estejam devidamente pré-processados.  
   - **Observação**: O arquivo com os dados interpolados foi salvo para referência. Se os novos dados forem interpolados novamente, podem ocorrer variações mínimas nos resultados devido à metodologia utilizada.
2. Implementar um pipeline para monitorar desempenho e identificar possíveis drifts nos dados.


---

## **Anexos**
1. **Código-Fonte Completo**.
2. **Visualizações**:
   - Display com Matriz confusão.
   - Curva de aprendizado do modelo.
   - Gráfico de métricas em função de Threshold.
   - Curva ROC e Precision-Recall.
   - Histogramas de probabilidades.
   - Gráfico de Visualização de uma Árvore da Floresta Aleatória
3. **Tabela de Hiperparâmetros Otimizados**.
# **Análise Comparativa dos Modelos e Custos - Visão de Negócio**

---
## **1. Contexto**
O objetivo principal foi avaliar o impacto financeiro do uso dos modelos (Regressão Logística e Random Forest) com e sem ajuste de threshold ótimo. A análise considera os seguintes fatores:

- **Falsos Positivos (FP)**: Custam **R$ 5** cada, pois representam esforços de retenção aplicados em clientes que não churnaram.

- **Falsos Negativos (FN)**: Custam **R$ 25** cada, pois representam clientes que churnaram sem serem identificados, resultando em perda de receita.

**Nota**: Os valores atribuídos aos custos são fictícios, baseados em práticas de mercado, e não possuem fonte específica. A intenção é apenas ilustrar o impacto relativo entre falsos positivos e falsos negativos.

---

<h2><b>2. Comparação de Resultados</b></h2>
<p>Os custos totais para cada cenário foram calculados com base nas matrizes de confusão geradas:</p>

<table border="1" style="border-collapse: collapse; width: 100%; text-align: center;">
    <thead>
        <tr>
            <th>Modelo</th>
            <th>Threshold</th>
            <th>Falsos Positivos (FP)</th>
            <th>Custo FP (R$)</th>
            <th>Falsos Negativos (FN)</th>
            <th>Custo FN (R$)</th>
            <th>Custo Total (R$)</th>
        </tr>
    </thead>
    <tbody>
        {%- for resultado in resultados %}
        <tr>
            <td>{{ resultado['Modelo'] }}</td>
            <td>{{ resultado['Threshold'] }}</td>
            <td>{{ resultado['FP'] }}</td>
            <td>{{ '{:,.2f}'.format(resultado['FP'] * 5) }}</td>
            <td>{{ resultado['FN'] }}</td>
            <td>{{ '{:,.2f}'.format(resultado['FN'] * 25) }}</td>
            <td><b>{{ '{:,.2f}'.format(resultado['Custo Total (R$)']) }}</b></td>
        </tr>
        {%- endfor %}
    </tbody>
</table>

---


## **3. Observações dos Resultados**

### **Regressão Logística**:
1. **Sem threshold ótimo**:
   - Custos elevados devido ao maior número de **falsos negativos** (1203).
   - Recall menor compromete a identificação de clientes em risco de churn, resultando em um custo total de **R$ 34.450,00**.

2. **Com threshold ótimo (0.347)**:
   - Houve um aumento nos **falsos positivos** (de 875 para 1177), mas a redução significativa nos **falsos negativos** (de 1203 para 621) compensou o custo total.
   - **Custo total reduzido em R$ 13.040,00 (38%)** em relação ao threshold padrão.

---

### **Random Forest**:
1. **Sem threshold ótimo**:
   - Falsos negativos extremamente elevados (1903) geraram custos totalizando **R$ 52.015,00**.
   - Apesar do recall padrão mais baixo, o modelo gerou menor custo com falsos positivos (888).

2. **Com threshold ótimo (0.367)**:
   - O ajuste do threshold reduziu os falsos negativos de forma significativa (de 1903 para 1272), embora os falsos positivos tenham aumentado.
   - **Custo total reduzido em R$ 11.360,00 (22%)** em relação ao threshold padrão.

---

## **4. Conclusão: Melhor Custo-Benefício**
1. **Modelo Vencedor**:
   - A **Regressão Logística com Threshold Ótimo (0.347)** apresentou o **menor custo total (R$ 21.410,00)**, sendo a solução mais rentável para o negócio.

2. **Impacto Estratégico**:
   - **Redução significativa nos falsos negativos**: Clientes churnados foram corretamente identificados, reduzindo perdas financeiras críticas.
   - **Decisão orientada por recall e custo-benefício**: O aumento dos falsos positivos foi aceitável para priorizar a retenção de clientes em risco.

---

## **5. Recomendações**
1. Implementar a **Regressão Logística com Threshold Ótimo** como a solução principal para identificar clientes em risco de churn.
2. Continuar monitorando as métricas de desempenho e custos, ajustando o threshold conforme necessário.
3. Realizar análises de impacto regularmente para garantir que as decisões continuem alinhadas com as metas financeiras e estratégicas do negócio.

---
## Deployment - Implementação