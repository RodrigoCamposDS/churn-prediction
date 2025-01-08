import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)
import os

def ajustar_threshold(model, X_train, X_test, y_train, y_test, threshold_precision=0.6, metric='recall', model_name='model'):
    """
    Ajusta o threshold para maximizar uma métrica específica e salva os gráficos com nome dinâmico.

    Parâmetros:
    -----------
    model : modelo treinado
        Modelo de regressão logística ajustado.
    X_train : DataFrame
        Conjunto de dados de treino.
    X_test : DataFrame
        Conjunto de dados de teste.
    y_train : Series
        Rótulos reais do conjunto de treino.
    y_test : Series
        Rótulos reais do conjunto de teste.
    threshold_precision : float, default 0.6
        O valor mínimo de precisão para ajustar o threshold.
    metric : str, default 'recall'
        A métrica para otimizar (pode ser 'recall', 'precision', 'f1' ou 'accuracy').
    model_name : str, default 'model'
        Nome do modelo a ser usado nos nomes dos arquivos salvos.

    Retorna:
    --------
    optimal_threshold : float
        O threshold que maximiza a métrica escolhida.
    y_pred_otimo : array
        As previsões com o threshold ótimo ajustado.
    """
    # Caminho para salvar os gráficos
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Gera thresholds uniformemente espaçados entre 0 e 1
    thresholds = np.linspace(0, 1, 50)
    y_prob_th = model.predict_proba(X_test)[:, 1]

    # Lista para armazenar as métricas
    metric_values = []
    optimal_threshold = 0
    best_metric = 0

    # Iterar pelos thresholds para encontrar o melhor valor da métrica
    for threshold in thresholds:
        y_pred_th = (y_prob_th >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_th, zero_division=0)

        if precision >= threshold_precision:
            if metric == 'recall':
                metric_value = recall_score(y_test, y_pred_th)
            elif metric == 'precision':
                metric_value = precision_score(y_test, y_pred_th)
            elif metric == 'f1':
                metric_value = f1_score(y_test, y_pred_th)
            else:
                raise ValueError("Métrica não reconhecida. Use 'recall', 'precision', ou 'f1'.")

            metric_values.append(metric_value)
            if metric_value > best_metric:
                best_metric = metric_value
                optimal_threshold = threshold

    y_pred_th_otimo = (y_prob_th >= optimal_threshold).astype(int)

    # Gráfico 1: Métricas em função do threshold
    precisao = []
    recalls = []
    f1_scores = []
    acuracias = []

    for threshold in thresholds:
        y_pred_th = (y_prob_th >= threshold).astype(int)
        precisao.append(precision_score(y_test, y_pred_th, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_th))
        f1_scores.append(f1_score(y_test, y_pred_th))
        acuracias.append(accuracy_score(y_test, y_pred_th))

    plt.figure(figsize=(20, 6))
    plt.plot(thresholds, precisao, label='Precisão', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='o')
    plt.plot(thresholds, f1_scores, label='F1-Score', marker='o')
    plt.plot(thresholds, acuracias, label='Acurácia', marker='o')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f"Threshold Ótimo = {optimal_threshold:.2f}")
    plt.axvline(x=0.5, color='purple', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Valor da Métrica', fontsize=12)
    plt.title('Métricas em Função do Threshold Ótimo', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.5)

    # Nome dinâmico para o gráfico
    metrics_fig_path = os.path.join(figures_path, f'{model_name}_threshold_metrics_{metric}.png')
    plt.savefig(metrics_fig_path)
    print(f"Gráfico de métricas salvo")
    plt.show()

    # Gráfico 2: Sensibilidade e Especificidade
    tpr = []
    especificidade = []

    for threshold in thresholds:
        y_pred_th = (y_prob_th >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()
        tpr.append(tp / (tp + fn))
        especificidade.append(tn / (tn + fp))

    plt.figure(figsize=(20, 6))
    plt.plot(thresholds, tpr, label='Sensibilidade', marker='o', color='blue')
    plt.plot(thresholds, especificidade, label='Especificidade', marker='o', color='green')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f"Threshold Ótimo = {optimal_threshold:.2f}")
    plt.axvline(x=0.5, color='purple', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Ponto de Corte (d_0)', fontsize=12)
    plt.ylabel('Valores', fontsize=12)
    plt.title('Sensibilidade e Especificidade para Diferentes Pontos de Corte', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.5)

    # Nome dinâmico para o gráfico
    sensitivity_fig_path = os.path.join(figures_path, f'{model_name}_sensitivity_specificity.png')
    plt.savefig(sensitivity_fig_path)
    print(f"Gráfico de sensibilidade/especificidade salvo")
    plt.show()

    print(f"\nThreshold Ótimo (Precision >= {threshold_precision * 100}%): {optimal_threshold:.4f}")
    print(f"Melhor {metric.capitalize()} com Threshold Ótimo: {best_metric:.4f}")

    return optimal_threshold, y_pred_th_otimo, y_prob_th, thresholds


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def calcular_spearman_logit(model, X_test):
    """
    Função para calcular a correlação de Spearman entre as variáveis de entrada
    e o logit gerado pelo modelo de regressão logística.
    
    Parâmetros:
    -----------
    model : sklearn.linear_model.LogisticRegression
        O modelo de regressão logística treinado.
    X_test : pandas.DataFrame
        Conjunto de teste das variáveis preditoras.
        
    Retorna:
    --------
    spearman_df : pandas.DataFrame
        DataFrame com as variáveis e suas respectivas correlações de Spearman com o logit.
    """
    # Prever as probabilidades para o conjunto de teste
    probabilities_lr = model.predict_proba(X_test)
    
    # Calcular o logit
    logit = np.log(probabilities_lr[:, 1] / (1 - probabilities_lr[:, 1]))
    
    # Garantir que X_test está alinhado com o cálculo do logit
    X_test_df = pd.DataFrame(X_test, columns=model.feature_names_in_)
    
    # Criar um dicionário para armazenar os resultados das correlações
    spearman_results = {}
    
    # Calcular a correlação de Spearman para cada variável
    for column in X_test_df.columns:
        spearman_corr, _ = spearmanr(X_test_df[column], logit)
        spearman_results[column] = spearman_corr
    
    # Converter os resultados em um DataFrame para melhor visualização
    spearman_df = pd.DataFrame(list(spearman_results.items()), columns=['Variable', 'Spearman Correlation'])
    spearman_df = spearman_df.sort_values(by='Spearman Correlation', ascending=False)
    
    # Retornar o DataFrame com as correlações
    return spearman_df


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from IPython.display import display, HTML

def treino_e_teste(model, X_train, y_train, X_test, y_test):
    """
    Função para treinar o modelo, calcular as métricas e exibir os resultados em HTML.
    
    Parâmetros:
    -----------
    model : modelo de aprendizado de máquina
        O modelo a ser treinado.
    X_train : DataFrame
        Conjunto de variáveis preditoras para treino.
    y_train : Series
        Variável alvo para treino.
    X_test : DataFrame
        Conjunto de variáveis preditoras para teste.
    y_test : Series
        Variável alvo para teste.
        
    Exibe as métricas de desempenho para treino e teste em HTML.
    """
    # Treinando o modelo
    model.fit(X_train, y_train)
    
    # Predições para o conjunto de treino e teste
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Cálculo das métricas para o conjunto de treino
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Cálculo das métricas para o conjunto de teste
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Gerar HTML com as métricas para treino e teste
    metrics_html = f"""
    <table border="1" style="border-collapse: collapse; width: 50%;">
        <thead>
            <tr>
                <th style="text-align: left;">Métrica</th>
                <th style="text-align: left;">TREINO</th>
                <th style="text-align: left;">TESTE</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="text-align: left;">Acurácia</td>
                <td style="text-align: left;">{train_accuracy:.4f}</td>
                <td style="text-align: left;">{test_accuracy:.4f}</td>
            </tr>
            <tr>
                <td style="text-align: left;">Precisão</td>
                <td style="text-align: left;">{train_precision:.4f}</td>
                <td style="text-align: left;">{test_precision:.4f}</td>
            </tr>
            <tr>
                <td style="text-align: left;">Recall</td>
                <td style="text-align: left;">{train_recall:.4f}</td>
                <td style="text-align: left;">{test_recall:.4f}</td>
            </tr>
            <tr>
                <td style="text-align: left;">F1-Score</td>
                <td style="text-align: left;">{train_f1:.4f}</td>
                <td style="text-align: left;">{test_f1:.4f}</td>
            </tr>
        </tbody>
    </table>
    """
    
    display(HTML(metrics_html))

    return




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

# Ajuste na função para desenhar os intervalos de conformidade
def plot_conformal_intervals(y_true, y_pred, y_intervals, sample_size=100):
    indices = range(sample_size)

    # Verificar o formato dos intervalos
    if y_intervals.ndim == 2:
        y_lower = y_intervals[:, 0]
        y_upper = y_intervals[:, 1]
    elif y_intervals.ndim == 1:
        y_lower = y_pred - y_intervals
        y_upper = y_pred + y_intervals
    else:
        raise ValueError("Os intervalos de conformidade não estão no formato esperado.")

    # Plotagem
    plt.plot(indices, y_true[:sample_size], 'o', label="Valor Verdadeiro", color='black')
    plt.plot(indices, y_pred[:sample_size], 'x', label="Previsão", color='red')
    plt.fill_between(
        indices,
        y_lower[:sample_size],
        y_upper[:sample_size],
        color='gray',
        alpha=0.5,
        label="Intervalo de Conformidade"
    )
    plt.title("Intervalos de Conformidade nas Previsões")
    plt.xlabel("Amostra")
    plt.legend()
    plt.show()



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from sklearn.metrics import accuracy_score

# Função genérica para aplicar o MAPIE
def apply_mapie(model, X_train, X_test, y_train, y_test, methods=["score", "lac"], alpha=0.1):
    """
    Aplica o MAPIE para conformal predictions em um modelo de classificação.
    
    Parâmetros:
        - model: Modelo de classificação a ser usado (ex: LogisticRegression, RandomForestClassifier).
        - X_train: Dados de treino (features).
        - X_test: Dados de teste (features).
        - y_train: Dados de treino (target).
        - y_test: Dados de teste (target).
        - methods: Lista de métodos de conformal predictions a serem aplicados. Padrão: ["score", "lac"].
        - alpha: Nível de significância para os intervalos de conformidade. Padrão: 0.1.
    
    Retorno:
        - Um DataFrame com os resultados de acurácia e tamanhos médios dos intervalos.
    """
    results = []
    for method in methods:
        try:
            # Aplicar o MAPIE
            mapie = MapieClassifier(estimator=model, method=method)
            mapie.fit(X_train, y_train)
            y_pred, y_pred_intervals = mapie.predict(X_test, alpha=alpha)
            
            # Garantir que os intervalos são numéricos
            y_pred_intervals = y_pred_intervals.astype(float)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            interval_sizes = np.mean(np.abs(y_pred_intervals[:, 1] - y_pred_intervals[:, 0]))
            
            # Armazenar resultados
            results.append({
                "Modelo": type(model).__name__,
                "Método": method,
                "Acurácia": accuracy,
                "Tamanho Médio do Intervalo": interval_sizes
            })
        except Exception as e:
            print(f"Erro com o modelo {type(model).__name__} usando o método {method}: {e}")

    # Retornar os resultados em um DataFrame
    return pd.DataFrame(results)

# --- Exemplo de uso ---

