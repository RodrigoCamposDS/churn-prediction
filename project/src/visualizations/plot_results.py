import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Função para plotar histograma, Q-Q plot, boxplot e gráfico de densidade lado a lado
def plot_continuous_data(coluna):
    # Remover valores NaN
    coluna = coluna.dropna()

    # Verificar se a coluna não está vazia após a remoção de NaN
    if len(coluna) == 0:
        print("Erro: a coluna está vazia após remover valores NaN.")
        return

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))  # Ajustando para 4 gráficos

    # Histograma
    axs[0].hist(coluna, bins="auto", color="skyblue", edgecolor="black")
    axs[0].set_title("Histograma")
    axs[0].set_xlabel("Valor")
    axs[0].set_ylabel("Frequência")

        # Gráfico de Densidade
    sns.kdeplot(coluna, ax=axs[1], fill=True, color="skyblue")  # Atualizado para 'fill=True'
    axs[1].set_title("Gráfico de Densidade")
    axs[1].set_xlabel("Valor")
    axs[1].set_ylabel("Densidade")

    # Q-Q Plot
    stats.probplot(coluna, dist="norm", plot=axs[2])
    axs[2].set_title("Q-Q Plot")

    # Boxplot
    axs[3].boxplot(coluna, vert=True, patch_artist=True, boxprops=dict(facecolor="skyblue", color="black"))
    axs[3].set_title("Boxplot")
    axs[3].set_xlabel("Valor")



    plt.tight_layout()
    plt.show()

    return


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Função para plotar histograma, gráfico de barras e boxplot para dados discretos com cores diferentes
def plot_discrete_data(coluna):
    # Remover valores NaN
    coluna = coluna.dropna()

    # Verificar se a coluna não está vazia após a remoção de NaN
    if len(coluna) == 0:
        print("Erro: a coluna está vazia após remover valores NaN.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(8, 6))  # Ajustando para 3 gráficos

    # Gerando uma paleta de cores com uma cor para cada valor único
    unique_values = np.unique(coluna)
    colors = sns.color_palette("Set2", len(unique_values))

  
    # Histograma
    axs[0].hist(coluna, bins=np.arange(min(coluna), max(coluna) + 1, 1), color=colors[0], edgecolor="black")
    axs[0].set_title("Histograma")
    axs[0].set_xlabel("Valor")
    axs[0].set_ylabel("Frequência")

    # Boxplot
    axs[1].boxplot(coluna, vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue", color="black"))
    axs[1].set_title("Boxplot")
    axs[1].set_xlabel("Valor")

    plt.tight_layout()
    plt.show()

    return

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Função para plotar gráfico de barras para dados categóricos com cores diferentes
def plot_categorical_data(coluna):
    # Remover valores NaN
    coluna = coluna.dropna()

    # Verificar se a coluna não está vazia após a remoção de NaN
    if len(coluna) == 0:
        print("Erro: a coluna está vazia após remover valores NaN.")
        return

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))  # Gráfico de barras único

    # Gerando uma paleta de cores com uma cor para cada categoria
    unique_categories = np.unique(coluna)
    colors = sns.color_palette("Set2", len(unique_categories))

    # Gráfico de Barras para dados categóricos com cores diferenciadas
    sns.countplot(x=coluna, ax=axs, palette=colors, hue=coluna, legend=False)
    axs.set_title("Distribuição de Categóricos")
    axs.set_xlabel("Categoria")
    axs.set_ylabel("Contagem")

    plt.tight_layout()
    plt.show()

    return


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(df, class_column, color_palette="Set2", figsize=(9.1, 6)):
    """
    Função simplificada para plotar a distribuição das classes, corrigindo todos os avisos futuros.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    class_column : str
        Nome da coluna contendo as classes.
    color_palette : str, opcional
        Nome da paleta de cores a ser utilizada pelo gráfico. Padrão é "Set2".
    figsize : tuple, opcional
        Tamanho fixo do gráfico (largura, altura). Padrão é (8, 6).
    """
    
    # Conta as classes
    class_counts = df[class_column].value_counts()
    num_classes = len(class_counts)
    
    # Ajusta a paleta para o número exato de classes
    color_palette = sns.color_palette(color_palette, num_classes)
    
    # Configura o tamanho do gráfico
    plt.figure(figsize=figsize)
    
    # Usando `x` como `hue` para evitar avisos
    sns.countplot(
        x=class_column, 
        data=df, 
        hue=class_column, 
        dodge=False, 
        palette=color_palette, 
        legend=False  # Remove a legenda se não for necessária
    )
    
    # Configurando título e rótulos
    plt.title(f"Distribuição das Classes: {class_column}")
    plt.xlabel("Classe")
    plt.ylabel("Contagem")
    
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_class_distribution_comparison(original_data, resampled_data, column_name, nome_arquivo="model", figsize=(12, 5), palette=["Gold", "#FF7043"]):
    """
    Cria gráficos comparativos da distribuição de classes antes e depois do balanceamento e salva como imagem.

    Parâmetros:
    -----------
    original_data : pandas.DataFrame
        DataFrame contendo os dados originais (com a coluna a ser analisada).
    resampled_data : pandas.DataFrame
        DataFrame contendo os dados balanceados (com a coluna a ser analisada).
    column_name : str
        Nome da coluna que representa a classe a ser analisada.
    nome_arquivo : str, opcional (default="model")
        Nome do modelo usado para nomear o arquivo de imagem.
    figsize : tuple, opcional (default=(12, 5))
        Tamanho da figura para os gráficos.
    palette : list, opcional (default=["Gold", "#FF7043"])
        Paleta de cores para as barras do gráfico.

    Retorna:
    --------
    None
    """
    # Caminho para salvar os gráficos
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Criando DataFrames para gráficos
    original_df = pd.DataFrame({column_name: original_data})
    resampled_df = pd.DataFrame({column_name: resampled_data})

    # Criando os gráficos lado a lado
    plt.figure(figsize=figsize)
    
    # Gráfico antes do balanceamento
    plt.subplot(1, 2, 1)
    sns.countplot(x=column_name, data=original_df, hue=column_name, dodge=False, palette=palette)
    plt.title(f"Antes do Balanceamento")
    plt.xlabel("Classe")
    plt.ylabel("Frequência")
    
    # Gráfico depois do balanceamento
    plt.subplot(1, 2, 2)
    sns.countplot(x=column_name, data=resampled_df, hue=column_name, dodge=False, palette=palette)
    plt.title(f"Depois do Balanceamento")
    plt.xlabel("Classe")
    plt.ylabel("Frequência")

    # Ajustando o layout
    plt.tight_layout()

    # Caminho do arquivo para salvar
    fig_path = os.path.join(figures_path, f"{nome_arquivo}.png")

    # Salvar o gráfico
    plt.savefig(fig_path)
    print(f"Gráfico de distribuição de classes salvo em: {fig_path}")
    plt.show()

    return



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import warnings
from sklearn.exceptions import ConvergenceWarning
import os

def monitorar_custo_gradiente(model_lr, X_train, y_train, max_iter=100, nome_arquivo='logistic_gradient'):
    """
    Função para monitorar a evolução do custo (log-loss) e do gradiente ao longo das iterações do modelo
    e salvar os gráficos gerados.

    Parâmetros:
    -----------
    model_lr : modelo treinado
        O modelo de regressão logística a ser utilizado.
    X_train : DataFrame
        O conjunto de treino.
    y_train : Series
        O vetor de rótulos do conjunto de treino.
    max_iter : int, default 100
        Número máximo de iterações para monitorar o custo e o gradiente.
    nome_arquivo : str, opcional (default='logistic_gradient')
        Nome base para os arquivos PNG a serem salvos.
    """

    # Ignorar warning de bins pequenos e de convergência
    warnings.filterwarnings("ignore", message="Bins whose width are too small")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Variáveis para armazenar custos e gradientes
    iterations = []
    costs = []
    gradients = []

    # Iterar sobre as iterações
    for i in range(1, max_iter + 1):  # 100 iterações
        model_lr.set_params(max_iter=i)  # Ajustar o número de iterações do modelo
        model_lr.fit(X_train, y_train)  # Treinar o modelo
        
        # Previsões de probabilidade
        probabilities = model_lr.predict_proba(X_train)[:, 1]  # Probabilidades para a classe positiva (1)
        
        # Cálculo do custo logístico (log-loss)
        cost = log_loss(y_train, probabilities)  # Calcular log-loss
        costs.append(cost)  # Armazenar custo
        iterations.append(i)  # Armazenar iteração atual
        
        # Gradientes simulados (calculando norma L2 dos coeficientes do modelo)
        gradient_norm = np.linalg.norm(model_lr.coef_)
        gradients.append(gradient_norm)  # Armazenar gradiente

    # Obter o caminho absoluto do diretório do projeto
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))

    # Caminho completo para a pasta reports/figures
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Gráfico de Custo vs Iterações
    plt.figure(figsize=(15, 6))
    plt.plot(iterations, costs, label="Custo", color='blue')
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.title("Evolução da Função de Custo")
    plt.legend()
    plt.grid(True)

    # Caminho para salvar o gráfico de custo
    cost_fig_path = os.path.join(figures_path, f'{nome_arquivo}_custo.png')
    plt.savefig(cost_fig_path)
    print(f"Gráfico de Custo salvo em: {cost_fig_path}")
    plt.show()

    # Gráfico de Gradiente vs Iterações
    plt.figure(figsize=(15, 6))
    plt.plot(iterations, gradients, label="Gradiente", color='orange')
    plt.xlabel("Iterações")
    plt.ylabel("Magnitude do Gradiente")
    plt.title("Evolução do Gradiente")
    plt.legend()
    plt.grid(True)

    # Caminho para salvar o gráfico de gradiente
    gradient_fig_path = os.path.join(figures_path, f'{nome_arquivo}_gradiente.png')
    plt.savefig(gradient_fig_path)
    print(f"Gráfico de Gradiente salvo em: {gradient_fig_path}")
    plt.show()

    return



   

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

def comparar_gradientes(X_train, y_train, model_lr, max_iter=100, nome_arquivo='gradient_comparison'):
    """
    Função para comparar a evolução do gradiente entre o modelo sem regularização e o modelo com Elasticnet,
    e salvar o gráfico gerado.

    Parâmetros:
    -----------
    X_train : DataFrame
        O conjunto de dados de treino.
    y_train : Series
        O vetor de rótulos do conjunto de treino.
    model_lr : modelo treinado
        O modelo de regressão logística com Elasticnet (regularização).
    max_iter : int, default 100
        O número máximo de iterações para monitorar os gradientes.
    nome_arquivo : str, opcional (default='gradient_comparison')
        Nome base para o arquivo PNG a ser salvo.
    """

    # Definir modelo sem regularização
    model_no_regularization = LogisticRegression(penalty=None, solver='saga', max_iter=1000)

    # Variáveis para armazenar gradientes
    gradients_no_regularization = []
    gradients_Elasticnet = []
    iterations = []

    # Monitorar gradientes para 100 iterações
    for i in range(1, max_iter + 1):
        # Sem regularização
        model_no_regularization.set_params(max_iter=i)
        model_no_regularization.fit(X_train, y_train)
        gradients_no_regularization.append(np.linalg.norm(model_no_regularization.coef_))

        # Com Elasticnet (regularização L2)
        model_lr.set_params(max_iter=i)
        model_lr.fit(X_train, y_train)
        gradients_Elasticnet.append(np.linalg.norm(model_lr.coef_))

        iterations.append(i)

    # Obter o caminho absoluto do diretório do projeto
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))

    # Caminho completo para a pasta reports/figures
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Gráfico de Gradiente vs Iterações
    plt.figure(figsize=(15, 6))
    plt.plot(iterations, gradients_no_regularization, label="Gradiente (Sem Regularização)", color='blue')
    plt.plot(iterations, gradients_Elasticnet, label="Gradiente (Elasticnet)", color='red')
    plt.xlabel("Iterações")
    plt.ylabel("Magnitude do Gradiente")
    plt.title("Comparação da Evolução do Gradiente")
    plt.legend()
    plt.grid(True)

    # Caminho para salvar o gráfico
    gradient_fig_path = os.path.join(figures_path, f'{nome_arquivo}.png')
    plt.savefig(gradient_fig_path)
    print(f"Gráfico salvo em: {gradient_fig_path}")
    plt.show()

    return


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def exibir_metricas_e_matriz_confusao(y_test, y_pred, nome_arquivo='confusion_matrix'):
    """
    Função para exibir as métricas do modelo, a matriz de confusão
    e salvar a matriz como um arquivo PNG.

    Parâmetros:
    -----------
    y_test : array-like
        Rótulos reais do conjunto de teste.
    y_pred : array-like
        Rótulos previstos pelo modelo.
    nome_arquivo : str, opcional (default='confusion_matrix')
        Nome do arquivo PNG para salvar a matriz de confusão.
    """
    # Exibir as métricas de classificação
    print("\nMétricas do modelo:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Criar a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No', 'Yes'])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    
    # Caminho absoluto do projeto
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
    
    # Caminho completo para a pasta reports/figures
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)
    
    # Caminho completo para salvar a figura
    caminho_arquivo = os.path.join(figures_path, f'{nome_arquivo}.png')
    
    # Salvar a figura
    plt.savefig(caminho_arquivo)
    print(f"\nMatriz de confusão salva como: {caminho_arquivo}")
    
    # Exibir o gráfico
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

def curva_de_aprendizado(model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), nome_arquivo='learning_curve'):
    """
    Função para gerar e salvar a curva de aprendizado de um modelo.

    Parâmetros:
    -----------
    model : objeto
        O modelo a ser avaliado.
    X_train : DataFrame ou array
        Conjunto de dados de treino.
    y_train : Series ou array
        Rótulos reais do conjunto de treino.
    cv : int, default 5
        Número de divisões para validação cruzada.
    scoring : str, default 'accuracy'
        A métrica a ser usada para avaliar o modelo.
    train_sizes : array-like, default np.linspace(0.1, 1.0, 10)
        Tamanhos do conjunto de treino para os quais calcular a curva de aprendizado.
    nome_arquivo : str, opcional (default='learning_curve')
        Nome do arquivo PNG a ser salvo.

    Retorna:
    --------
    test_std, test_mean, test_scores : tuple
        Desvio padrão, média e todas as pontuações de teste.
    """
    
    # Ignorar warnings específicos
    warnings.filterwarnings("ignore", message="Bins whose width are too small")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    # Gerar as curvas de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring=scoring, train_sizes=train_sizes
    )

    # Calcular médias e desvios padrão
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Obter o caminho absoluto do diretório do projeto
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))

    # Caminho completo para a pasta reports/figures
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Plotar as curvas de aprendizado
    plt.figure(figsize=(15, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.1)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Treino')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Teste')

    plt.xlabel('Tamanho do conjunto de treino', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title('Curva de Aprendizado - ' + str(model.__class__.__name__), fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)

    # Caminho para salvar o gráfico
    learning_curve_fig_path = os.path.join(figures_path, f'{nome_arquivo}.png')
    plt.savefig(learning_curve_fig_path)
    print(f"Gráfico salvo em: {learning_curve_fig_path}")
    plt.show()

    return test_std, test_mean, test_scores

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def plot_roc_curve(model, X_test, y_test, nome_arquivo='roc_curve'):
    """
    Calcula, plota e salva a curva ROC para um modelo de classificação.

    Parâmetros:
        - model: Modelo treinado (ex: LogisticRegression)
        - X_test: Features de teste
        - y_test: Target de teste
        - nome_arquivo: Nome do arquivo de imagem para salvar o gráfico (default='roc_curve')

    Retorna:
        - roc_auc: Valor da AUC (Area Under the Curve)
    """
    # Prever as probabilidades da classe positiva
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcular os valores da curva ROC
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Obter o caminho absoluto do diretório do projeto
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))

    # Caminho completo para a pasta reports/figures
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Plotar a curva ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Aleatório')
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.5)

    # Caminho para salvar o gráfico
    roc_curve_fig_path = os.path.join(figures_path, f'{nome_arquivo}.png')
    plt.savefig(roc_curve_fig_path)
    print(f"Gráfico ROC salvo em: {roc_curve_fig_path}")
    plt.show()

    # Exibir o valor da AUC
    print(f"AUC: {roc_auc:.4f}")

    # Retornar a AUC
    return roc_auc


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import os

def plot_precision_recall_curve(model, X_test, y_test, nome_arquivo='precision_recall_curve'):
    """
    Calcula, plota e salva a curva Precision-Recall para um modelo de classificação.

    Parâmetros:
        - model: Modelo treinado (ex: LogisticRegression).
        - X_test: Features de teste.
        - y_test: Target de teste.
        - nome_arquivo: Nome do arquivo de imagem para salvar o gráfico (default='precision_recall_curve').

    Retorna:
        - auc_pr: Valor da AUC-PR (Área sob a curva Precision-Recall).
    """
    # Prever as probabilidades da classe positiva
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcular os valores da curva Precision-Recall
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recalls, precisions)

    # Obter o caminho absoluto do diretório do projeto
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))

    # Caminho completo para a pasta reports/figures
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Plotar a curva Precision-Recall
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='blue', label=f'Curva Precision-Recall (AUC = {auc_pr:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precisão', fontsize=12)
    plt.title('Curva Precision-Recall', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.5)

    # Caminho para salvar o gráfico
    pr_curve_fig_path = os.path.join(figures_path, f'{nome_arquivo}.png')
    plt.savefig(pr_curve_fig_path)
    print(f"Gráfico Precision-Recall salvo em: {pr_curve_fig_path}")
    plt.show()

    # Exibir o valor da AUC-PR
    print(f"AUC-PR: {auc_pr:.4f}")

    # Retornar a AUC-PR
    return auc_pr


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import os

def plot_probability_distribution(y_prob, nome_arquivo="model", bins=100):
    """
    Plota e salva a distribuição das probabilidades previstas por um modelo de classificação.

    Parâmetros:
        - y_prob: Array com as probabilidades previstas (ex: saída de predict_proba)
        - nome_arquivo: Nome do modelo usado para nomear o arquivo de imagem (default: "model")
        - bins: Número de bins do histograma (padrão: 100)

    Retorna:
        - None (plota o gráfico e salva a imagem)
    """
    # Caminho para salvar os gráficos
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Criando o histograma
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob, bins=bins, alpha=0.7, color='royalblue', edgecolor='black')
    
    # Personalização do gráfico
    plt.title('Distribuição das Probabilidades Preditas', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Probabilidade', fontsize=12, labelpad=10)
    plt.ylabel('Frequência', fontsize=12, labelpad=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    # Caminho do arquivo para salvar
    fig_path = os.path.join(figures_path, f"{nome_arquivo}.png")

    # Salvar o gráfico
    plt.savefig(fig_path)
    print(f"Gráfico de distribuição de probabilidades salvo em: {fig_path}")
    plt.show()

    return




import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def matriz_confusao(y_test, y_pred, nome_arquivo="model"):
    """
    Exibe e salva a matriz de confusão como uma imagem PNG.

    Parâmetros:
    -----------
    y_test : array-like
        Rótulos reais do conjunto de teste.
    y_pred : array-like
        Rótulos previstos pelo modelo.
    nome_arquivo : str, default 'model'
        Nome do modelo usado para nomear o arquivo de imagem.

    Retorna:
    --------
    None
    """
    # Caminho para salvar o gráfico
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
    figures_path = os.path.join(project_path, 'reports/figures')
    os.makedirs(figures_path, exist_ok=True)

    # Exibir o relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Criar a matriz de confusão
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['No', 'Yes'])
    disp.plot(cmap="Blues")
    plt.title(f'Matriz de Confusão - {nome_arquivo}')
    
    # Caminho do arquivo para salvar
    matriz_fig_path = os.path.join(figures_path, f'{nome_arquivo}_confusion_matrix.png')
    
    # Salvar o gráfico
    plt.savefig(matriz_fig_path)
    print(f"Matriz de confusão salva em: {matriz_fig_path}")
    plt.close()

    return