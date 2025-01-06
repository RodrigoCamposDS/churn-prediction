def remove_duplicates(df):
    """
    Remove linhas duplicadas de um DataFrame.
    
    Args:
    df (pd.DataFrame): DataFrame com linhas duplicadas.
    
    Returns:
    pd.DataFrame: DataFrame sem linhas duplicadas.
    """
    # Excluir linhas totalmente duplicadas
    duplicatas = df.duplicated()
    qtd_duplicatas = duplicatas.sum()
    print(f"\nQuantidade de linhas duplicadas encontradas: {qtd_duplicatas}")

    # Excluir linhas totalmente duplicadas
    df_sem_duplicatas = df.drop_duplicates()

    # Mostrar a quantidade de linhas excluídas
    linhas_excluidas = len(df) - len(df_sem_duplicatas)
    print(f"Quantidade de linhas duplicadas removidas: {linhas_excluidas}")
    
    return 


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def imputar_dados_continuos(df, coluna_continua, num_bins=10):
    """
    Função para interpolar valores ausentes em uma coluna contínua usando RandomForestClassifier
    e amostragem dos valores dentro de bins.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados com valores ausentes.
    coluna_continua : str
        Nome da coluna contínua a ser tratada.
    num_bins : int, opcional (default=10)
        Número de bins a ser utilizado para dividir os dados contínuos.

    Retorna:
    --------
    df : pandas.DataFrame
        DataFrame com os valores ausentes imputados na coluna contínua.
    """
    # Criar bins para a variável contínua
    bins = pd.qcut(df[coluna_continua].dropna(), q=num_bins, labels=False, duplicates='drop')
    df.loc[~df[coluna_continua].isnull(), f'{coluna_continua}_bin'] = bins

    # Preparar dados para o RandomForestClassifier
    features = df.dropna(subset=[f'{coluna_continua}_bin']).drop([coluna_continua, f'{coluna_continua}_bin'], axis=1)
    target = df[f'{coluna_continua}_bin'].dropna()

        # Contar valores ausentes antes da imputação
    antes_imputacao = df[coluna_continua].isnull().sum()
    print(f"Valores ausentes antes da imputação: {antes_imputacao}")

    # Treinar o RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)

    # Prever os bins para valores faltantes
    missing_features = df.loc[df[coluna_continua].isnull()].drop([coluna_continua, f'{coluna_continua}_bin'], axis=1)
    predicted_bins = model.predict(missing_features)

    # Amostrar valores dentro de cada bin previsto
    for i, bin_value in enumerate(predicted_bins):
        possible_values = df[df[f'{coluna_continua}_bin'] == bin_value][coluna_continua]
        df.loc[missing_features.index[i], coluna_continua] = np.random.choice(possible_values)

    # Remover a coluna auxiliar
    df.drop(columns=[f'{coluna_continua}_bin'], inplace=True)

       
    # Calcular quantos valores NaN foram imputados no total
    nans_depois = df[coluna_continua].isnull().sum()
    print(f"Valores ausentes após a imputação: {nans_depois}")
    
    # Quantidade de valores imputados (sem considerar NaN)
    imputados = abs(antes_imputacao - nans_depois)
    print(f"Total de valores imputados: {imputados}")

    return 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def imputar_target(df, target_column):
    """
    Função para imputar os valores ausentes do target usando RandomForestClassifier.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados com valores ausentes.
    target_column : str
        Nome da coluna alvo que contém os valores ausentes a serem imputados.
    
    Retorna:
    --------
    df : pandas.DataFrame
        DataFrame com os valores ausentes do target substituídos pelos valores previstos.
    tabela_html : str
        Tabela HTML comparando a distribuição antes e depois da imputação.
    """
    # Identificar as colunas preditoras automaticamente (excluindo a coluna alvo)
    features_columns = [col for col in df.columns if col != target_column]
    
    # Contar valores antes da imputação
    valores_antes = df[target_column].value_counts(dropna=False)

    # Remover linhas onde o 'target_column' é nulo (para treinar o modelo)
    features = df.dropna(subset=[target_column])[features_columns]
    target = df[target_column].dropna()

    # Treinar o RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)

    # Prever os valores faltantes do target
    missing_features = df[df[target_column].isnull()].drop(target_column, axis=1)
    predicted_target = model.predict(missing_features)

    # Substituir os valores ausentes no DataFrame original com as previsões
    df.loc[df[target_column].isnull(), target_column] = predicted_target

    # Contar valores após a imputação
    valores_depois = df[target_column].value_counts(dropna=False)

    # Criar uma tabela comparativa em HTML
    tabela_html = """
    <table border="1" style="border-collapse: collapse; width: 40%;">
        <thead>
            <tr>
                <th style="text-align: left;">Categoria</th>
                <th style="text-align: left;">Distribuição Antes da Imputação</th>
                <th style="text-align: left;">Distribuição Após a Imputação</th>
            </tr>
        </thead>
        <tbody>
    """
    for categoria in valores_antes.index:
        tabela_html += f"""
            <tr>
                <td style="text-align: center;">{categoria}</td>
                <td style="text-align: center;">{valores_antes[categoria]}</td>
                <td style="text-align: center;">{valores_depois.get(categoria, 0)}</td>
            </tr>
        """
    tabela_html += """
        </tbody>
    </table>
    """

    # Exibir o gráfico
    plt.figure(figsize=(10, 6))
    valores_depois.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribuição das Categorias Após a Imputação: {target_column}')
    plt.ylabel('Contagem')
    plt.xlabel('Categoria')
    plt.show()

    # Exibir a tabela em HTML
    display(HTML(tabela_html))

    return 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def imputar_dados_categoricos(df, coluna_categorica, num_bins=10):
    """
    Função para interpolar valores ausentes em uma coluna categórica usando RandomForestClassifier
    e amostragem dos valores dentro de bins.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados com valores ausentes.
    coluna_categorica : str
        Nome da coluna categórica a ser tratada.
    num_bins : int, opcional (default=10)
        Número de bins a ser utilizado para dividir os dados contínuos.

    Retorna:
    --------
    df : pandas.DataFrame
        DataFrame com os valores ausentes imputados na coluna categórica.
    """
    # Transformar a variável categórica em valores numéricos com LabelEncoder
    label_encoder = LabelEncoder()
    df[coluna_categorica] = label_encoder.fit_transform(df[coluna_categorica].astype(str))
    df[coluna_categorica] = df[coluna_categorica].replace(0, np.nan)  # Substitui valores 0 por NaN

    # Criar bins para a variável categórica (usando valores numéricos)
    bins = pd.qcut(df[coluna_categorica].dropna(), q=num_bins, labels=False, duplicates='drop')
    df.loc[~df[coluna_categorica].isnull(), f'{coluna_categorica}_bin'] = bins

    # Preparar dados para o RandomForestClassifier
    features = df.dropna(subset=[f'{coluna_categorica}_bin']).drop([coluna_categorica, f'{coluna_categorica}_bin'], axis=1)
    
    # Aqui converta as variáveis de features para valores numéricos (se necessário)
    for col in features.select_dtypes(include=['object']).columns:
        features[col] = label_encoder.fit_transform(features[col].astype(str))

    target = df[f'{coluna_categorica}_bin'].dropna()

    # Contar valores ausentes antes da imputação
    antes_imputacao = df[coluna_categorica].isnull().sum()
    print(f"Valores ausentes antes da imputação: {antes_imputacao}")

    # Treinar o RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)

    # Prever os bins para valores faltantes
    missing_features = df.loc[df[coluna_categorica].isnull()].drop([coluna_categorica, f'{coluna_categorica}_bin'], axis=1)
    
    # Aqui também transformamos os valores de missing_features
    for col in missing_features.select_dtypes(include=['object']).columns:
        missing_features[col] = label_encoder.transform(missing_features[col].astype(str))

    predicted_bins = model.predict(missing_features)

    # Amostrar valores dentro de cada bin previsto para preencher os valores ausentes
    for i, bin_value in enumerate(predicted_bins):
        possible_values = df[df[f'{coluna_categorica}_bin'] == bin_value][coluna_categorica]
        df.loc[missing_features.index[i], coluna_categorica] = np.random.choice(possible_values)

    # Remover a coluna auxiliar de bins
    df.drop(columns=[f'{coluna_categorica}_bin'], inplace=True)

    # Contar valores por categoria após a imputação
    valores_depois = df[coluna_categorica].value_counts(dropna=False)
    print(f"Valores por categoria após a imputação:\n{valores_depois}\n")
    
    # Visualizar a distribuição após a imputação
    plt.figure(figsize=(10, 6))
    valores_depois.plot(kind='bar')
    plt.title(f'Distribuição das Categorias Após a Imputação: {coluna_categorica}')
    plt.ylabel('Contagem')
    plt.xlabel('Categoria')
    plt.show()
    
    # Calcular quantos valores NaN foram imputados no total
    nans_depois = df[coluna_categorica].isnull().sum()
    print(f"Valores ausentes após a imputação: {nans_depois}")
    
    # Quantidade de valores imputados (sem considerar NaN)
    imputados = abs(antes_imputacao - nans_depois)
    print(f"Total de valores imputados: {imputados}")

    return 



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from IPython.display import display, HTML
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def imputar_dados_discretos(df, coluna_discreta):
    """
    Função para imputar valores ausentes em variáveis discretas usando RandomForestClassifier.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    coluna_discreta : str
        Nome da coluna discreta a ser tratada.
    
    Retorna:
    --------
    df : pandas.DataFrame
        DataFrame com os valores ausentes imputados.
    tabela_html : str
        Tabela HTML comparando a distribuição antes e depois da imputação.
    """
    # Identificar as colunas preditoras automaticamente (excluindo a coluna alvo)
    columns = [col for col in df.columns if col != coluna_discreta]

    # Dividindo as variáveis independentes (X) e dependente (y)
    X = df[columns].copy()
    y = df[coluna_discreta]

    # Removendo linhas com valores ausentes na variável alvo (coluna_discreta) para treinar o modelo
    X = X.loc[~y.isna()]
    y = y.dropna()

    # Contar valores antes da imputação
    valores_antes = df[coluna_discreta].value_counts(dropna=False)

    # Criando e treinando o modelo RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Preencher valores ausentes na variável alvo
    X_missing = df.loc[df[coluna_discreta].isna(), columns]
    if not X_missing.empty:
        y_pred = model.predict(X_missing)
        df.loc[df[coluna_discreta].isna(), coluna_discreta] = y_pred

    # Contar valores após a imputação
    valores_depois = df[coluna_discreta].value_counts(dropna=False)

    # Criar uma tabela comparativa em HTML
    tabela_html = """
    <table border="1" style="border-collapse: collapse; width: 40%;">
        <thead>
            <tr>
                <th style="text-align: left;">Categoria</th>
                <th style="text-align: left;">Distribuição Antes da Imputação</th>
                <th style="text-align: left;">Distribuição Após a Imputação</th>
            </tr>
        </thead>
        <tbody>
    """
    for categoria in valores_antes.index:
        tabela_html += f"""
            <tr>
                <td style="text-align: center;">{categoria}</td>
                <td style="text-align: center;">{valores_antes[categoria]}</td>
                <td style="text-align: center;">{valores_depois.get(categoria, 0)}</td>
            </tr>
        """
    tabela_html += """
        </tbody>
    </table>
    """

    # Exibir o gráfico
    plt.figure(figsize=(10, 6))
    valores_depois.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribuição das Categorias Após a Imputação: {coluna_discreta}')
    plt.ylabel('Contagem')
    plt.xlabel('Categoria')
    plt.show()

    # Exibir a tabela em HTML
    display(HTML(tabela_html))

    return 

# df = pd.read_csv('/Users/rodrigocampos/Documents/Pandas/Desafio06/streaming_data.csv')

# imputar_dados_discretos(df, 'Avg_rating')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def aplicar_pca(X_train, n_components=None, plot_variance=True):
    """
    Função para aplicar o PCA nas features de um DataFrame.

    Parâmetros:
    -----------
    X_train : pandas.DataFrame
        DataFrame contendo as features de treino.
    n_components : int, opcional (default=None)
        Número de componentes principais a serem mantidos. Se None, mantém todas as componentes.
    plot_variance : bool, opcional (default=True)
        Se True, exibe o gráfico da variância explicada acumulada.

    Retorna:
    --------
    X_pca_final : numpy.ndarray
        As features transformadas após a aplicação do PCA.
    pca : PCA
        O objeto PCA treinado.
    components_df : pandas.DataFrame
        DataFrame contendo as contribuições das features originais para os componentes principais.
    """
    
    # 1. Aplicar PCA nas features
    pca = PCA(n_components=n_components)  # Inicializa o PCA com o número desejado de componentes
    X_pca = pca.fit_transform(X_train)  # Aplica o PCA e obtém as features transformadas
    
    # 2. Visualizar a variância explicada acumulada, se solicitado
    if plot_variance:
        explained_variance = pca.explained_variance_ratio_
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Explicada Acumulada')
        plt.title('Variância Explicada por Componentes Principais')
        plt.grid()
        plt.show()
    
    # 3. Reduzir dimensionalidade com o número ideal de componentes, se n_components for fornecido
    if n_components is None:
        n_components = len(X_train.columns)  # Manter todas as componentes principais
    
    X_pca_final = pca.transform(X_train)  # Aplica a transformação para reduzir dimensionalidade
    
    # 4. Analisar as contribuições das features originais
    components = pca.components_  # Obter as componentes principais
    feature_names = X_train.columns  # Nomes das features originais
    
    # Criar um DataFrame para visualizar a contribuição das features
    components_df = pd.DataFrame(components, columns=feature_names, index=[f"PC{i+1}" for i in range(n_components)])
    
    print(f"Formato original das features: {X_train.shape}")
    print(f"Formato após PCA: {X_pca_final.shape}")
    print("\nContribuição das Features para os Componentes Principais:")
    print(components_df.T)
    
    return 


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def calcular_vif(X):
    """
    Função para calcular o VIF (Variance Inflation Factor) para detectar multicolinearidade.
    
    Parâmetros:
    -----------
    X : pandas.DataFrame
        DataFrame contendo as variáveis independentes (features) de treino.
    
    Retorna:
    --------
    vif_data : pandas.DataFrame
        DataFrame com as colunas e seus respectivos valores de VIF.
    """
    # Calcular VIF para cada variável em X
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from imblearn.under_sampling import NearMiss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import pandas as pd
from collections import Counter

def aplicar_lda_com_nearmiss(df, target_column, exclude_columns, n_neighbors=15, test_size=0.2, random_state=42):
    """
    Função para aplicar LDA (Linear Discriminant Analysis) após balanceamento de dados usando NearMiss.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    target_column : str
        Nome da coluna alvo (target).
    exclude_columns : list
        Lista de colunas a serem excluídas ao criar as variáveis preditoras.
    n_neighbors : int, opcional (default=15)
        Número de vizinhos para o balanceamento com NearMiss.
    test_size : float, opcional (default=0.2)
        Proporção dos dados a serem usados no conjunto de teste.
    random_state : int, opcional (default=42)
        Semente para garantir reprodutibilidade dos resultados.
    
    Retorna:
    --------
    X_lda : numpy.ndarray
        Dados após a redução de dimensionalidade com LDA.
    y_balanced : pandas.Series
        A variável alvo balanceada.
    """
    
    # Separar as variáveis independentes (X) e dependente (y)
    X = df.drop(columns=exclude_columns)
    y = df[target_column]
    
    # Certificar-se de mapear para numérico
    y = y.map({"Yes": 1, "No": 0}).astype(int)

    # Verificar e garantir que todas as variáveis em X são numéricas
    X = X.apply(pd.to_numeric, errors='coerce')

    # Tratar valores ausentes ou nulos em X
    X.fillna(0, inplace=True)

    # Balanceamento com NearMiss
    nearmiss = NearMiss(n_neighbors=n_neighbors)
    X_balanced, y_balanced = nearmiss.fit_resample(X, y)
    print("Distribuição das classes após NearMiss:", sorted(Counter(y_balanced).items()))

    # Aplicar LDA
    lda = LinearDiscriminantAnalysis(n_components=1)  # Reduzir para 1 componente
    X_lda = lda.fit_transform(X_balanced, y_balanced)

    # Dividir os dados balanceados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_lda, y_balanced, test_size=test_size, random_state=random_state)

    # Aplicar Regressão Logística para previsão
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Previsões
    y_pred = log_reg.predict(X_test)

    # Pesos das variáveis no LDA
    coeficientes_lda = lda.scalings_.flatten()  # Transformar os coeficientes em um array
    variaveis = X.columns  # Nomes das variáveis originais

    # Criar um DataFrame para visualizar os coeficientes associados às variáveis
    pesos_lda = pd.DataFrame({'Variável': variaveis, 'Peso': coeficientes_lda})
    pesos_lda['Peso_Abs'] = pesos_lda['Peso'].abs()  # Criar uma coluna com valores absolutos
    pesos_lda = pesos_lda.sort_values(by='Peso_Abs', ascending=False)  # Ordenar pela importância

    # Exibir as variáveis mais importantes
    print("Variáveis mais influentes no LDA:")
    print(pesos_lda[['Variável', 'Peso']])

    # Avaliação do Modelo
    print(f'\nAcurácia:, {accuracy_score(y_test, y_pred)}')
    print("Recall:", recall_score(y_test, y_pred, pos_label=1))

    return 