def clean_data(df):
    """
    Função para limpar o dataset (exemplo de exclusão de valores ausentes e duplicados).
    """
    df = df.dropna()  # Remove valores ausentes
    df = df.drop_duplicates()  # Remove duplicatas
    return df


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
from IPython.display import display, HTML
"""
Função para calcular estatísticas essenciais (medidas):
"""
def estatisticas_resumidas(coluna):
    # Verifica se os dados são numéricos
    if not pd.api.types.is_numeric_dtype(coluna):
        raise ValueError("A coluna deve conter valores numéricos.")

        # Medidas de Tendência Central
    media = round(coluna.mean(), 3)
    mediana = round(coluna.median(), 3)
    moda = round(coluna.mode().values[0], 3)
    Q1 = round(coluna.quantile(0.25), 3)
    Q2 = round(coluna.quantile(0.5), 3)
    Q3 = round(coluna.quantile(0.75), 3)

    # Dispersão
    maximo = round(coluna.max(), 2)
    minimo = round(coluna.min(), 2)
    desvio_padrao = round(coluna.std(), 2)
    variancia = round(coluna.var(), 2)
    amplitude = maximo - minimo
    IQR = round(Q3 - Q1, 2)

    # Gerando HTML
    stats_html = f"""
     <table border="1" style="border-collapse: collapse; width: 50%;">
        <thead>
            <tr>
                <th style="text-align: left;">Tendência Central</th>
                <th style="text-align: left;">Medidas de Dispersão</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="text-align: left;">Média:<span style="padding-left: 50px;">{media:.2f}</span></td>
                <td style="text-align: left;">Mínimo:<span style="padding-left: 80px;">{minimo:.2f}</span></td>
            </tr>
            <tr>
                <td style="text-align: left;">Mediana:<span style="padding-left: 33px;">{mediana:.2f}</span></td>
                <td style="text-align: left;">Máximo:<span style="padding-left: 75px;">{maximo:.2f}</span></td>
            </tr>
            <tr>
                <td style="text-align: left;">Moda:<span style="padding-left: 51px;">{moda:.2f}</span></td>
                <td style="text-align: left;">Desvio Padrão:<span style="padding-left: 30px;">{desvio_padrao:.2f}</span></td>
            </tr>
            <tr>
                <td style="text-align: left;">Q1:<span style="padding-left: 71px;">{Q1:.2f}</span></td>
                <td style="text-align: left;">Variância:<span style="padding-left: 65px;">{variancia:.2f}</span></td>
            </tr>
            <tr>
                <td style="text-align: left;">Q2:<span style="padding-left: 69px;">{Q2:.2f}</span></td>
                <td style="text-align: left;">Amplitude:<span style="padding-left: 58px;">{amplitude:.2f}</span></td>
            </tr>
            <tr>
                <td style="text-align: left;">Q3:<span style="padding-left: 69px;">{Q3:.2f}</span></td>
                <td style="text-align: left;">DQ (IQR):<span style="padding-left: 65px;">{IQR:.2f}</span></td>
            </tr>
        </tbody>
    </table>
    """
    return stats_html

# Função para exibir as estatísticas
def exibir_estatisticas(coluna):
    stats_html = estatisticas_resumidas(coluna)
    display(HTML(stats_html))

 #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    """ 
    Calcula a esparsidade dos dados em cada coluna do DataFrame.
    """
import pandas as pd

# Função para calcular a esparsidade dos dados em cada coluna do DataFrame e exibir em HTML
def calcular_esparsidade(df):
    esparsidade = {}
    
    # Calculando a esparsidade para cada coluna
    for coluna in df.columns:
        # Contar zeros e valores ausentes
        valores_esparsos = (df[coluna] == 0).sum() + df[coluna].isna().sum()
        total_valores = len(df[coluna])
        esparsidade[coluna] = valores_esparsos / total_valores
    
    # Gerando HTML
    stats_html = f"""
    <table>
        <thead>
            <tr>
                <th style="text-align: left; padding-right: 50px;">Coluna</th>
                <th style="text-align: left;">Esparsidade</th>
            </tr>
        </thead>
        <tbody>
    """

    # Preenchendo as linhas da tabela com os valores de esparsidade
    for coluna, espar in esparsidade.items():
        stats_html += f"""
        <tr>
            <td style="text-align: left;">{coluna}</td>
            <td style="text-align: left;">{espar:.2%}</td>
        </tr>
        """
    
    # Fechando a tabela
    stats_html += """
        </tbody>
    </table>
    """

    # Exibir HTML
    from IPython.display import display, HTML
    display(HTML(stats_html))

    return


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from IPython.display import display, HTML

# Função para calcular estatísticas de dados categóricos
def estatisticas_categoricas(coluna):
    # Remover valores NaN e substituir por 'Desconhecido'
    coluna = coluna.fillna('Desconhecido')

    # Garantir que os dados sejam categóricos
    if not pd.api.types.is_categorical_dtype(coluna):
        coluna = coluna.astype('category')

    # Frequências Absolutas
    frequencias = coluna.value_counts()

    # Proporções
    proporcoes = coluna.value_counts(normalize=True)

    # Gerando HTML
    stats_html = f"""
    <table>
            <tr>
                <th style="text-align: center; ">Frequência Absoluta</td>
                <th style="text-align: center; ">Proporções</th>
            </tr>

        <tbody>
            <tr>
                <td style="text-align: left;">{frequencias.to_frame().to_html( header=False)}</td>
                <td style="text-align: left;">{proporcoes.to_frame().to_html( header=False)}</td>
            </tr>
        </tbody>
    </table>
    """
    return stats_html

# Função para exibir as estatísticas de dados categóricos
def exibir_estatisticas_categoricas(coluna):
    stats_html = estatisticas_categoricas(coluna)
    display(HTML(stats_html))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

# Função retorna a quantidade de valores nulos e zeros por coluna
def valores_nulos_zeros(df):
    # Calcula a quantidade de zeros e valores nulos por coluna
    num_zeros = (df == 0).sum()
    num_vazios = df.isnull().sum()

    # Gerando HTML
    stats_html = f"""
    <table>
        <thead>
            <tr>
                <th style="text-align: left; padding-right: 50px;">Coluna</th>
                <th style="text-align: left; padding-right: 50px;">Zerados</th>
                <th style="text-align: left;">Valores Nulos</th>
            </tr>
        </thead>
        <tbody>
    """

    # Preenchendo as linhas da tabela
    for coluna in df.columns:
        stats_html += f"""
        <tr>
            <td style="text-align: left;">{coluna}</td>
            <td style="text-align: left;">{num_zeros[coluna]}</td>
            <td style="text-align: left;">{num_vazios[coluna]}</td>
        </tr>
        """

    # Fechando a tabela
    stats_html += """
        </tbody>
    </table>
    """

    # Exibir HTML
    from IPython.display import display, HTML
    display(HTML(stats_html))

    return

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from IPython.display import display, HTML

def analisar_qualidade_dados(df):
    # Inicializando dicionários
    nulos_zeros = {}
    esparsidade = {}
    
    # Calculando valores nulos e zeros
    for coluna in df.columns:
        num_zeros = (df[coluna] == 0).sum()
        num_vazios = df[coluna].isna().sum()
        nulos_zeros[coluna] = {"Zeros": num_zeros, "Valores Vazios": num_vazios}
        
        # Calculando a esparsidade
        valores_esparsos = num_zeros + num_vazios
        total_valores = len(df[coluna])
        esparsidade[coluna] = valores_esparsos / total_valores

    # Gerando o HTML
    stats_html = f"""
     <table border="1" style="border-collapse: collapse; width: 50%;">
        <thead>
            <tr>
                <th style="text-align: left; padding-right: 50px;">Coluna</th>
                <th style="text-align: left; padding-right: 50px;">Zeros</th>
                <th style="text-align: left; padding-right: 50px;">Valores Vazios</th>
                <th style="text-align: left;">Esparsidade</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Preenchendo as linhas da tabela com as informações de zeros, valores vazios e esparsidade
    for coluna in df.columns:
        stats_html += f"""
        <tr>
            <td style="text-align: left;">{coluna}</td>
            <td style="text-align: left;">{nulos_zeros[coluna]["Zeros"]}</td>
            <td style="text-align: left;">{nulos_zeros[coluna]["Valores Vazios"]}</td>
            <td style="text-align: left;">{esparsidade[coluna]:.2%}</td>
        </tr>
        """
    
    # Fechando a tabela
    stats_html += """
        </tbody>
    </table>
    """
    
    # Exibir HTML
    display(HTML(stats_html))

    return

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import warnings


def calcular_faixa_por_variavel(df, column_name, target_column, num_bins, target_value=None, order='none'):
    """
    Função para criar faixas (bins) de uma variável contínua e calcular o total do alvo (target_column) por faixa,
    permitindo selecionar o valor específico do alvo (ex: 0, 1, 'Yes', 'No') e a ordenação (crescente, decrescente ou nenhuma).

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    column_name : str
        Nome da coluna que representa a variável contínua a ser dividida em faixas.
    target_column : str
        Nome da coluna alvo para calcular o total (pode ser 0, 1, 'Yes', 'No', etc.).
    num_bins : int
        Número de bins que o usuário deseja para dividir os dados.
    target_value : int, str, optional
        Valor do alvo para o qual será feita a soma (ex: 1, 0, 'Yes', 'No'). Se não for especificado, 
        será considerada a soma total do alvo (geral).
    order : str, optional
        Ordenação dos resultados. Pode ser 'asc' para crescente, 'desc' para decrescente ou 'none' para nenhuma ordenação.

    Retorna:
    --------
    pandas.DataFrame
        DataFrame com as faixas e o total do alvo em cada faixa.
    """
    
    # Determinando os limites dos bins automaticamente com base nos valores da coluna
    bins = pd.cut(df[column_name], bins=num_bins, retbins=True, right=False)[1]
    
    # Gerando labels para as faixas
    labels = [f'Faixa {i+1}: {int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins)-1)]
    
    # Adicionando a nova coluna de faixas
    df.loc[:, 'Variable_bins'] = pd.cut(df[column_name], bins=bins, labels=labels, right=False)
    

    # Ignorar o aviso específico do Pandas relacionado ao uso de tipos incompatíveis
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Substituindo NaN por uma categoria (como 'Falta de Dados' ou outro nome apropriado)
    df['Variable_bins'] = df['Variable_bins'].cat.add_categories('Falta de Dados')
    df['Variable_bins'] = df['Variable_bins'].fillna('Falta de Dados')

    # Garantir que a coluna 'Variable_bins' seja do tipo 'category'
    df['Variable_bins'] = df['Variable_bins'].astype('category')
    
    # Filtrando a coluna do alvo se o valor for especificado
    if target_value is not None:
        # Filtra os valores da coluna target de acordo com o valor especificado
        df_filtered = df[df[target_column] == target_value]
    else:
        # Se não for especificado o valor do alvo, considera todos os valores
        df_filtered = df
    
    # Calculando a soma do alvo por faixa
    qtd_target = (df_filtered.groupby('Variable_bins', observed=False)[target_column]
                  .count()  # Usamos count() para contar as instâncias do alvo em cada faixa
                  .reset_index(name='Total_Target'))
    
    # Ordenação, se solicitado
    if order == 'asc':
        qtd_target = qtd_target.sort_values(by='Total_Target', ascending=True)
    elif order == 'desc':
        qtd_target = qtd_target.sort_values(by='Total_Target', ascending=False)
    
    return qtd_target


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
from IPython.display import display, HTML

def generate_class_table(df, class_column):
    """
    Função para gerar uma tabela HTML com a contagem e proporção por classe.
    
    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    class_column : str
        Nome da coluna contendo as classes para análise.
    
    Retorna:
    --------
    str
        Tabela HTML com a contagem e proporção por classe.
    """
    # Contar o número de observações por classe
    class_counts = df[class_column].value_counts()

    # Calcular a proporção de cada classe
    class_proportions = df[class_column].value_counts(normalize=True) * 100

    # Criar um DataFrame com as informações
    class_summary = pd.DataFrame({
        'Contagem': class_counts,
        'Proporção (%)': class_proportions
    })

    # Gerar a tabela HTML manualmente com a concatenação correta
    stats_html = """
    <table border="1" style="border-collapse: collapse; width: 50%;">
        <thead>
            <tr>
                <th style="padding: 10px; text-align: center;">Classe</th>
                <th style="padding: 10px; text-align: center;">Contagem</th>
                <th style="padding: 10px; text-align: center;">Proporção (%)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for index, row in class_summary.iterrows():
        stats_html += f"""
        <tr>
            <td style="padding: 8px; text-align: center;">{index}</td>
            <td style="padding: 8px; text-align: center;">{row['Contagem']}</td>
            <td style="padding: 8px; text-align: center;">{row['Proporção (%)']:.2f}</td>
        </tr>
        """
    
    stats_html += """
        </tbody>
    </table>
    """
    
    # Exibir a tabela HTML
    display(HTML(stats_html))

    return




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from IPython.display import display, HTML

def comparar_variavel_com_target(df, var_comparada, target_column):
    """
    Função para comparar uma variável com o target e retornar uma tabela HTML com a contagem e a proporção.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os dados.
    var_comparada : str
        Nome da variável a ser comparada.
    target_column : str
        Nome da coluna do target para análise.

    Retorna:
    --------
    str
        Tabela HTML com a comparação.
    """
    # Agrupar os dados pela variável e calcular a soma para o target
    comparacao = df.groupby(var_comparada)[target_column].sum().reset_index(name='Total_Churned')

    # Calcular a proporção por grupo
    comparacao['Proporção (%)'] = (comparacao['Total_Churned'] / comparacao['Total_Churned'].sum()) * 100

    # Gerar a tabela HTML
    stats_html = """
    <table border="1" style="border-collapse: collapse; width: 51%;">
        <thead>
            <tr>
                <th style="text-align: left; padding-right: 50px;">Categoria</th>
                <th style="text-align: left; padding-right: 50px;">Total de Churned</th>
                <th style="text-align: left; padding-right: 50px;">Proporção (%)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for index, row in comparacao.iterrows():
        stats_html += f"""
        <tr>
            <td style="text-align: left; padding-left: 10px;">{row[var_comparada]}</td>
            <td style="text-align: left; padding-left: 10px;">{row['Total_Churned']}</td>
            <td style="text-align: left; padding-left: 10px;">{row['Proporção (%)']:.2f}</td>
        </tr>
        """
    
    stats_html += """
        </tbody>
    </table>
    """
    
    # Exibir a tabela HTML
    display(HTML(stats_html))

    return comparacao