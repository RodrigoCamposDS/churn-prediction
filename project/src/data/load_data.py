import pandas as pd
import os

def load_data(filename):
    """
    Função para carregar o dataset da pasta raw.
    """
    # Caminho absoluto do diretório raiz do projeto
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Caminho completo até a pasta raw
    raw_path = os.path.join(base_dir, 'src', 'data', 'raw')
    
    # Concatena o caminho com o nome do arquivo
    file_path = os.path.join(raw_path, filename)
    
    # Carrega o arquivo CSV
    return pd.read_csv(file_path)