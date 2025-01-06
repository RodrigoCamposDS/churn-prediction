import pandas as pd
import pytest
from data.load_data import load_data

import sys
import os

# Adiciona o caminho absoluto para src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Agora você pode importar os módulos
from data.load_data import load_data

def test_no_missing_values():
    """
    Verifica se não há valores ausentes após o pré-processamento.
    """
    df = load_data('src/data/processed_data/processed_data.csv')
    assert df.isnull().sum().sum() == 0, "Ainda há valores ausentes no DataFrame!"

def test_expected_columns():
    """
    Verifica se as colunas esperadas estão presentes após o pré-processamento.
    """
    expected_columns = ['Age', 'Time_on_platform', 'Devices_connected', 'Num_active_profiles', 'Avg_rating', 'Churn']
    df = load_data('src/data/processed_data/processed_data.csv')
    assert all(col in df.columns for col in expected_columns), "Colunas esperadas estão faltando no DataFrame!"

def test_data_types():
    """
    Verifica se os tipos de dados estão corretos.
    """
    df = load_data('src/data/processed_data/processed_data.csv')
    assert pd.api.types.is_numeric_dtype(df['Age']), "A coluna 'Age' deve ser numérica!"
    assert pd.api.types.is_numeric_dtype(df['Avg_rating']), "A coluna 'Avg_rating' deve ser numérica!"