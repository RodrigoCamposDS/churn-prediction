import os
import pytest
from PIL import Image

def test_graph_saving():
    """
    Verifica se os gráficos estão sendo salvos no diretório correto.
    """
    figures_path = 'reports/figures/'
    expected_graphs = [
        'Distribuição de Probabilidade modelo LogisticRegression.png',
        'Distribuição de Probabilidade modelo RandomForest.png',
        'Curva Precision-Recall modelo RandomForest.png'
    ]
    for graph in expected_graphs:
        assert os.path.exists(os.path.join(figures_path, graph)), f"O gráfico {graph} não foi encontrado!"

def test_graph_dimensions():
    """
    Verifica se os gráficos gerados têm as dimensões esperadas.
    """
    graph_path = 'reports/figures/Distribuição de Probabilidade modelo LogisticRegression.png'
    image = Image.open(graph_path)
    width, height = image.size
    assert width == 1000 and height == 600, "As dimensões do gráfico não estão corretas!"