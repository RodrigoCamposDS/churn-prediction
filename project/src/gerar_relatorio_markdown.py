import pandas as pd
from jinja2 import Template
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Gerar a data atual para o cabeçalho
data_atual = datetime.now().strftime("%d/%m/%Y")


markdown_header = """
title: "Relatório Final: Análise de Churn"
author: "Rodrigo Campos"
date: "{data_atual}"
"""

# Substituir o marcador de data no YAML
markdown_header = markdown_header.format(data_atual=data_atual)

# # Dados de exemplo
# modelos = ["Reg. Logística", "Reg. Logística", "Random Forest", "Random Forest"]
# thresholds = [0.5, 0.347, 0.5, 0.367]
# fp_values = [875, 1177, 888, 1771]
# fn_values = [1203, 621, 1903, 1272]

# Carregar os dados reais do arquivo .pkl
df_dados = pd.read_pickle("../data/metrics_churn.pkl")

# Extrair os valores das colunas
modelos = df_dados["Modelo"].tolist()
thresholds = df_dados["Threshold"].tolist()
fp_values = df_dados["FP"].tolist()
fn_values = df_dados["FN"].tolist()

# Cálculo dos custos
custos_fp = [fp * 5 for fp in fp_values]
custos_fn = [fn * 25 for fn in fn_values]
custos_totais = [fp + fn for fp, fn in zip(custos_fp, custos_fn)]

# Criando o DataFrame
df_resultados = pd.DataFrame({
    "Modelo": modelos,
    "Threshold": thresholds,
    "FP": fp_values,
    "FN": fn_values,
    "Custo Total (R$)": custos_totais
})

# Encontrando o modelo vencedor
modelo_vencedor = df_resultados.loc[df_resultados["Custo Total (R$)"].idxmin(), "Modelo"]
menor_custo = df_resultados["Custo Total (R$)"].min()

# Carregando o template
with open("../templates/template_relatorio.md", "r") as file:
    template_content = file.read()

# Processando o template com Jinja2
template = Template(template_content)
markdown_content = template.render(
    resultados=df_resultados.to_dict(orient="records"),
    modelo_vencedor=modelo_vencedor,
    menor_custo=menor_custo
)

# Concatenar o cabeçalho YAML e o conteúdo do relatório
full_markdown_content = markdown_header + markdown_content

# Salvando o relatório em Markdown
with open("../reports/relatorio_churn.md", "w") as file:
    file.write(full_markdown_content)

print("Relatório Markdown gerado com sucesso!")