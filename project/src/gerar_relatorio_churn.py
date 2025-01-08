import pandas as pd
from jinja2 import Template
from datetime import datetime

# Carregar os dados do arquivo .pkl
df_dados = pd.read_pickle("../data/metrics_churn.pkl")

# Data atual para o documento
data_atual = datetime.now().strftime("%d/%m/%Y")

# Extração de valores dos dados carregados
modelos = df_dados["Modelo"].tolist()
thresholds = df_dados["Threshold"].tolist()
fp_values = df_dados["FP"].tolist()
fn_values = df_dados["FN"].tolist()

# Calcular os custos
custos_fp = [fp * 5 for fp in fp_values]
custos_fn = [fn * 25 for fn in fn_values]
custos_totais = [fp + fn for fp, fn in zip(custos_fp, custos_fn)]

# Encontrar o modelo vencedor
modelo_vencedor = df_dados.loc[df_dados["Custo Total (R$)"].idxmin(), "Modelo"]
menor_custo = df_dados["Custo Total (R$)"].min()

# Preparar os dados para o LaTeX
dados_tabela = [
    {
        "Modelo": modelo,
        "Threshold": threshold,
        "FP": fp,
        "FN": fn,
        "Custo_FP": custo_fp,
        "Custo_FN": custo_fn,
        "Custo_Total": custo_total,
    }
    for modelo, threshold, fp, fn, custo_fp, custo_fn, custo_total in zip(
        modelos, thresholds, fp_values, fn_values, custos_fp, custos_fn, custos_totais
    )
]

# Template LaTeX com placeholders do Jinja2
template_latex = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{graphicx}
\usepackage{hyperref}
\title{Relatório Final: Análise de Churn}
\author{Rodrigo Campos}
\date{{{{ data_atual }}}}

\begin{document}

\maketitle
\tableofcontents

\section{Objetivo do Projeto}
O objetivo deste relatório é apresentar os resultados da análise de churn.

\section{Resultados}
\subsection{Tabela de Resultados}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
Modelo & Threshold & FP & FN & Custo FP & Custo FN & Custo Total \\ \hline
{% for item in dados_tabela %}
{{ item.Modelo }} & {{ item.Threshold }} & {{ item.FP }} & {{ item.FN }} & R${{ item.Custo_FP }} & R${{ item.Custo_FN }} & R${{ item.Custo_Total }} \\ \hline
{% endfor %}
\end{tabular}

\subsection{Modelo Vencedor}
O modelo com menor custo total foi o \textbf{{{{ modelo_vencedor }}}}, com um custo total de \textbf{R${{{ menor_custo }}}}.

\end{document}
"""

# Renderizar o template com os valores
template = Template(template_latex)
conteudo_latex = template.render(
    data_atual=data_atual, dados_tabela=dados_tabela, modelo_vencedor=modelo_vencedor, menor_custo=menor_custo
)

# Salvar o conteúdo gerado em um arquivo LaTeX
with open("../reports/relatorio_churn.tex", "w") as file:
    file.write(conteudo_latex)

print("Arquivo LaTeX gerado com sucesso!")