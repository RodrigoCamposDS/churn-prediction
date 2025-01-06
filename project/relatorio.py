
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

# Função para calcular custos e gerar relatório
def gerar_relatorio_custos(modelos, thresholds, fp_values, fn_values, saida_pdf="relatorio_custos.pdf"):
    """
    Gera um relatório PDF comparando os custos de diferentes modelos e thresholds.
    
    Parâmetros:
        - modelos: Lista com os nomes dos modelos.
        - thresholds: Lista de thresholds usados.
        - fp_values: Lista com os valores de Falsos Positivos para cada modelo/threshold.
        - fn_values: Lista com os valores de Falsos Negativos para cada modelo/threshold.
        - saida_pdf: Nome do arquivo PDF de saída.
    """

    # Cálculo dos custos
    custos_fp = [fp * 5 for fp in fp_values]
    custos_fn = [fn * 25 for fn in fn_values]
    custos_totais = [fp + fn for fp, fn in zip(custos_fp, custos_fn)]

    # Criando o DataFrame com os resultados
    df_resultados = pd.DataFrame({
        "Modelo": modelos,
        "Threshold": thresholds,
        "Falsos Positivos (FP)": fp_values,
        "Custo FP (R$)": custos_fp,
        "Falsos Negativos (FN)": fn_values,
        "Custo FN (R$)": custos_fn,
        "Custo Total (R$)": custos_totais
    })

    # Ordenar pelo menor custo total
    df_resultados = df_resultados.sort_values(by="Custo Total (R$)")

    # Gera gráfico de comparação de custos
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_resultados, x="Modelo", y="Custo Total (R$)", hue="Threshold")
    plt.title("Comparação de Custos por Modelo e Threshold")
    plt.tight_layout()
    plt.savefig("grafico_custos.png")
    plt.close()

    # Criando o PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Adiciona Título
    pdf.cell(200, 10, txt="Relatório Comparativo de Custos - Modelos e Thresholds", ln=True, align='C')

    # Adiciona Tabela
    pdf.ln(10)
    pdf.cell(200, 10, txt="Tabela de Resultados:", ln=True)
    pdf.ln(5)
    for i in range(len(df_resultados)):
        linha = df_resultados.iloc[i]
        pdf.cell(200, 10, txt=f"Modelo: {linha['Modelo']} | Threshold: {linha['Threshold']} | FP: {linha['Falsos Positivos (FP)']} | FN: {linha['Falsos Negativos (FN)']} | Custo Total: R$ {linha['Custo Total (R$)']:.2f}", ln=True)

    # Adiciona Gráfico
    pdf.image("grafico_custos.png", x=10, y=80, w=180)

    # Salva o PDF
    pdf.output(saida_pdf)

    print(f"Relatório gerado com sucesso: {saida_pdf}")


# Exemplo de uso
modelos = ["Reg. Logística", "Reg. Logística", "Random Forest", "Random Forest"]
thresholds = [0.5, 0.347, 0.5, 0.367]
fp_values = [875, 1177, 888, 1771]
fn_values = [1203, 621, 1903, 1272]

gerar_relatorio_custos(modelos, thresholds, fp_values, fn_values)


#----------------------------------------------------------------------------------------------------



from jinja2 import Environment, FileSystemLoader

def gerar_relatorio(resultados, nome_saida="relatorio_custos.html"):
    """
    Gera um relatório HTML baseado em um template.

    Parâmetros:
        - resultados: Lista de dicionários com os resultados.
        - nome_saida: Nome do arquivo HTML gerado.
    """

    # Configurando o Jinja2 para carregar o template
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)
    template = env.get_template('template.html')

    # Renderizando o template com os dados
    output = template.render(resultados=resultados)

    # Salvando o relatório como HTML
    with open(nome_saida, 'w') as f:
        f.write(output)

    print(f"Relatório gerado com sucesso: {nome_saida}")
