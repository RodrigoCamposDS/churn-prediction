import pandas as pd
from jinja2 import Template

# Dados de exemplo para cálculo de custos
modelos = ["Reg. Logística", "Reg. Logística", "Random Forest", "Random Forest"]
thresholds = [0.5, 0.347, 0.5, 0.367]
fp_values = [875, 1177, 888, 1771]
fn_values = [1203, 621, 1903, 1272]

# Cálculo dos custos
custos_fp = [fp * 5 for fp in fp_values]
custos_fn = [fn * 25 for fn in fn_values]
custos_totais = [fp + fn for fp, fn in zip(custos_fp, custos_fn)]

# Criando DataFrame de resultados
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

# Carregar o template Markdown
with open("/..templates/template_relatorio.md", "r") as file:
    template_content = file.read()

# Processar o template usando Jinja2
template = Template(template_content)
markdown_content = template.render(resultados=df_resultados.to_dict(orient="records"),
                                   modelo_vencedor=modelo_vencedor,
                                   menor_custo=menor_custo)

# Salvar o relatório Markdown
with open("../reports/relatorio_churn.md", "w") as file:
    file.write(markdown_content)

print("Relatório Markdown gerado com sucesso!")