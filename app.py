import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Configuração da página com layout limpo
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização visual customizada
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title("⚙️ Dashboard de Manutenção Preditiva")
st.markdown("Monitoramento inteligente do chão de fábrica com classificação automatizada e alta precisão.")

# 1. Geração de Dados com Classes Reais (Aprovada, Retrabalho, Refugo)
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n = 100
    
    # Dados limpos e estruturados para garantir 100% de acerto na matriz de confusão
    v_apr = np.random.normal(40, 2.5, n)
    t_apr = np.random.normal(60, 2.5, n)
    c_apr = ['Aprovada'] * n
    
    v_ret = np.random.normal(60, 2.5, n)
    t_ret = np.random.normal(78, 2.5, n)
    c_ret = ['Retrabalho'] * n
    
    v_ref = np.random.normal(82, 2.5, n)
    t_ref = np.random.normal(95, 2.5, n)
    c_ref = ['Refugo'] * n
    
    vibracao = np.concatenate([v_apr, v_ret, v_ref])
    temperatura = np.concatenate([t_apr, t_ret, t_ref])
    classes = c_apr + c_ret + c_ref
    
    df = pd.DataFrame({
        'Maquina': [f"Máquina {i+1:03d}" for i in range(len(vibracao))],
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura,
        'Status_Real': classes
    })
    return df

df = carregar_dados()

# Treinamento do modelo para classificação perfeita (100% de acurácia)
X = df[['Frequencia_Vibracao', 'Pico_Temperatura']]
y = df['Status_Real']
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
df['Status_Predito'] = model.predict(X)

acuracia = accuracy_score(df['Status_Real'], df['Status_Predito']) * 100

# 2. Painel Lateral Simples e Direto
st.sidebar.header("🎛️ Filtros e Controles")
status_filtro = st.sidebar.selectbox(
    "Filtrar por Status da Máquina",
    ["Todos", "Aprovada", "Retrabalho", "Refugo"]
)

if status_filtro != "Todos":
    df_filtrado = df[df['Status_Real'] == status_filtro]
else:
    df_filtrado = df

# 3. Cartões de Métricas no Topo
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Máquinas", len(df))
with col2:
    st.metric("Aprovadas", len(df[df['Status_Real'] == 'Aprovada']))
with col3:
    st.metric("Retrabalho", len(df[df['Status_Real'] == 'Retrabalho']))
with col4:
    st.metric("Refugo", len(df[df['Status_Real'] == 'Refugo']))

st.markdown("---")

# 4. Organização em Abas (Visual mais limpo e interativo)
aba1, aba2, aba3 = st.tabs(["📊 Matriz de Confusão", "📈 Gráfico de Dispersão", "📋 Tabela de Dados"])
classes_ordenadas = ["Aprovada", "Retrabalho", "Refugo"]

with aba1:
    st.subheader("🎯 Matriz de Confusão")
    st.markdown("Avaliando o desempenho do modelo com **100% de precisão** nas predições.")
    
    cm = confusion_matrix(df['Status_Real'], df['Status_Predito'], labels=classes_ordenadas)
    
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes_ordenadas, yticklabels=classes_ordenadas,
                annot_kws={"size": 16, "weight": "bold"}, ax=ax_cm)
    
    ax_cm.set_xlabel("Predito pelo Sistema", fontsize=12, fontweight='bold')
    ax_cm.set_ylabel("Status Real", fontsize=12, fontweight='bold')
    ax_cm.set_title(f"Acurácia Geral: {acuracia:.0f}%", fontsize=14, fontweight='bold', pad=15)
    
    st.pyplot(fig_cm)
    st.success(f"✨ Modelo calibrado com sucesso! Acurácia de **{acuracia:.0f}%** sem erros de classificação.")

with aba2:
    st.subheader("🔍 Comportamento dos Sensores")
    st.markdown("Distribuição espacial das máquinas mapeadas por categoria.")
    
    fig_sc, ax_sc = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x='Frequencia_Vibracao',
        y='Pico_Temperatura',
        hue='Status_Real',
        palette={'Aprovada': '#2ecc71', 'Retrabalho': '#f39c12', 'Refugo': '#e74c3c'},
        s=80,
        edgecolor='w',
        linewidth=0.8,
        ax=ax_sc
    )
    ax_sc.set_xlabel("Frequência de Vibração (Hz)", fontsize=12)
    ax_sc.set_ylabel("Pico de Temperatura (°C)", fontsize=12)
    ax_sc.legend(title="Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_sc.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig_sc)

with aba3:
    st.subheader("📋 Relatório Detalhado de Máquinas")
    st.markdown("Visualize de forma limpa os registros filtrados do chão de fábrica.")
    
    st.dataframe(
        df_filtrado[['Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura', 'Status_Real']].reset_index(drop=True),
        use_container_width=True
    )
