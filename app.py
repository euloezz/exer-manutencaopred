import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Configuração da página em modo compacto
st.set_page_config(
    page_title="Dashboard Preditivo Compacto",
    page_icon="⚡",
    layout="wide"
)

# Estilo CSS para compactar o espaçamento
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .stMetric { background-color: #f1f3f5; padding: 10px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Dashboard Compacto de Manutenção Preditiva")

# Controlo na barra lateral para gerar novos dados aleatórios
st.sidebar.header("⚙️ Controlos")
if st.sidebar.button("🔄 Gerar Novos Dados Aleatórios"):
    st.cache_data.clear()

# 1. Geração de Dados Aleatórios
@st.cache_data
def gerar_dados_aleatorios():
    np.random.seed(None) # Garante aleatoriedade real
    n = 40
    
    # Dados aleatórios para cada categoria
    v_apr = np.random.uniform(30, 45, n)
    t_apr = np.random.uniform(50, 65, n)
    
    v_ret = np.random.uniform(50, 65, n)
    t_ret = np.random.uniform(70, 85, n)
    
    v_ref = np.random.uniform(70, 95, n)
    t_ref = np.random.uniform(85, 105, n)
    
    vibracao = np.concatenate([v_apr, v_ret, v_ref])
    temperatura = np.concatenate([t_apr, t_ret, t_ref])
    classes = ['Aprovada']*n + ['Retrabalho']*n + ['Refugo']*n
    
    df = pd.DataFrame({
        'Maquina': [f"Máq {i+1:02d}" for i in range(len(vibracao))],
        'Vibracao': vibracao,
        'Temperatura': temperatura,
        'Status_Real': classes
    })
    return df

df = gerar_dados_aleatorios()

# Modelo para garantir 100% de precisão nos dados gerados
X = df[['Vibracao', 'Temperatura']]
y = df['Status_Real']
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X, y)
df['Status_Predito'] = model.predict(X)
acuracia = accuracy_score(df['Status_Real'], df['Status_Predito']) * 100

# 2. Métricas Compactas numa única linha
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total", len(df))
c2.metric("Aprovadas", len(df[df['Status_Real'] == 'Aprovada']))
c3.metric("Retrabalho", len(df[df['Status_Real'] == 'Retrabalho']))
c4.metric("Refugo", len(df[df['Status_Real'] == 'Refugo']))
c5.metric("Acurácia", f"{acuracia:.0f}%")

st.markdown("---")

# 3. Layout Compacto Lado a Lado (Gráficos e Tabela)
col_esq, col_dir = st.columns(2)

classes_ordenadas = ["Aprovada", "Retrabalho", "Refugo"]

with col_esq:
    st.subheader("🎯 Matriz de Confusão")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
    cm = confusion_matrix(df['Status_Real'], df['Status_Predito'], labels=classes_ordenadas)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes_ordenadas, yticklabels=classes_ordenadas,
                annot_kws={"size": 12, "weight": "bold"}, ax=ax_cm)
    ax_cm.set_xlabel("Predito", fontsize=10)
    ax_cm.set_ylabel("Real", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_cm)

with col_dir:
    st.subheader("📊 Dispersão de Sensores")
    fig_sc, ax_sc = plt.subplots(figsize=(5, 3.5))
    sns.scatterplot(
        data=df, x='Vibracao', y='Temperatura', hue='Status_Real',
        palette={'Aprovada': '#2ecc71', 'Retrabalho': '#f39c12', 'Refugo': '#e74c3c'},
        s=50, ax=ax_sc
    )
    ax_sc.set_xlabel("Vibração (Hz)", fontsize=10)
    ax_sc.set_ylabel("Temperatura (°C)", fontsize=10)
    ax_sc.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_sc)

# 4. Tabela Compacta Ocultável/Expansível
with st.expander("📋 Ver Tabela de Dados Detalhada"):
    st.dataframe(df[['Maquina', 'Vibracao', 'Temperatura', 'Status_Real']].reset_index(drop=True), use_container_width=True, height=200)
