import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard 3D de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard 3D de Manutenção Preditiva")
st.markdown(
    "Monitoramento avançado de sensores com agrupamento **DBSCAN** em modelo tridimensional e visualização interativa."
)

# 1. Geração de Dados com NumPy (Incluindo 3 métricas para o gráfico 3D)
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n_samples = 350
    
    # Dados normais (funcionamento padrão)
    vibracao_normal = np.random.normal(loc=50, scale=4.5, size=n_samples)
    temperatura_normal = np.random.normal(loc=70, scale=5.0, size=n_samples)
    ruido_normal = np.random.normal(loc=60, scale=3.0, size=n_samples)
    
    # Inclusão intencional de outliers (anomalias)
    n_outliers = 20
    vibracao_outliers = np.random.uniform(low=20, high=95, size=n_outliers)
    temperatura_outliers = np.random.uniform(low=45, high=105, size=n_outliers)
    ruido_outliers = np.random.uniform(low=40, high=85, size=n_outliers)
    
    # Concatenando os dados
    vibracao = np.concatenate([vibracao_normal, vibracao_outliers])
    temperatura = np.concatenate([temperatura_normal, temperatura_outliers])
    ruido = np.concatenate([ruido_normal, ruido_outliers])
    
    # Identificadores das máquinas
    maquinas = [f"Maquina_{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        'Maquina': maquinas,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura,
        'Nivel_Ruido': ruido
    })
    return df

df_sensores = carregar_dados()

# 2. Controles na Barra Lateral para ajustar o DBSCAN
st.sidebar.header("🔧 Parâmetros do DBSCAN")
eps_val = st.sidebar.slider(
    "Eps (Raio da vizinhança)", 
    min_value=0.5, max_value=10.0, value=3.5, step=0.1
)
min_samples_val = st.sidebar.slider(
    "Min Samples (Mínimo de pontos)", 
    min_value=2, max_value=15, value=5, step=1
)

# 3. Processamento com DBSCAN considerando as 3 métricas
X = df_sensores[['Frequencia_Vibracao', 'Pico_Temperatura', 'Nivel_Ruido']]
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
clusters = dbscan.fit_predict(X)

# Rótulo amigável para os clusters e anomalias
df_sensores['Cluster'] = [str(c) if c != -1 else 'Anomalia (-1)' for c in clusters]

# 4. Gráfico 3D de Dispersão (Sem legenda e com valores ao passar o rato)
st.subheader("📊 Gráfico 3D: Frequência x Temperatura x Ruído")

# Mapeamento de cores: Outliers em preto, demais clusters automáticos
color_map = {'Anomalia (-1)': 'black'}

fig = px.scatter_3d(
    df_sensores,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    z='Nivel_Ruido',
    color='Cluster',
    hover_name='Maquina',
    hover_data={
        'Frequencia_Vibracao': ':.2f',
        'Pico_Temperatura': ':.2f',
        'Nivel_Ruido': ':.2f',
        'Cluster': False
    },
    color_discrete_map=color_map,
    opacity=0.85
)

fig.update_layout(
    showlegend=False,  # Removida a legenda conforme solicitado
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis_title='Frequência de Vibração (Hz)',
        yaxis_title='Pico de Temperatura (°C)',
        zaxis_title='Nível de Ruído (dB)'
    )
)

st.plotly_chart(fig, use_container_width=True)

# 5. Tabela Interativa com Filtro (Normais vs Anomalias)
st.markdown("---")
st.subheader("📋 Gestão de Máquinas e Filtro de Estado")

filtro_status = st.selectbox(
    "Filtrar tabela por tipo de comportamento:",
    ["Todos os Registros", "Apenas Máquinas Normais (Clusters)", "Apenas Anomalias (-1)"]
)

if filtro_status == "Apenas Anomalias (-1)":
    df_filtrado = df_sensores[df_sensores['Cluster'] == 'Anomalia (-1)']
elif filtro_status == "Apenas Máquinas Normais (Clusters)":
    df_filtrado = df_sensores[df_sensores['Cluster'] != 'Anomalia (-1)']
else:
    df_filtrado = df_sensores

if not df_filtrado.empty:
    st.dataframe(df_filtrado.reset_index(drop=True), use_container_width=True)
else:
    st.info("Nenhum registo encontrado para o filtro selecionado.")
