import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva")
st.markdown(
    "Monitoramento de sensores de chão de fábrica e detecção automática de anomalias utilizando o algoritmo **DBSCAN**."
)

# 1. Geração de Dados com NumPy
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n_samples = 350
    
    # Dados normais (funcionamento padrão das máquinas)
    vibracao_normal = np.random.normal(loc=50, scale=4.5, size=n_samples)
    temperatura_normal = np.random.normal(loc=70, scale=5.0, size=n_samples)
    
    # Inclusão intencional de outliers (anomalias)
    n_outliers = 20
    vibracao_outliers = np.random.uniform(low=20, high=95, size=n_outliers)
    temperatura_outliers = np.random.uniform(low=45, high=105, size=n_outliers)
    
    # Concatenando os dados
    vibracao = np.concatenate([vibracao_normal, vibracao_outliers])
    temperatura = np.concatenate([temperatura_normal, temperatura_outliers])
    
    # Identificadores das máquinas
    maquinas = [f"Maquina_{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        'Maquina': maquinas,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura
    })
    return df

df_sensores = carregar_dados()

# 2. Controles na Barra Lateral (Sidebar) para ajustar o DBSCAN
st.sidebar.header("🔧 Parâmetros do DBSCAN")
eps_val = st.sidebar.slider(
    "Eps (Raio da vizinhança)", 
    min_value=0.5, max_value=5.0, value=1.8, step=0.1,
    help="Define a distância máxima entre duas amostras para que uma seja considerada no Vizinhança da outra."
)
min_samples_val = st.sidebar.slider(
    "Min Samples (Mínimo de pontos)", 
    min_value=2, max_value=15, value=5, step=1,
    help="Número de amostras em uma vizinhança para que um ponto seja considerado um núcleo."
)

# 3. Processamento com DBSCAN
X = df_sensores[['Frequencia_Vibracao', 'Pico_Temperatura']]
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
df_sensores['Cluster'] = dbscan.fit_predict(X)

# 4. Layout Principal: Gráfico de Dispersão
st.subheader("📊 Comportamento dos Sensores de Chão de Fábrica")

fig, ax = plt.subplots(figsize=(10, 6))

# Mapeamento de cores: Outliers (-1) ficam em preto, clusters normais ganham cores da paleta
clusters_unicos = sorted(df_sensores['Cluster'].unique())
palette = sns.color_palette("Set2", n_colors=len(clusters_unicos) - (1 if -1 in clusters_unicos else 0))

color_map = {}
palette_idx = 0
for cluster in clusters_unicos:
    if cluster == -1:
        color_map[cluster] = 'black'  # Outliers destacados em preto
    else:
        color_map[cluster] = palette[palette_idx % len(palette)]
        palette_idx += 1

sns.scatterplot(
    data=df_sensores,
    x='Frequencia_Vibracao',
    y='Pico_Temperatura',
    hue='Cluster',
    palette=color_map,
    style='Cluster',
    s=70,
    ax=ax,
    edgecolor='w',
    linewidth=0.5
)

ax.set_title("Agrupamento e Detecção de Anomalias (DBSCAN)", fontsize=14, fontweight='bold')
ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=12)
ax.set_ylabel("Pico de Temperatura (°C)", fontsize=12)
ax.legend(title="Clusters / Anomalias (-1)", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig)

# 5. Tabela Interativa de Anomalias
st.markdown("---")
st.subheader("⚠️ Máquinas Consideradas Anômalas")

df_anomalias = df_sensores[df_sensores['Cluster'] == -1]

if not df_anomalias.empty:
    st.warning(f"Atenção! Foram identificadas **{len(df_anomalias)}** máquinas com comportamento fora do padrão sob os parâmetros atuais.")
    st.dataframe(df_anomalias.reset_index(drop=True), use_container_width=True)
else:
    st.success("Tudo em ordem! Nenhuma anomalia (outlier) foi detectada com os parâmetros atuais.")
