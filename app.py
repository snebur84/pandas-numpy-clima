import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Módulos do projeto existente
from cliente_api import criar_sessao, obter_coordenadas, obter_dados_climaticos
import processamento as proc
import analises_avancadas as adv

# =============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA (CLEAN & EXECUTIVO)
# =============================================================================
st.set_page_config(
    page_title="Monitor Climático Executivo",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. CACHING E CARREGAMENTO DE DADOS
# =============================================================================
@st.cache_data(ttl=1800)
def carregar_dados_cidade(cidade: str):
    sessao = criar_sessao()
    localidade = obter_coordenadas(cidade, sessao)
    
    if not localidade:
        return None, None, "Cidade não encontrada."
        
    dados_raw = obter_dados_climaticos(
        localidade["latitude"],
        localidade["longitude"],
        localidade["timezone"],
        sessao
    )
    
    if not dados_raw or "hourly" not in dados_raw:
        return None, None, "Erro ao obter dados da API."
        
    df_nofilter = proc.criar_dataframe_horario(dados_raw)
    df_horario = proc.filtrar_ate_momento_atual(df_nofilter, localidade["timezone"])
    df_horario = proc.calcular_metricas_físicas_numpy(df_horario)
    df_horario = proc.aplicar_classificacao_np_select(df_horario)
    
    return df_horario, localidade, None


# =============================================================================
# 3. CONTROLES NA BARRA LATERAL (AMPLIAÇÃO/REDUÇÃO DE PERÍODO E VARIÁVEIS)
# =============================================================================
st.sidebar.title("⚙️ Filtros & Período")
cidade_input = st.sidebar.text_input("Cidade:", value="Belo Horizonte")

df_horario, localidade, erro = carregar_dados_cidade(cidade_input)

if erro:
    st.error(f"⚠️ {erro}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Janela Temporal dos Gráficos")

# Filtro rápido de atalho para ampliar ou reduzir o período
opcao_periodo = st.sidebar.radio(
    "Seleção Rápida:",
    options=["Últimas 24h", "Últimos 3 Dias", "Últimos 7 Dias", "Personalizado"],
    index=2
)

data_min_base = df_horario.index.min().date()
data_max_base = df_horario.index.max().date()

# Aplicação da lógica de período rápida vs personalizada
if opcao_periodo == "Últimas 24h":
    df_filtrado = df_horario.tail(24)
elif opcao_periodo == "Últimos 3 Dias":
    df_filtrado = df_horario.tail(72)
elif opcao_periodo == "Últimos 7 Dias":
    df_filtrado = df_horario
else:
    # Filtro seletor livre de datas
    intervalo = st.sidebar.date_input(
        "Intervalo Personalizado:",
        value=(data_min_base, data_max_base),
        min_value=data_min_base,
        max_value=data_max_base
    )
    if isinstance(intervalo, tuple) and len(intervalo) == 2:
        d_ini, d_fim = intervalo
        df_filtrado = df_horario[(df_horario.index.date >= d_ini) & (df_horario.index.date <= d_fim)]
    else:
        df_filtrado = df_horario

if df_filtrado.empty:
    st.warning("⚠️ Nenhum registro encontrado para o período selecionado.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Variável Estatística")

mapa_variaveis = {
    "Temperatura (°C)": "temp",
    "Umidade Relativa (%)": "umidade",
    "Pressão Atmosférica (hPa)": "pressao",
    "Velocidade do Vento (km/h)": "vento_vel"
}

var_rotulo = st.sidebar.selectbox("Variável do Estudo:", list(mapa_variaveis.keys()))
var_coluna = mapa_variaveis[var_rotulo]


# =============================================================================
# 4. KPIS DE IMPACTO DIRETO COM INDICADOR POLAR DE DIREÇÃO DO VENTO
# =============================================================================
ultima_leitura = df_filtrado.iloc[-1]

st.title(f"🌤️ Clima em {localidade['name']}, {localidade.get('country', '')}")
st.caption(f"Última atualização: {df_filtrado.index[-1].strftime('%d/%m/%Y às %H:%M')}")

# Layout dos KPIs + Gráfico Polar de Vento
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_vento = st.columns([1, 1, 1, 1, 1.2])

col_kpi1.metric("Temperatura", f"{ultima_leitura['temp']:.1f} °C")
col_kpi2.metric("Umidade", f"{ultima_leitura['umidade']:.0f} %")
col_kpi3.metric("Pressão", f"{ultima_leitura['pressao']:.1f} hPa")
col_kpi4.metric("Condição", f"{ultima_leitura['categoria_clima']}")

# --- MINI GRÁFICO POLAR DA DIREÇÃO DO VENTO ---
with col_vento:
    direcao_graus = ultima_leitura["vento_dir"]
    vel_vento = ultima_leitura["vento_vel"]
    
    # Montagem do ponteiro de bússola usando Plotly Polar
    fig_vento = go.Figure(go.Barpolar(
        r=[vel_vento],
        theta=[direcao_graus],
        width=[25],
        marker_color="#2563eb",
        marker_line_color="#1e40af",
        marker_line_width=1,
        opacity=0.85
    ))
    
    fig_vento.update_layout(
        title=dict(text="Direção Vento", font=dict(size=12)),
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N", "NE", "E", "SE", "S", "SO", "O", "NO"],
                direction="clockwise",
                rotation=90
            ),
            radialaxis=dict(visible=False)
        ),
        margin=dict(l=10, r=10, t=25, b=10),
        height=130,
        showlegend=False,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_vento, use_container_width=True)

st.divider()


# =============================================================================
# 5. TENDÊNCIA TEMPORAL E MÉDIA MÓVEL (RESPEITANDO O FILTRO DE PERÍODO)
# =============================================================================
st.subheader("📈 Tendência Temporal e Média Móvel")

df_filtrado["media_movil_12h"] = df_filtrado[var_coluna].rolling(window=12, min_periods=1).mean()
tendencia = adv.calcular_tendencia_linear(df_filtrado, coluna=var_coluna)

fig_linha = go.Figure()

# Linha de medição bruta
fig_linha.add_trace(go.Scatter(
    x=df_filtrado.index, 
    y=df_filtrado[var_coluna], 
    mode='lines', 
    name='Medição Horária',
    line=dict(color='#cbd5e1', width=1.5)
))

# Linha de tendência média
fig_linha.add_trace(go.Scatter(
    x=df_filtrado.index, 
    y=df_filtrado["media_movil_12h"], 
    mode='lines', 
    name='Média Móvel',
    line=dict(color='#2563eb', width=3)
))

fig_linha.update_layout(
    title=f"Evolução no Período Selecionado ({var_rotulo}) — Tendência: {tendencia['tendencia']}",
    xaxis_title="Data / Hora",
    yaxis_title=var_rotulo,
    template="plotly_white",
    height=360,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig_linha, use_container_width=True)

st.divider()


# =============================================================================
# 6. ANÁLISE DE OUTLIERS E DISTRIBUIÇÃO (DISPERSÃO, BOXPLOT, HISTOGRAMA)
# =============================================================================
col_g1, col_g2, col_g3 = st.columns([1.2, 1, 1])

# --- DISPERSÃO E DETECÇÃO DE OUTLIERS ---
with col_g1:
    st.subheader("🎯 Outliers (Dispersão)")
    
    media_val = df_filtrado[var_coluna].mean()
    std_val = df_filtrado[var_coluna].std()
    
    # Evita divisão por zero se a variância for nula
    if std_val > 0:
        z_scores = (df_filtrado[var_coluna] - media_val) / std_val
    else:
        z_scores = 0
        
    df_filtrado["outlier"] = np.abs(z_scores) > 2.0
    
    fig_scatter = px.scatter(
        df_filtrado,
        x=df_filtrado.index,
        y=var_coluna,
        color="outlier",
        color_discrete_map={False: "#64748b", True: "#ef4444"},
        title="Medições Atípicas (|Z-Score| > 2.0)"
    )
    fig_scatter.update_layout(template="plotly_white", height=320, showlegend=False)
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- BOXPLOT DEDICADO ---
with col_g2:
    st.subheader("📦 Boxplot (Quartis & IQR)")
    
    fig_box = px.box(
        df_filtrado,
        y=var_coluna,
        points="outliers",
        color_discrete_sequence=["#10b981"],
        title="Quartuis, Mediana e Intervalo IQR"
    )
    fig_box.update_layout(template="plotly_white", height=320)
    st.plotly_chart(fig_box, use_container_width=True)

# --- HISTOGRAMA DE FREQUÊNCIA ---
with col_g3:
    st.subheader("📊 Frequência")
    
    fig_hist = px.histogram(
        df_filtrado,
        x=var_coluna,
        nbins=15,
        color_discrete_sequence=["#6366f1"],
        title="Distribuição de Leituras"
    )
    fig_hist.update_layout(template="plotly_white", height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()


# =============================================================================
# 7. TABELA DE CÁLCULOS ESTATÍSTICOS
# =============================================================================
st.subheader("📐 Parâmetros Estatísticos Calculados no Período Exibido")

serie = df_filtrado[var_coluna].dropna()

q1 = serie.quantile(0.25)
q3 = serie.quantile(0.75)
iqr = q3 - q1

df_estatistico = pd.DataFrame({
    "Categoria": [
        "Tendência Central", "Tendência Central", "Tendência Central",
        "Posição Relativa", "Posição Relativa", "Posição Relativa",
        "Dispersão", "Dispersão", "Dispersão"
    ],
    "Métrica Estatística": [
        "Média (x̄)", "Mediana (Md)", "Moda (Mo)",
        "1º Quartil (Q1)", "3º Quartil (Q3)", "Intervalo Interquartil (IQR)",
        "Amplitude Total", "Variância (s²)", "Desvio Padrão (s)"
    ],
    "Valor Calculado": [
        f"{serie.mean():.2f}", f"{serie.median():.2f}", f"{serie.mode()[0]:.2f}" if not serie.mode().empty else "N/A",
        f"{q1:.2f}", f"{q3:.2f}", f"{iqr:.2f}",
        f"{(serie.max() - serie.min()):.2f}", f"{serie.var():.2f}", f"{serie.std():.2f}"
    ]
})

st.table(df_estatistico)

with st.expander("📄 Exportar e Visualizar Dados Brutos do Período (CSV)"):
    st.dataframe(df_filtrado, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=df_filtrado.to_csv().encode("utf-8"),
        file_name=f"clima_{localidade['name'].lower()}.csv",
        mime="text/csv"
    )