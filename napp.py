import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Módulos do projeto existente
from cliente_api import criar_sessao, obter_coordenadas, obter_dados_climaticos
import processamento as proc
import analises_avancadas as adv


st.set_page_config(
    page_title="Monitor Climático Executivo",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.title("Monitor Climático")
cidade = st.text_input("Cidade:", value="Belo Horizonte")

df_horario, localidade, erro = carregar_dados_cidade(cidade)

if erro:
    st.error(f"⚠️ {erro}")
    st.stop()

ultima_leitura = df_horario.iloc[-1]

st.divider()

st.subheader(f"🌤️ Clima em {localidade['name']}, {localidade.get('country', '')}")
st.caption(f"Última atualização: {df_horario.index[-1].strftime('%d/%m/%Y às %H:%M')}")

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
    
    st.plotly_chart(fig_vento, width='stretch')

st.divider()

col_g1, col_g2 = st.columns([1,1])
with col_g1:
    st.subheader("Variação temperatura (últimas 24h)")
    df_24 = df_horario.tail(24)
    temp_dia = px.line(
        df_24,
        x=df_24.index,
        y='temp'
        )
    st.plotly_chart(temp_dia, width='stretch')

with col_g2:
    

st.divider()