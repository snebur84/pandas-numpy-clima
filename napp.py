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
    direcao_graus = ultima_leitura["vento_dir"]  # ex: 90°
    vel_vento = ultima_leitura["vento_vel"]  # ex: 14.2 km/h

    # 1. Definição das faixas de velocidade
    faixas_vento = [
        {"nome": "Fraco (0-10 km/h)", "r_min": 0, "r_max": 10, "cor": "#1e3a8a"},
        {
            "nome": "Moderado (10-20 km/h)",
            "r_min": 10,
            "r_max": 20,
            "cor": "#2563eb",
        },
        {"nome": "Forte (20-30 km/h)", "r_min": 20, "r_max": 30, "cor": "#d97706"},
        {
            "nome": "Muito Forte (>30 km/h)",
            "r_min": 30,
            "r_max": 40,
            "cor": "#dc2626",
        },
    ]

    fig_vento = go.Figure()

    # 2. Faixas circulares sem exibir legenda no gráfico (showlegend=False)
    angulos_360 = list(range(0, 361, 2))

    for faixa in faixas_vento:
        largura = faixa["r_max"] - faixa["r_min"]
        fig_vento.add_trace(
            go.Barpolar(
                r=[largura] * len(angulos_360),
                theta=angulos_360,
                base=[faixa["r_min"]] * len(angulos_360),
                width=2.5,
                marker_color=faixa["cor"],
                opacity=1.0,
                marker_line_width=0,
                name=faixa["nome"],
                hoverinfo="name",
                showlegend=False,  # Oculta da área do gráfico
            )
        )

    # 3. Haste da seta
    fig_vento.add_trace(
        go.Scatterpolar(
            r=[0, vel_vento],
            theta=[direcao_graus, direcao_graus],
            mode="lines",
            line=dict(color="#ffffff", width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 4. Ponta da seta
    fig_vento.add_trace(
        go.Scatterpolar(
            r=[vel_vento],
            theta=[direcao_graus],
            mode="markers",
            marker=dict(
                symbol="triangle-up", size=12, color="#ffffff", angle=direcao_graus
            ),
            hovertemplate=f"Velocidade: {vel_vento:.1f} km/h<br>Direção: {direcao_graus}°<extra></extra>",
            showlegend=False,
        )
    )

    # 5. Layout compacto otimizado para a coluna de KPI
    fig_vento.update_layout(
        title=dict(text="Direção Vento", font=dict(size=12)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N", "NE", "E", "SE", "S", "SO", "O", "NO"],
                direction="clockwise",
                rotation=90,
                gridcolor="rgba(255, 255, 255, 0.2)",
                tickfont=dict(size=9),
            ),
            radialaxis=dict(
                visible=False,  # Oculta os números radiais para economizar espaço
                range=[0, 40],
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),  # Margens mínimas
        height=140,  # Altura reduzida para alinhar com os KPIs
    )
    st.plotly_chart(fig_vento, width='stretch')

    with st.popover("ℹ️ Escala de Vento", width='stretch'):
        for faixa in faixas_vento:
            st.markdown(
                f"<div style='display: flex; align-items: center; gap: 8px; margin-bottom: 4px;'>"
                f"<div style='width: 14px; height: 14px; background-color: {faixa['cor']}; border-radius: 3px;'></div>"
                f"<span style='font-size: 13px;'>{faixa['nome']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

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
    st.subheader("Análise comparativa (últimos 7 dias)")
    df_plot = df_horario.copy()
    df_plot["data"] = df_plot.index.strftime("%Y-%m-%d")
    opcoes_metricas = {
        "Temperatura (°C)": "temp",
        "Umidade (%)": "umidade",
        "Pressão (hPa)": "pressao",
        "Velocidade do Vento": "vento_vel",
        "Precipitação / Chuva (mm)": "chuva",
        "Ponto de Orvalho (°C)": "ponto_orvalho",
        "Déficit de Pressão de Vapor (VPD)": "vpd",
    }
    metricas_disponiveis = {
        label: col
        for label, col in opcoes_metricas.items()
        if col in df_plot.columns
    }
    metrica_selecionada_label = st.selectbox(
        "Selecione o item a ser comparado:",
        options=list(metricas_disponiveis.keys()),
    )
    coluna_y = metricas_disponiveis[metrica_selecionada_label]
    fig = px.box(
        df_plot,
        x="data",
        y=coluna_y,
        labels={"data": "Dia", coluna_y: metrica_selecionada_label},
        title=f"Distribuição Diária: {metrica_selecionada_label} (Últimos 7 dias)",
        points="outliers",  # Exibe os pontos considerados discrepantes
    )
    fig.update_layout(
        xaxis_title="Dia",
        yaxis_title=metrica_selecionada_label,
        hovermode="x unified",
        xaxis={"type": "category"},  # Garante ordenação sequencial das datas
    )
    st.plotly_chart(fig, width='stretch')

st.divider()