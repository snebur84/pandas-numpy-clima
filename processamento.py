import numpy as np
import pandas as pd


def filtrar_ate_momento_atual(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Tratamento de fuso horário (tz-aware vs tz-naive) 
    e filtragem booleana em DatetimeIndex no Pandas.
    """
    # 1. Captura o momento exato de agora no fuso correto (ex: 'America/Sao_Paulo')
    agora = pd.Timestamp.now(tz=timezone)

    # 2. Garante que o índice do DataFrame possua a informação de fuso horário
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)

    # 3. Filtra mantendo apenas os registros até a hora atual
    df_filtrado = df[df.index <= agora]

    return df_filtrado

def criar_dataframe_horario(dados_raw: dict) -> pd.DataFrame:
    """Criação de DataFrame a partir de dicionários, 

    conversão de colunas em datetime, definição de Index e renomeação.
    """
    df = pd.DataFrame(dados_raw["hourly"])

    # PANDAS: Conversão de tipos e Série Temporal
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    # PANDAS: Renomeação padronizada de colunas
    df = df.rename(
        columns={
            "temperature_2m": "temp",
            "relative_humidity_2m": "umidade",
            "surface_pressure": "pressao",
            "wind_speed_10m": "vento_vel",
            "wind_direction_10m": "vento_dir",
            "precipitation": "chuva",
        }
    )
    return df


def calcular_metricas_físicas_numpy(df: pd.DataFrame) -> pd.DataFrame:
    """Demonstra: Extração de Arrays NumPy (.to_numpy()) e cálculos

    vetorizados com funções matemáticas nativas do NumPy.
    """
    # Extraindo Ndarrays do DataFrame
    temp = df["temp"].to_numpy()
    umid = df["umidade"].to_numpy()
    v_vel = df["vento_vel"].to_numpy()
    v_dir = df["vento_dir"].to_numpy()

    # 1. NUMPY: Equação de Magnus para Ponto de Orvalho (Dew Point)
    a, b = 17.27, 237.7
    alpha = ((a * temp) / (b + temp)) + np.log(umid / 100.0)
    df["ponto_orvalho"] = (b * alpha) / (a - alpha)

    # 2. NUMPY: Déficit de Pressão de Vapor (VPD)
    es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3)) * 10
    df["vpd"] = es * (1.0 - (umid / 100.0))

    # 3. NUMPY: Decomposição vetorial do vento (Álgebra/Trigonometria)
    rad = np.deg2rad(v_dir)
    df["vento_u"] = -v_vel * np.sin(rad)  # Componente Leste-Oeste
    df["vento_v"] = -v_vel * np.cos(rad)  # Componente Norte-Sul

    return df


def aplicar_classificacao_np_select(df: pd.DataFrame) -> pd.DataFrame:
    """Demonstra: np.select para lógica condicional vetorizada complexa

    (sem uso de 'if' ou laços 'for').
    """
    condicoes = [
        (df["temp"] >= 30.0) & (df["umidade"] >= 70.0),
        (df["temp"] >= 30.0) & (df["umidade"] < 70.0),
        (df["temp"] < 30.0) & (df["temp"] >= 18.0),
        df["temp"] < 18.0,
    ]
    rotulos = ["Calor Abafado", "Calor Seco", "Agradável", "Frio"]

    df["categoria_clima"] = np.select(condicoes, rotulos, default="Indefinido")
    return df


def agregar_dados_pandas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstra: Reamostragem temporal (resample), agregações múltiplas,

    e janelas móveis (rolling).
    """
    # PANDAS: Médias móveis de 24 horas para tendência
    df["temp_media_movil_24h"] = df["temp"].rolling(window=24).mean()

    # PANDAS: Reamostragem diária com agregações customizadas
    df_diario = df.resample("D").agg(
        {
            "temp": ["mean", "min", "max"],
            "umidade": "mean",
            "chuva": "sum",
            "vpd": "mean",
        }
    )

    # Ajuste dos nomes das colunas após agregação múltipla
    df_diario.columns = ["_".join(col).strip() for col in df_diario.columns.values]

    return df, df_diario