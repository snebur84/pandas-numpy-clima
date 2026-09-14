import numpy as np
import pandas as pd


def analisar_picos_por_horario(df: pd.DataFrame) -> pd.DataFrame:
    """Groupby temporal extraindo a hora do índice
    e agregando médias para entender o ciclo diurno.
    """
    df_perfil_diurno = df.groupby(df.index.hour).agg(
        temp_media=("temp", "mean"),
        umidade_media=("umidade", "mean"),
        vento_max=("vento_vel", "max"),
        vento_medio=("vento_vel", "mean"),
    ).round(2)
    df_perfil_diurno.index.name = "hora_dia"
    return df_perfil_diurno


def calcular_matriz_correlacao(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação linear (Pearson) no Pandas
    para identificar o grau de dependência entre variáveis.
    """
    colunas_interesse = ["temp", "umidade", "pressao", "chuva", "vpd"]
    return df[colunas_interesse].corr()


def simular_e_tratar_dados_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Demonstra: Geração de ruído/NaNs em arrays e recomposição

    de séries temporais usando interpolação linear no Pandas.
    """
    df_copia = df.copy()

    # Simula falha de leitura em 5% dos registros
    np.random.seed(42)
    mascara = np.random.rand(len(df_copia)) < 0.05
    df_copia.loc[mascara, "temp"] = np.nan

    # Recupera os dados faltantes via interpolação baseada no tempo
    df_copia["temp_interpolada"] = df_copia["temp"].interpolate(method="time")
    return df_copia


def detectar_anomalias_zscore(
    df: pd.DataFrame, coluna: str = "temp", limiar: float = 2.0
) -> pd.DataFrame:
    """ Uso de np.mean e np.std para cálculo do Z-Score
    e detecção de anomalias/outliers estatísticos.
    """
    dados = df[coluna].to_numpy()
    media = np.mean(dados)
    desvio_padrao = np.std(dados)

    # Cálculo vetorizado: Z = (X - μ) / σ
    z_scores = (dados - media) / desvio_padrao

    df_anomalias = df.copy()
    df_anomalias[f"zscore_{coluna}"] = z_scores
    df_anomalias["eh_anomalia"] = np.abs(z_scores) > limiar

    return df_anomalias[df_anomalias["eh_anomalia"]]


def calcular_tendencia_linear(df: pd.DataFrame, coluna: str = "temp") -> dict:
    """Regressão linear (np.polyfit) no NumPy para identificar
    a taxa de variação (inclinação da reta) ao longo da série.
    """
    # Cria eixo x numérico ordinal (0, 1, 2, ..., n-1)
    x = np.arange(len(df))
    y = df[coluna].to_numpy()

    # Ajuste polinomial de 1º grau (y = ax + b)
    inclinacao, interseccao = np.polyfit(x, y, 1)

    return {
        "taxa_variacao_por_hora": inclinacao,
        "interseccao": interseccao,
        "tendencia": "Aquecimento" if inclinacao > 0 else "Resfriamento",
    }


def reconstituir_vetor_vento(df: pd.DataFrame) -> pd.DataFrame:
    """Operações inversas no NumPy com np.hypot e np.arctan2
    para recalcular magnitude e direção do vento a partir das componentes U e V.
    """
    u = df["vento_u"].to_numpy()
    v = df["vento_v"].to_numpy()

    # 1. Magnitude: sqrt(U² + V²)
    df["vento_vel_reconstituido"] = np.hypot(u, v)

    # 2. Direção em graus
    rad = np.arctan2(-u, -v)
    df["vento_dir_reconstituido"] = np.rad2deg(rad) % 360

    return df