import sys
from cliente_api import criar_sessao, obter_coordenadas, obter_dados_climaticos
import processamento as proc
import analises_avancadas as adv
import graficos as gr


def executar(cidade: str):
    sessao = criar_sessao()

    print(f"1. Buscando coordenadas para '{cidade}'...")
    localidade = obter_coordenadas(cidade, sessao)
    if not localidade:
        print("Cidade não encontrada.")
        return

    print("2. Requisitando séries temporais da API...")
    dados_raw = obter_dados_climaticos(
        localidade["latitude"],
        localidade["longitude"],
        localidade["timezone"],
        sessao,
    )
    if not dados_raw or "hourly" not in dados_raw:
        print("Falha ao recuperar dados climáticos.")
        return

    # --- PIPELINE PANDAS / NUMPY ---
    print("3. Processando dados no Pandas e NumPy...")
    df_nofilter = proc.criar_dataframe_horario(dados_raw)
    df_horario = proc.filtrar_ate_momento_atual(df_nofilter, localidade["timezone"])
    df_horario = proc.calcular_metricas_físicas_numpy(df_horario)
    df_horario = proc.aplicar_classificacao_np_select(df_horario)
    df_horario, df_diario = proc.agregar_dados_pandas(df_horario)
    # --- EXIBIÇÃO DE DADOS TRATADOS --- #
    print('# --- EXIBIÇÃO DE DADOS TRATADOS --- #')
    print('# --- df_horario --- #')
    print(df_horario)
    print('# --- df_diario --- #')
    print(df_diario)
    print('# --- FIM EXIBIÇÃO DE DADOS TRATADOS --- #')

    # --- EXIBIÇÃO DE RESULTADOS DIDÁTICOS ---
    print(f"\n=== RESUMO DIÁRIO (PANDAS RESAMPLE) - {cidade.upper()} ===")
    print(df_diario.tail(5))

    print(f"\n=== ÚLTIMA LEITURA HORÁRIA (CÁLCULOS NUMPY) ===")
    ultima_linha = df_horario.iloc[-1]
    print(f"Horário:            {df_horario.index[-1]}")
    print(f"Temperatura:        {ultima_linha['temp']} °C")
    print(f"Ponto de Orvalho:   {ultima_linha['ponto_orvalho']:.2f} °C")
    print(f"VPD:                {ultima_linha['vpd']:.2f} hPa")
    # Direção do vento: U > 0 Oeste -> Leste, U < 0 Leste -> Oeste; V > 0 Sul -> Norte, V < 0 Norte -> Sul
    print(f"Vento (U/V):        U={ultima_linha['vento_u']:.1f}, V={ultima_linha['vento_v']:.1f}")
    print(f"Classificação:      {ultima_linha['categoria_clima']}")

    # --- EXIBIÇÃO DE ANALISES AVANÇADAS --- #
    print('# --- EXIBIÇÃO DE ANALISES AVANÇADAS --- #')
    # Exemplo 1: Matriz de Correlação
    matriz_corr = adv.calcular_matriz_correlacao(df_horario)
    print("\n=== MATRIZ DE CORRELAÇÃO (PANDAS) ===")
    print(matriz_corr)

    # Exemplo 2: Tendência de Temperatura
    tendencia = adv.calcular_tendencia_linear(df_horario, coluna="temp")
    print(
        f"\nTendência Térmica: {tendencia['tendencia']} ({tendencia['taxa_variacao_por_hora']:.3f} °C/hora)"
    )

    # Exemplo 3: Anomalias de Vento
    anomalias_vento = adv.detectar_anomalias_zscore(
        df_horario, coluna="vento_vel", limiar=2.5
    )
    print(f"\nRegistros com rajadas atípicas (Z-Score > 2.5): {len(anomalias_vento)}\n")

    # Exemplo 4: Picos diários
    picos = adv.analisar_picos_por_horario(df_horario)
    print(picos)
    gr.cria_grafico(df_horario)


if __name__ == "__main__":
    cidade_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Cidade: ").strip()
    if cidade_input:
        executar(cidade_input)
    else:
        print("Nenhuma cidade foi informada.")
