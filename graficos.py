import matplotlib.pyplot as plt
import pandas as pd

def cria_grafico(df: pd.DataFrame):
    valores = df["temp"].tail(10)
    plt.plot(valores)
    plt.style.use("classic")
    plt.title("Grafico de Linhas Simples")
    plt.xlabel("Hora")
    plt.grid(True)
    plt.ylabel("Temp")
    plt.ylim(0,50)
    plt.show()
