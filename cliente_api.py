import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def criar_sessao() -> requests.Session:
    sessao = requests.Session()
    estrategia = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adaptador = HTTPAdapter(max_retries=estrategia)
    sessao.mount("https://", adaptador)
    sessao.mount("http://", adaptador)
    return sessao


def obter_coordenadas(cidade: str, sessao: requests.Session) -> dict | None:
    url_geo = f"https://geocoding-api.open-meteo.com/v1/search?name={cidade}&count=1&language=pt&format=json"
    try:
        resp = sessao.get(url_geo, timeout=(5, 15))
        resp.raise_for_status()
        dados = resp.json()
        if dados.get("results"):
            return dados["results"][0]
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar coordenadas de '{cidade}': {e}")
    return None


def obter_dados_climaticos(lat: float, lon: float, timezone: str, sessao: requests.Session) -> dict | None:
    # Solicitando dados horários dos últimos 7 dias
    url_clima = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation"
        f"&timezone={timezone}&past_days=7&forecast_days=1"
    )
    try:
        resp = sessao.get(url_clima, timeout=(5, 15))
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao obter dados climáticos: {e}")
        return None