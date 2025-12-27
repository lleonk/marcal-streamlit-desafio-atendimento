# parametros_v3.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from math import lgamma, sqrt, ceil, pi, exp


# ============================================================
# Funções auxiliares (para distribuição e demanda)
# ============================================================

def poisson_pmf(k, lam):
    k = np.asarray(k, dtype=float)
    log_pmf = -lam + k * np.log(lam) - np.vectorize(lgamma)(k + 1.0)
    return np.exp(log_pmf)


def normal_pdf(x, mu, sigma2):
    x = np.asarray(x, dtype=float)
    return (1.0 / np.sqrt(2 * pi * sigma2)) * np.exp(-(x - mu) ** 2 / (2 * sigma2))


# ============================================================
# Dataclasses de configuração e dados
# ============================================================

@dataclass
class DemandScenarioConfig:
    """Configuração básica de um cenário de demanda (toy problem)."""

    # Identidade / narrativa
    nome: str
    descricao: str

    # Janela de funcionamento
    hora_inicio: int          # ex.: 9
    hora_fim: int             # ex.: 18 (exclusivo)

    # Perfil de demanda
    horas_pico: List[int]     # ex.: [11,12,13]
    media_pico: float         # lambda em horas de pico
    media_vale: float         # lambda em horas de vale
    seed_demanda: int | None  # para reprodutibilidade (ou None)

    # Time base por tipo (quantos funcionários existem no quadro "normal")
    # ex.: {"junior": 5, "pleno": 3, "senior": 2}
    base_team_por_tipo: Dict[str, int]

    # Quantos funcionários adicionais podem ser contratados no total
    # ex.: 0 (sem extra), 5 (pode contratar até mais 5 funcionários quaisquer)
    extra_max_total: int

    # Ticket médio (R$) – usado como penalidade por cliente perdido
    ticket_medio: float


@dataclass
class DemandScenarioData:
    """Dados gerados a partir de uma DemandScenarioConfig."""
    config: DemandScenarioConfig

    horas: np.ndarray                # shape (T,), horas inteiras [9,10,...]
    lambdas: np.ndarray              # shape (T,), lambda por hora
    clientes_por_hora: np.ndarray    # shape (T,), nº de clientes em cada hora

    # Chegadas contínuas ao longo do dia
    arrival_times_min: np.ndarray    # shape (N_total,), minutos desde hora_inicio
    arrival_hour_index: np.ndarray   # shape (N_total,), índice da hora (0..T-1)


# ============================================================
# Função geradora: de config -> dados (incluindo chegadas contínuas)
# ============================================================

def gerar_dados_cenario(config: DemandScenarioConfig) -> DemandScenarioData:
    hora_inicio = config.hora_inicio
    hora_fim = config.hora_fim

    if hora_fim <= hora_inicio:
        raise ValueError("hora_fim deve ser maior que hora_inicio.")

    # Horas discretas do dia
    horas = np.arange(hora_inicio, hora_fim)

    # Lambda de cada hora (pico x vale)
    lambdas = np.array(
        [
            config.media_pico if h in config.horas_pico else config.media_vale
            for h in horas
        ],
        dtype=float,
    )

    # Gera nº de clientes por hora (demanda fixa do cenário)
    if config.seed_demanda is not None:
        np.random.seed(config.seed_demanda)
    clientes_por_hora = np.random.poisson(lambdas)

    # Chegadas individuais (uniformes dentro da hora)
    arrival_times_min_list: List[float] = []
    arrival_hour_index_list: List[int] = []

    for idx, h in enumerate(horas):
        n = int(clientes_por_hora[idx])
        if n <= 0:
            continue

        inicio_min = (h - hora_inicio) * 60.0
        fim_min = inicio_min + 60.0
        times_min = np.linspace(inicio_min, fim_min, n, endpoint=False)

        arrival_times_min_list.append(times_min)
        arrival_hour_index_list.append(np.full(n, idx, dtype=int))

    if arrival_times_min_list:
        arrival_times_min = np.concatenate(arrival_times_min_list)
        arrival_hour_index = np.concatenate(arrival_hour_index_list)
    else:
        arrival_times_min = np.array([], dtype=float)
        arrival_hour_index = np.array([], dtype=int)

    return DemandScenarioData(
        config=config,
        horas=horas,
        lambdas=lambdas,
        clientes_por_hora=clientes_por_hora,
        arrival_times_min=arrival_times_min,
        arrival_hour_index=arrival_hour_index,
    )


# ============================================================
# Definição dos 4 cenários (missões)
# ============================================================

# Convenção de time “normal”:
#   Antes da tempestade / Promoção / Black Friday:
#       5 juniores, 3 plenos, 2 seniores
#   Black Swan:
#       5 juniores, 2 plenos, 2 seniores (1 pleno a menos)

CENARIO_1 = DemandScenarioConfig(
    nome="Antes da tempestade",
    descricao=(
        "Dia relativamente calmo, com poucos picos suaves. "
        "Serve como baseline da operação em dia normal."
    ),
    hora_inicio=9,
    hora_fim=18,
    horas_pico=[11, 12],
    media_pico=18.0,
    media_vale=8.0,
    seed_demanda=101,
    base_team_por_tipo={
        "junior": 5,
        "pleno": 3,
        "senior": 2,
    },
    extra_max_total=0,   # sem extras neste cenário
    ticket_medio=120.0,
)

CENARIO_2 = DemandScenarioConfig(
    nome="Dia de campanha de Marketing",
    descricao=(
        "Campanha forte de marketing gera pico acentuado no meio do dia. "
        "Boa para mostrar impacto de ações comerciais na operação."
    ),
    hora_inicio=9,
    hora_fim=18,
    horas_pico=[11, 12, 13],
    media_pico=28.0,
    media_vale=10.0,
    seed_demanda=202,
    # mesma equipe base do cenário 1
    base_team_por_tipo={
        "junior": 5,
        "pleno": 3,
        "senior": 2,
    },
    extra_max_total=0,   # ainda sem extras, só redistribuição
    ticket_medio=180.0,
)

CENARIO_3 = DemandScenarioConfig(
    nome="Black Friday",
    descricao=(
        "Demanda muito alta em boa parte do dia, com picos intensos. "
        "Serve para mostrar o caos se a equipe não acompanhar."
    ),
    hora_inicio=8,
    hora_fim=20,
    horas_pico=[9, 10, 11, 14, 15, 16],
    media_pico=45.0,
    media_vale=18.0,
    seed_demanda=303,
    # mesmo time base, mas com possibilidade de extras
    base_team_por_tipo={
        "junior": 5,
        "pleno": 3,
        "senior": 2,
    },
    extra_max_total=5,   # pode contratar até mais 5 funcionários no total
    ticket_medio=250.0,
)

CENARIO_4 = DemandScenarioConfig(
    nome="Operação Black Swan",
    descricao=(
        "Dia fora da curva: picos inesperados espalhados, "
        "demanda muito volátil. Ótimo para mostrar que "
        "decidir na média não funciona em cenário extremo."
    ),
    hora_inicio=6,
    hora_fim=22,
    horas_pico=[8, 9, 11, 13, 15, 17, 19],
    media_pico=15.0,
    media_vale=5.0,
    seed_demanda=404,
    # equipe normal, porém com 1 pleno a menos
    base_team_por_tipo={
        "junior": 5,
        "pleno": 2,
        "senior": 2,
    },
    extra_max_total=0,
    ticket_medio=300.0,
)

CENARIOS: Dict[str, DemandScenarioConfig] = {
    "antes_tempestade": CENARIO_1,
    "campanha_mkt": CENARIO_2,
    "black_friday": CENARIO_3,
    "black_swan": CENARIO_4,
}


# ============================================================
# Helper conveniente para o main
# ============================================================

def carregar_cenario(chave: str) -> DemandScenarioData:
    if chave not in CENARIOS:
        raise KeyError(
            f"Cenário '{chave}' não encontrado. Opções: {list(CENARIOS.keys())}"
        )
    config = CENARIOS[chave]
    return gerar_dados_cenario(config)
