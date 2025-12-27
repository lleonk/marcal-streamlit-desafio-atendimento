# FO_v1.py
"""
Função Objetivo v1 (versão com chegadas contínuas) para o problema
da Escala Inteligente da Equipe.

Decisão:
- O usuário (ou a metaheurística) escolhe TURNOS:
    - quantos funcionários de cada proficiência (junior, pleno, senior)
    - e o horário de início de cada grupo (turnos fixos de 6 horas).

Objetivo:
- Minimizar o custo total:
    custo_funcionarios
  + custo_penalidade_por_clientes_perdidos
  + (opcional) penalização por uso de funcionários acima do limite
    permitido pelo cenário (time base + extras).

Restrições (via penalização):
- Tempo de espera máximo desejado: 10 minutos.
  - Cliente com espera > 10 min é considerado "perdido" e gera penalização.
- Todos os clientes deveriam ser atendidos até o final do horário de funcionamento.
  - Se não forem (não há mais capacidade no dia), são considerados "não atendidos"
    e geram penalização.
- (Opcional) Limite total de funcionários:
  - Se for informado base_team_por_tipo + extra_max_total, qualquer escala
    que use mais funcionários do que esse limite recebe uma penalização forte.

Penalização:
- Cada cliente "perdido" (atraso > 10 minutos OU não atendido ao final do dia)
  gera um custo proporcional ao ticket médio (penalidade_por_cliente).
- Cada funcionário acima do limite total permitido gera uma penalização adicional
  fator_penal_func_extra * penalidade_por_cliente.

Simulação:
- Usa chegadas contínuas (em minutos) e capacidade distribuída ao longo de cada hora
  (serviços "espalhados" uniformemente dentro da hora).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


# ============================================================
# CONSTANTES DO MODELO (hardcoded)
# ============================================================

DURACAO_TURNO_HORAS = 6

# Produtividade de cada tipo (clientes atendidos por hora por funcionário)
PRODUTIVIDADE_TIPOS: Dict[str, float] = {
    "junior": 2.0,
    "pleno": 5.0,
    "senior": 10.0,
}

# Custo por hora de cada tipo (R$/hora)
CUSTO_HORA_TIPOS: Dict[str, float] = {
    "junior": 30.0,
    "pleno": 50.0,
    "senior": 70.0,
}

# Interpretação: ticket médio perdido por cliente não atendido / atrasado
PENALIDADE_POR_CLIENTE = 1000.0  # valor default, sobrescrito pelo cenário na prática

# Limite de espera "ideal" em minutos
LIMITE_ESPERA_MIN = 10.0
LIMITE_ESPERA_HORAS = LIMITE_ESPERA_MIN / 60.0


# ============================================================
# DATACLASS DE RESULTADO
# ============================================================

@dataclass
class FOResultado:
    # Custos
    valor_objetivo: float
    custo_funcionarios: float
    custo_clientes_perdidos: float

    # Contagens de "perda"
    num_clientes_atrasados: int
    num_clientes_nao_atendidos_final: int

    # Métricas para análise/visualização
    horas: np.ndarray
    clientes_por_hora: np.ndarray
    capacidade_por_hora: np.ndarray
    fila_inicio_hora: np.ndarray
    fila_fim_hora: np.ndarray
    clientes_atendidos_por_hora: np.ndarray
    tempo_medio_espera_horas: np.ndarray
    tempo_max_espera_horas: np.ndarray
    backlog_acumulado_por_hora: np.ndarray

    # Debug/auxiliar
    escala_por_hora: Dict[str, np.ndarray]

    # NOVO: informação sobre uso de funcionários
    total_func_por_tipo: Dict[str, int]
    total_funcionarios: int
    num_funcionarios_excesso: int
    custo_penal_func_extra: float


# ============================================================
# TURNOS (decisão) → ESCALA POR HORA (engine)
# ============================================================

def turnos_para_escala_por_hora(
    horas: np.ndarray,
    turnos: Dict[str, List[Dict[str, int]]],
    duracao_turno_horas: int = DURACAO_TURNO_HORAS,
) -> Dict[str, np.ndarray]:
    """
    Converte a decisão em termos de TURNOS em uma escala por hora.
    """
    horas = np.asarray(horas)
    T = len(horas)
    escala_por_hora: Dict[str, np.ndarray] = {}

    for tipo, lista_turnos in turnos.items():
        escala = np.zeros(T, dtype=int)
        for turno in lista_turnos:
            inicio = int(turno["inicio"])
            qtd = int(turno["quantidade"])
            h_ini = inicio
            h_fim = inicio + duracao_turno_horas

            # marcamos de h_ini (inclusive) até h_fim (exclusivo), limitado ao range de horas
            mask = (horas >= h_ini) & (horas < h_fim)
            escala[mask] += qtd

        escala_por_hora[tipo] = escala

    # Garante que todos os tipos conhecidos existam no dict
    for tipo in PRODUTIVIDADE_TIPOS.keys():
        if tipo not in escala_por_hora:
            escala_por_hora[tipo] = np.zeros(T, dtype=int)

    return escala_por_hora


# ============================================================
# SIMULAÇÃO EVENTO-A-EVENTO (chegadas contínuas)
# ============================================================

def simular_fila_eventos(
    horas: np.ndarray,
    clientes_por_hora: np.ndarray,
    arrival_times_min: np.ndarray,
    arrival_hour_index: np.ndarray,
    escala_por_hora: Dict[str, np.ndarray],
    produtividade_tipos: Dict[str, float] = PRODUTIVIDADE_TIPOS,
    custo_hora_tipos: Dict[str, float] = CUSTO_HORA_TIPOS,
    limite_espera_horas: float = LIMITE_ESPERA_HORAS,
) -> Dict[str, np.ndarray | float | int]:
    """
    Simula a operação cliente a cliente, usando chegadas contínuas e
    capacidade agregada por hora distribuída ao longo do tempo.
    """

    horas = np.asarray(horas)
    clientes_por_hora = np.asarray(clientes_por_hora, dtype=int)
    arrival_times_min = np.asarray(arrival_times_min, dtype=float)
    arrival_hour_index = np.asarray(arrival_hour_index, dtype=int)

    T = len(horas)
    N = arrival_times_min.shape[0]

    if clientes_por_hora.shape[0] != T:
        raise ValueError("clientes_por_hora deve ter o mesmo tamanho de horas.")
    if arrival_hour_index.shape[0] != N:
        raise ValueError("arrival_hour_index deve ter o mesmo tamanho de arrival_times_min.")

    # --------------------------------------------------------
    # 1) Capacidade por hora e slots de serviço (em minutos)
    # --------------------------------------------------------
    capacidade_por_hora = np.zeros(T, dtype=int)
    for tipo, prod in produtividade_tipos.items():
        escala_tipo = np.asarray(escala_por_hora.get(tipo, np.zeros(T, dtype=int)), dtype=int)
        if escala_tipo.shape[0] != T:
            raise ValueError(f"Escala do tipo '{tipo}' deve ter o mesmo tamanho de horas.")
        capacidade_por_hora += np.round(escala_tipo * prod).astype(int)

    # Custo da folha
    custo_folha_por_hora = np.zeros(T, dtype=float)
    for tipo, custo in custo_hora_tipos.items():
        escala_tipo = np.asarray(escala_por_hora.get(tipo, np.zeros(T, dtype=int)), dtype=int)
        custo_folha_por_hora += escala_tipo * float(custo)
    custo_total_folha = float(custo_folha_por_hora.sum())

    # Geração dos "slots de serviço" (momentos em que um atendimento pode terminar)
    service_times_min_list: List[np.ndarray] = []
    for t in range(T):
        n_serv = int(capacidade_por_hora[t])
        if n_serv <= 0:
            continue

        # Hora t: intervalo [t*60, (t+1)*60)
        inicio_min = t * 60.0
        fim_min = inicio_min + 60.0

        times_min = np.linspace(inicio_min, fim_min, n_serv, endpoint=False)
        service_times_min_list.append(times_min)

    if service_times_min_list:
        service_times_min = np.concatenate(service_times_min_list)
    else:
        service_times_min = np.array([], dtype=float)

    M = service_times_min.shape[0]

    # --------------------------------------------------------
    # 2) Atribuição cliente -> slot de serviço (FIFO)
    # --------------------------------------------------------
    service_times_client_min = np.full(N, np.nan, dtype=float)
    j = 0  # índice de slots de serviço

    for i in range(N):
        a = arrival_times_min[i]

        # Avança slots que já passaram antes do cliente chegar (slots ociosos)
        while j < M and service_times_min[j] < a:
            j += 1

        if j == M:
            # Acabou a capacidade do dia; cliente não será atendido
            continue

        service_times_client_min[i] = service_times_min[j]
        j += 1

    served_mask = ~np.isnan(service_times_client_min)
    num_atendidos = int(served_mask.sum())
    num_nao_atendidos = int((~served_mask).sum())

    # --------------------------------------------------------
    # 3) Cálculo dos tempos de espera (por cliente)
    # --------------------------------------------------------
    espera_min = np.zeros(N, dtype=float)
    espera_min[served_mask] = service_times_client_min[served_mask] - arrival_times_min[served_mask]
    espera_horas = espera_min / 60.0

    num_clientes_atrasados = int((espera_horas > limite_espera_horas).sum())

    # --------------------------------------------------------
    # 4) Agregação por hora de chegada e por hora de atendimento
    # --------------------------------------------------------
    tempo_medio_espera_horas = np.zeros(T, dtype=float)
    tempo_max_espera_horas = np.zeros(T, dtype=float)

    for t in range(T):
        mask_t = (arrival_hour_index == t) & served_mask
        if not np.any(mask_t):
            tempo_medio_espera_horas[t] = 0.0
            tempo_max_espera_horas[t] = 0.0
        else:
            arr_esp = espera_horas[mask_t]
            tempo_medio_espera_horas[t] = arr_esp.mean()
            tempo_max_espera_horas[t] = arr_esp.max()

    clientes_atendidos_por_hora = np.zeros(T, dtype=int)
    if num_atendidos > 0:
        service_hour_index = np.floor(service_times_client_min[served_mask] / 60.0).astype(int)
        for idx in service_hour_index:
            if 0 <= idx < T:
                clientes_atendidos_por_hora[idx] += 1

    # --------------------------------------------------------
    # 5) Fila no início e no fim de cada hora (snapshot)
    # --------------------------------------------------------
    fila_inicio_hora = np.zeros(T, dtype=int)
    fila_fim_hora = np.zeros(T, dtype=int)

    for t in range(T):
        end_min = (t + 1) * 60.0
        chegou_ate_fim = arrival_times_min < end_min
        nao_atendidos_ate_fim = (~served_mask) | (service_times_client_min >= end_min)
        fila_fim_hora[t] = int(np.sum(chegou_ate_fim & nao_atendidos_ate_fim))

    fila_inicio_hora[0] = 0
    for t in range(1, T):
        fila_inicio_hora[t] = fila_fim_hora[t - 1]

    backlog_acumulado_por_hora = np.cumsum(fila_fim_hora)

    return {
        "capacidade_por_hora": capacidade_por_hora,
        "fila_inicio_hora": fila_inicio_hora,
        "fila_fim_hora": fila_fim_hora,
        "clientes_atendidos_por_hora": clientes_atendidos_por_hora,
        "tempo_medio_espera_horas": tempo_medio_espera_horas,
        "tempo_max_espera_horas": tempo_max_espera_horas,
        "backlog_acumulado_por_hora": backlog_acumulado_por_hora,
        "clientes_nao_atendidos_apos_ultimo_slot": num_nao_atendidos,
        "custo_total_folha": custo_total_folha,
        "num_clientes_atrasados": num_clientes_atrasados,
    }


# ============================================================
# FUNÇÃO OBJETIVO (usa a simulação evento-a-evento)
# ============================================================

def avaliar_turnos(
    horas: np.ndarray,
    clientes_por_hora: np.ndarray,
    arrival_times_min: np.ndarray,
    arrival_hour_index: np.ndarray,
    turnos: Dict[str, List[Dict[str, int]]],
    duracao_turno_horas: int = DURACAO_TURNO_HORAS,
    produtividade_tipos: Dict[str, float] = PRODUTIVIDADE_TIPOS,
    custo_hora_tipos: Dict[str, float] = CUSTO_HORA_TIPOS,
    penalidade_por_cliente: float = PENALIDADE_POR_CLIENTE,
    limite_espera_horas: float = LIMITE_ESPERA_HORAS,
    # NOVO: limite total de funcionários
    base_team_por_tipo: Dict[str, int] | None = None,
    extra_max_total: int | None = None,
    fator_penal_func_extra: float = 10.0,
) -> FOResultado:
    """
    Avalia uma decisão de TURNOS para um dado cenário de demanda.

    Se base_team_por_tipo e extra_max_total forem fornecidos, qualquer solução
    que use mais funcionários do que (sum(base_team_por_tipo) + extra_max_total)
    recebe uma penalização forte multiplicada por fator_penal_func_extra.
    """

    # 1) Converte turnos em escala por hora
    escala_por_hora = turnos_para_escala_por_hora(
        horas=horas,
        turnos=turnos,
        duracao_turno_horas=duracao_turno_horas,
    )

    # 2) Simula fila evento-a-evento
    sim = simular_fila_eventos(
        horas=horas,
        clientes_por_hora=clientes_por_hora,
        arrival_times_min=arrival_times_min,
        arrival_hour_index=arrival_hour_index,
        escala_por_hora=escala_por_hora,
        produtividade_tipos=produtividade_tipos,
        custo_hora_tipos=custo_hora_tipos,
        limite_espera_horas=limite_espera_horas,
    )

    capacidade_por_hora = sim["capacidade_por_hora"]
    fila_inicio_hora = sim["fila_inicio_hora"]
    fila_fim_hora = sim["fila_fim_hora"]
    clientes_atendidos_por_hora = sim["clientes_atendidos_por_hora"]
    tempo_medio_espera_horas = sim["tempo_medio_espera_horas"]
    tempo_max_espera_horas = sim["tempo_max_espera_horas"]
    backlog_acumulado_por_hora = sim["backlog_acumulado_por_hora"]
    clientes_nao_atendidos_apos_ultimo_slot = int(sim["clientes_nao_atendidos_apos_ultimo_slot"])
    custo_total_folha = float(sim["custo_total_folha"])
    num_clientes_atrasados = int(sim["num_clientes_atrasados"])

    # 3) Cálculo do custo de clientes perdidos
    num_clientes_perdidos_total = clientes_nao_atendidos_apos_ultimo_slot + num_clientes_atrasados
    custo_clientes_perdidos = penalidade_por_cliente * float(num_clientes_perdidos_total)

    # 4) Cálculo de número total de funcionários usados na escala
    total_func_por_tipo: Dict[str, int] = {t: 0 for t in PRODUTIVIDADE_TIPOS.keys()}
    for tipo, lista_turnos in turnos.items():
        total = 0
        for turno in lista_turnos:
            total += int(turno.get("quantidade", 0))
        total_func_por_tipo[tipo] = total_func_por_tipo.get(tipo, 0) + total

    total_funcionarios = sum(total_func_por_tipo.values())

    # 5) Penalização por exceder o limite de funcionários
    num_funcionarios_excesso = 0
    custo_penal_func_extra = 0.0
    if base_team_por_tipo is not None and extra_max_total is not None:
        base_total = sum(base_team_por_tipo.get(t, 0) for t in total_func_por_tipo.keys())
        max_total = base_total + int(extra_max_total)
        num_funcionarios_excesso = max(0, total_funcionarios - max_total)
        if num_funcionarios_excesso > 0:
            # penalização forte proporcional ao ticket médio
            custo_penal_func_extra = (
                float(num_funcionarios_excesso) * penalidade_por_cliente * float(fator_penal_func_extra)
            )

    # 6) Função objetivo = custo funcionários + penalidades
    valor_objetivo = custo_total_folha + custo_clientes_perdidos + custo_penal_func_extra

    return FOResultado(
        valor_objetivo=valor_objetivo,
        custo_funcionarios=custo_total_folha,
        custo_clientes_perdidos=custo_clientes_perdidos,
        num_clientes_atrasados=num_clientes_atrasados,
        num_clientes_nao_atendidos_final=clientes_nao_atendidos_apos_ultimo_slot,
        horas=np.asarray(horas),
        clientes_por_hora=np.asarray(clientes_por_hora),
        capacidade_por_hora=capacidade_por_hora,
        fila_inicio_hora=fila_inicio_hora,
        fila_fim_hora=fila_fim_hora,
        clientes_atendidos_por_hora=clientes_atendidos_por_hora,
        tempo_medio_espera_horas=tempo_medio_espera_horas,
        tempo_max_espera_horas=tempo_max_espera_horas,
        backlog_acumulado_por_hora=backlog_acumulado_por_hora,
        escala_por_hora=escala_por_hora,
        total_func_por_tipo=total_func_por_tipo,
        total_funcionarios=total_funcionarios,
        num_funcionarios_excesso=num_funcionarios_excesso,
        custo_penal_func_extra=custo_penal_func_extra,
    )
