# painel_utils.py
"""
Utilitários de apoio ao painel interativo (Streamlit).

Aqui ficam apenas funções "de negócio" e de visualização em Matplotlib,
sem nenhuma dependência direta de Streamlit. O app (`app_painel.py`)
apenas importa e usa essas funções.

Principais funções:

- obter_dados_cenario(scenario_key)
- avaliar_escala_usuario(scenario_key, turnos_usuario)
- obter_escala_otima(scenario_key, metodo_preferido="brkga")
- construir_resumo(dados_cenario, resultado_fo)
- comparar_resumos(resumo_user, resumo_otimo)
- construir_horas_inicio_validas(...)
- turnos_para_barras_individuais(...)
- criar_figura_gantt_triplo(...)
- criar_figura_distribuicao_demanda(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from parametros_v3 import carregar_cenario
from fo_v2 import (
    avaliar_turnos,
    DURACAO_TURNO_HORAS,
    PRODUTIVIDADE_TIPOS,
    FOResultado,
)
from otimos_io import carregar_turnos_otimos


# Lista padrão de tipos de funcionários usada em todo o projeto
TIPOS_DEFAULT: List[str] = list(PRODUTIVIDADE_TIPOS.keys())


# =====================================================================
# Estruturas de resumo numérico
# =====================================================================

@dataclass
class ResumoEscala:
    """
    Resumo numérico de uma escala (para mostrar no painel / tabelas).

    Todos os valores são derivados de um FOResultado + dados do cenário.
    """

    scenario_key: str
    label: str

    num_clientes_total: int
    atendidos: int
    perdidos: int

    ticket_medio: float
    faturamento: float

    custo_funcionarios: float
    caixa_apos_folha: float

    valor_objetivo: float

    num_clientes_atrasados: int
    num_clientes_nao_atendidos_final: int


# =====================================================================
# Helpers de cenário e avaliação
# =====================================================================

def obter_dados_cenario(scenario_key: str):
    """
    Wrapper simples para carregar_dados_cenario, só para deixar o painel
    mais legível.
    """
    return carregar_cenario(scenario_key)


def _construir_resumo_interno(
    scenario_key: str,
    label: str,
    dados_cenario,
    resultado: FOResultado,
) -> ResumoEscala:
    """
    Constrói um ResumoEscala a partir dos dados do cenário e de um FOResultado.
    """
    ticket = float(dados_cenario.config.ticket_medio)
    num_clientes_total = len(dados_cenario.arrival_times_min)

    perdidos = (
        int(resultado.num_clientes_atrasados)
        + int(resultado.num_clientes_nao_atendidos_final)
    )
    atendidos = max(0, num_clientes_total - perdidos)
    faturamento = atendidos * ticket
    caixa_apos_folha = faturamento - float(resultado.custo_funcionarios)

    return ResumoEscala(
        scenario_key=scenario_key,
        label=label,
        num_clientes_total=num_clientes_total,
        atendidos=atendidos,
        perdidos=perdidos,
        ticket_medio=ticket,
        faturamento=faturamento,
        custo_funcionarios=float(resultado.custo_funcionarios),
        caixa_apos_folha=caixa_apos_folha,
        valor_objetivo=float(resultado.valor_objetivo),
        num_clientes_atrasados=int(resultado.num_clientes_atrasados),
        num_clientes_nao_atendidos_final=int(
            resultado.num_clientes_nao_atendidos_final
        ),
    )


def avaliar_escala_usuario(
    scenario_key: str,
    turnos_usuario: Dict[str, List[Dict[str, int]]],
    tipos: Optional[List[str]] = None,
) -> Tuple[FOResultado, ResumoEscala]:
    """
    Avalia a escala definida pelo usuário para um dado cenário.

    Parameters
    ----------
    scenario_key : str
        Chave do cenário (antes_tempestade, campanha_mkt, black_friday, black_swan).
    turnos_usuario : dict
        Estrutura de turnos no formato já usado no projeto:
        {
            "junior": [{"inicio": 9, "quantidade": 2}, ...],
            "pleno":  [...],
            "senior": [...],
        }
    tipos : list[str], optional
        Lista de tipos. Se None, usa TIPOS_DEFAULT.

    Returns
    -------
    resultado_fo : FOResultado
    resumo : ResumoEscala
    """
    if tipos is None:
        tipos = TIPOS_DEFAULT

    dados = obter_dados_cenario(scenario_key)
    horas = dados.horas

    resultado = avaliar_turnos(
        horas=horas,
        clientes_por_hora=dados.clientes_por_hora,
        arrival_times_min=dados.arrival_times_min,
        arrival_hour_index=dados.arrival_hour_index,
        turnos=turnos_usuario,
        penalidade_por_cliente=dados.config.ticket_medio,
        base_team_por_tipo=dados.config.base_team_por_tipo,
        extra_max_total=dados.config.extra_max_total,
    )

    resumo = _construir_resumo_interno(
        scenario_key=scenario_key,
        label="Usuário",
        dados_cenario=dados,
        resultado=resultado,
    )

    return resultado, resumo


def obter_escala_otima(
    scenario_key: str,
    metodo_preferido: str = "brkga",
    tipos: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, int]]], FOResultado, ResumoEscala]:
    """
    Carrega a escala ótima salva em JSON (via otimos_io) e reavalia
    com a FO atual, devolvendo:

    - turnos_otimos (dict)
    - resultado_fo (FOResultado)
    - resumo (ResumoEscala)

    Se não houver solução salva para o cenário / método, levanta ValueError.
    """
    if tipos is None:
        tipos = TIPOS_DEFAULT

    dados_json = carregar_turnos_otimos(
        cenario_key=scenario_key,
        metodo_preferido=metodo_preferido,
    )
    if dados_json is None:
        raise ValueError(
            f"Não encontrei solução ótima salva para o cenário '{scenario_key}'."
        )

    turnos_otimos = dados_json["turnos"]

    dados = obter_dados_cenario(scenario_key)

    resultado = avaliar_turnos(
        horas=dados.horas,
        clientes_por_hora=dados.clientes_por_hora,
        arrival_times_min=dados.arrival_times_min,
        arrival_hour_index=dados.arrival_hour_index,
        turnos=turnos_otimos,
        penalidade_por_cliente=dados.config.ticket_medio,
        base_team_por_tipo=dados.config.base_team_por_tipo,
        extra_max_total=dados.config.extra_max_total,
    )

    label = f"Ótimo ({metodo_preferido.upper()})"
    resumo = _construir_resumo_interno(
        scenario_key=scenario_key,
        label=label,
        dados_cenario=dados,
        resultado=resultado,
    )

    return turnos_otimos, resultado, resumo


def comparar_resumos(
    resumo_user: ResumoEscala,
    resumo_otimo: ResumoEscala,
) -> Dict[str, Dict[str, float]]:
    """
    Gera um dicionário simples com comparação entre duas escalas
    (normalmente: usuário vs ótimo).

    Retorna algo como:
    {
        "atendidos": {"user": ..., "otimo": ..., "ganho": ...},
        "faturamento": {...},
        "custo_funcionarios": {...},
        "caixa": {...},
        ...
    }
    """
    comp: Dict[str, Dict[str, float]] = {}

    def bloco(user_val, opt_val):
        return {
            "user": float(user_val),
            "otimo": float(opt_val),
            "ganho": float(opt_val - user_val),
        }

    comp["atendidos"] = bloco(resumo_user.atendidos, resumo_otimo.atendidos)
    comp["perdidos"] = bloco(resumo_user.perdidos, resumo_otimo.perdidos)
    comp["faturamento"] = bloco(resumo_user.faturamento, resumo_otimo.faturamento)
    comp["custo_funcionarios"] = bloco(
        resumo_user.custo_funcionarios, resumo_otimo.custo_funcionarios
    )
    comp["caixa"] = bloco(resumo_user.caixa_apos_folha, resumo_otimo.caixa_apos_folha)
    comp["valor_objetivo"] = bloco(
        resumo_user.valor_objetivo, resumo_otimo.valor_objetivo
    )

    return comp


# =====================================================================
# Helpers de turnos e Gantt (reaproveitados do main)
# =====================================================================

def construir_horas_inicio_validas(
    horas: np.ndarray,
    duracao_turno_horas: int = DURACAO_TURNO_HORAS,
) -> List[int]:
    """
    Constrói a lista de horas possíveis de início de turno, dado o
    vetor de horas inteiras e a duração fixa do turno.
    """
    hora_inicio_dia = int(horas[0])
    hora_fim_dia = int(horas[-1]) + 1  # exclusivo
    return list(range(hora_inicio_dia, hora_fim_dia - duracao_turno_horas + 1))


def turnos_para_barras_individuais(
    turnos: Dict[str, List[Dict[str, int]]],
    duracao_turno_horas: int = DURACAO_TURNO_HORAS,
    tipos: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """
    Converte o dicionário de turnos em uma lista de "barras individuais"
    para plotar um Gantt em Matplotlib.

    Cada item da lista é:
    {
        "tipo": "junior" | "pleno" | "senior",
        "label": "Junior #1",
        "inicio": 9,
        "fim": 15,
    }
    """
    if tipos is None:
        tipos = TIPOS_DEFAULT

    barras: List[Dict[str, object]] = []
    for tipo in tipos:
        lista_turnos = turnos.get(tipo, [])
        contador = 0
        for turno in lista_turnos:
            inicio = int(turno["inicio"])
            fim = inicio + int(duracao_turno_horas)
            quantidade = int(turno["quantidade"])
            for _ in range(quantidade):
                contador += 1
                barras.append(
                    {
                        "tipo": tipo,
                        "label": f"{tipo.capitalize()} #{contador}",
                        "inicio": inicio,
                        "fim": fim,
                    }
                )
    return barras


def _plot_gantt_ax(
    ax: plt.Axes,
    barras: List[Dict[str, object]],
    horas: np.ndarray,
    titulo: str,
    cores_tipos: Optional[Dict[str, str]] = None,
    show_xlabel: bool = True,
):
    """
    Desenha um Gantt simples em um Axes.

    show_xlabel: se True, escreve 'Hora' no eixo X; se False, não.
    """
    if cores_tipos is None:
        cores_tipos = {
            "junior": "#4C72B0",
            "pleno": "#55A868",
            "senior": "#C44E52",
        }

    if not barras:
        ax.set_title(titulo + " (sem funcionários)")
        return

    y_positions: List[int] = []
    y_labels: List[str] = []

    for idx, b in enumerate(barras):
        y = idx
        x_ini = int(b["inicio"])
        x_fim = int(b["fim"])
        largura = x_fim - x_ini

        cor = cores_tipos.get(str(b["tipo"]), "#999999")
        ax.barh(
            y,
            largura,
            left=x_ini,
            height=0.8,
            color=cor,
            edgecolor="black",
            alpha=0.9,
        )

        y_positions.append(y)
        y_labels.append(str(b["label"]))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    if show_xlabel:
        ax.set_xlabel("Hora")

    ax.set_title(titulo)

    hora_min = int(horas[0])
    hora_max = int(horas[-1]) + 1
    ax.set_xlim(hora_min, hora_max)
    ax.set_xticks(range(hora_min, hora_max + 1))




# =====================================================================
# Curva de distribuição de demanda (Poisson suavizada)
# =====================================================================

def criar_figura_distribuicao_demanda(
    scenario_key: str,
    num_std: float = 3.0,
) -> plt.Figure:
    """
    Cria o gráfico com as duas curvas de Poisson (pico e vale) suavizadas
    para o cenário escolhido.

    Útil para explicar, no painel, como a demanda foi gerada.

    Parameters
    ----------
    scenario_key : str
        Chave do cenário (antes_tempestade, campanha_mkt, ...).
    num_std : float
        Número de desvios-padrão ao redor de cada lambda para definir o
        intervalo de X.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    dados = obter_dados_cenario(scenario_key)
    cfg = dados.config

    lam_pico = float(cfg.media_pico)
    lam_vale = float(cfg.media_vale)

    # Intervalo "largo" cobrindo ambas as distribuições
    lam_min = min(lam_pico, lam_vale)
    lam_max = max(lam_pico, lam_vale)
    sigma_aprox = np.sqrt(lam_max)

    x_min = max(0, int(np.floor(lam_min - num_std * sigma_aprox)))
    x_max = int(np.ceil(lam_max + num_std * sigma_aprox))

    x = np.linspace(x_min, x_max, 400)

    # Aproximação pela Normal (boa para lambdas mais altos) para suavizar
    def normal_pdf(x_arr, mu, sigma2):
        return (
            1.0
            / np.sqrt(2.0 * np.pi * sigma2)
            * np.exp(-(x_arr - mu) ** 2 / (2.0 * sigma2))
        )

    y_pico = normal_pdf(x, lam_pico, lam_pico)
    y_vale = normal_pdf(x, lam_vale, lam_vale)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_pico, label=f"Horas de pico (λ={lam_pico:.1f})")
    ax.plot(x, y_vale, label=f"Horas comuns (λ={lam_vale:.1f})")

    ax.set_xlabel("Número de clientes em 1 hora")
    ax.set_ylabel("Probabilidade (aprox.)")
    ax.set_title(f"Distribuição de demanda por hora - {cfg.nome}")
    ax.legend()
    fig.tight_layout()


    return fig


def criar_figura_capacidade_vs_demanda(
    dados,
    res_user,
    res_sa,
    res_brkga,
    mostrar_ias: bool = True,
):
    """
    Gráfico 'Chegadas x Capacidade de atendimento' para o cenário escolhido.

    - Barras: chegadas por hora (demanda)
    - Linhas:
        * Capacidade (Sua escala)               -> sempre aparece
        * Capacidade (IA-1)  -> SA              -> aparece só se mostrar_ias=True
        * Capacidade (IA-2)  -> BRKGA           -> aparece só se mostrar_ias=True
    """

    horas = np.array(dados.horas)
    labels = [f"{int(h):02d}:00" for h in horas]

    # 1) Chegadas (demanda por hora)
    def _extrair_chegadas():
        candidatos = [
            ("chegadas_por_hora", dados),
            ("demanda_por_hora", dados),
            ("clientes_por_hora", dados),
            ("chegadas", dados),
            ("demanda", dados),
            # fallback: às vezes deixamos isso no resultado da IA
            ("chegadas_por_hora", res_sa),
            ("demanda_por_hora", res_sa),
        ]
        for attr, obj in candidatos:
            if hasattr(obj, attr):
                arr = np.array(getattr(obj, attr))
                if arr.size == horas.size:
                    return arr
        # fallback: tudo zero (aparece só a capacidade)
        return np.zeros_like(horas, dtype=float)

    chegadas = _extrair_chegadas()

    # 2) Capacidades hora a hora
    def _extrair_capacidade(result_obj):
        for attr in ["capacidade_por_hora", "cap_por_hora", "capacidade"]:
            if hasattr(result_obj, attr):
                arr = np.array(getattr(result_obj, attr))
                if arr.size == horas.size:
                    return arr
        return np.zeros_like(horas, dtype=float)

    cap_user = _extrair_capacidade(res_user)
    cap_sa = _extrair_capacidade(res_sa)
    cap_brkga = _extrair_capacidade(res_brkga)

    # 3) Monta o gráfico
    fig, ax = plt.subplots(figsize=(10, 4))

    # Barras de chegadas
    ax.bar(labels, chegadas, alpha=0.3, label="Chegadas", edgecolor="none")

    # Linha da SUA escala (sempre aparece)
    ax.plot(labels, cap_user, marker="o", linestyle="-", label="Capacidade (Sua escala)")

    # Linhas das IAs: só se o usuário pediu comparação
    if mostrar_ias:
        ax.plot(labels, cap_sa, marker="s", linestyle="-.", label="Capacidade (IA-1)")
        ax.plot(labels, cap_brkga, marker="^", linestyle=":", label="Capacidade (IA-2)")
        titulo = "Chegadas x Capacidade de atendimento (com comparação às IAs)"
    else:
        titulo = "Chegadas x Capacidade de atendimento (sua escala)"

    ax.set_title(titulo)
    ax.set_xlabel("Horário")
    ax.set_ylabel("Clientes por hora / capacidade")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig


# mesmas cores usadas no app
COR_TIPO = {
    "junior": "#1f77b4",   # azul
    "pleno":  "#2ca02c",   # verde
    "senior": "#d62728",   # vermelho
}

# ordem vertical desejada: seniores em cima, juniores embaixo
ORDEM_TIPO = {"senior": 0, "pleno": 1, "junior": 2}

def criar_figura_gantt_triplo(
    horas,
    turnos_user: Dict[str, List[Dict]],
    turnos_ia1: Dict[str, List[Dict]],
    turnos_ia2: Dict[str, List[Dict]],
    tipos,
    titulo_user: str,
    titulo_ia1: str,
    titulo_ia2: str,
    funcionarios_user_ordenados: List[Dict],
):
    """
    Gera um Gantt triplo:

    - Subplot 1: escala do usuário (uma linha por funcionário, na ordem já ordenada).
    - Subplot 2: escala da IA-1 (uma linha por "funcionário genérico").
    - Subplot 3: escala da IA-2 (idem).

    Em todos os casos, a ordem vertical é:
        seniores em cima, plenos no meio, juniores embaixo.
    """

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)

    # ---------------------------------------------------------
    # 1) Escala do usuário (usa diretamente funcionarios_user_ordenados)
    # ---------------------------------------------------------
    ax_user = axes[0]
    for idx, f in enumerate(funcionarios_user_ordenados):
        tipo = f["tipo"]
        inicio = f["inicio"]
        ax_user.barh(
            idx,
            DURACAO_TURNO_HORAS,
            left=inicio,
            height=0.6,
            color=COR_TIPO.get(tipo, "gray"),
            edgecolor="black",
        )

    # sem rótulos no eixo Y (mais limpo)
    ax_user.set_yticks([])
    ax_user.invert_yaxis()
    ax_user.set_title(titulo_user)


    # ---------------------------------------------------------
    # helper para montar linhas das IAs na mesma ordem de tipo
    # ---------------------------------------------------------
    def montar_linhas_ia(turnos: Dict[str, List[Dict]]) -> List[Dict]:
        linhas = []
        for tipo in sorted(tipos, key=lambda t: ORDEM_TIPO.get(t, 99)):
            for turno in turnos.get(tipo, []):
                inicio = turno["inicio"]
                qtd = turno["quantidade"]
                for _ in range(qtd):
                    linhas.append({"tipo": tipo, "inicio": inicio})
        # garante que, dentro do tipo, ordena por hora de início
        linhas.sort(key=lambda f: (ORDEM_TIPO.get(f["tipo"], 99), f["inicio"]))
        return linhas

    # ---------------------------------------------------------
    # 2) Escala IA-1
    # ---------------------------------------------------------
    ax_ia1 = axes[1]
    linhas_ia1 = montar_linhas_ia(turnos_ia1)
    for idx, f in enumerate(linhas_ia1):
        tipo = f["tipo"]
        inicio = f["inicio"]
        ax_ia1.barh(
            idx,
            DURACAO_TURNO_HORAS,
            left=inicio,
            height=0.6,
            color=COR_TIPO.get(tipo, "gray"),
            edgecolor="black",
        )
    ax_ia1.set_yticks([])  # sem rótulos de eixos para não poluir
    ax_ia1.invert_yaxis()
    ax_ia1.set_title(titulo_ia1)

    # ---------------------------------------------------------
    # 3) Escala IA-2
    # ---------------------------------------------------------
    ax_ia2 = axes[2]
    linhas_ia2 = montar_linhas_ia(turnos_ia2)
    for idx, f in enumerate(linhas_ia2):
        tipo = f["tipo"]
        inicio = f["inicio"]
        ax_ia2.barh(
            idx,
            DURACAO_TURNO_HORAS,
            left=inicio,
            height=0.6,
            color=COR_TIPO.get(tipo, "gray"),
            edgecolor="black",
        )
    ax_ia2.set_yticks([])
    ax_ia2.invert_yaxis()
    ax_ia2.set_title(titulo_ia2)

    # ---------------------------------------------------------
    # Configuração comum do eixo x
    # ---------------------------------------------------------
    xmin = min(horas)
    xmax = max(horas) + 1
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(horas)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    ax_ia2.set_xlabel("Horário de trabalho de cada funcionário")

    return fig




def funcionarios_para_barras_ordenadas(
    funcionarios,
    duracao_turno_horas: int = DURACAO_TURNO_HORAS,
):
    """
    Converte a lista de funcionários (como usada no app Streamlit)
    em barras individuais para o Gantt, preservando a ORDEM da lista.

    Cada funcionário vira uma barra:
    {
        "tipo": "junior" | "pleno" | "senior",
        "label": "Junior #1",
        "inicio": 9,
        "fim": 15,
    }
    """
    barras = []
    contador_por_tipo = {}

    for f in funcionarios:
        tipo = f.get("tipo", "junior")
        inicio = int(f.get("inicio", 0))
        fim = inicio + int(duracao_turno_horas)

        contador_por_tipo[tipo] = contador_por_tipo.get(tipo, 0) + 1
        label = f"{tipo.capitalize()} #{contador_por_tipo[tipo]}"

        barras.append(
            {
                "tipo": tipo,
                "label": label,
                "inicio": inicio,
                "fim": fim,
            }
        )

    return barras

