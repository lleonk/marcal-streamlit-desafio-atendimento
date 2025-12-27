# app_painel.py
"""
Painel interativo (Streamlit) para o problema da Escala Inteligente.

Fluxo:
- Usuário escolhe um cenário.
- Carregamos as soluções "ótimas" salvas (JSON) para esse cenário:
    IA-1 = SA, IA-2 = BRKGA.
- Usuário monta/ajusta sua própria escala apenas definindo o horário
  de início de cada funcionário (turno fixo de DURACAO_TURNO_HORAS).
- Avaliamos a escala do usuário e comparamos com as IAs:
    - clientes atendidos, perdidos, atrasados, não atendidos;
    - faturamento;
    - custo com funcionários;
    - caixa após folha;
    - valor da função objetivo.
- Mostramos também:
    - Gráfico Chegadas x Capacidade (sua escala, IA-1, IA-2).
    - Gantt comparando a escala do usuário com IA-1 e IA-2;
    - Curva de distribuição de demanda (pico x vale).
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from fo_v2 import DURACAO_TURNO_HORAS, PRODUTIVIDADE_TIPOS, CUSTO_HORA_TIPOS
from painel_utils import (
    TIPOS_DEFAULT,
    obter_dados_cenario,
    avaliar_escala_usuario,
    obter_escala_otima,
    construir_horas_inicio_validas,
    criar_figura_gantt_triplo,
    criar_figura_distribuicao_demanda,
    criar_figura_capacidade_vs_demanda,
)

TIPOS = TIPOS_DEFAULT

# Cores coerentes com o Gantt:
# Junior -> azul, Pleno -> verde, Senior -> vermelho
COR_TIPO = {
    "junior": "#1f77b4",   # azul
    "pleno": "#2ca02c",    # verde
    "senior": "#d62728",   # vermelho
}

# Ordem para exibição dos sliders:
# seniores em cima, plenos no meio, juniores embaixo
ORDEM_TIPO = {"senior": 0, "pleno": 1, "junior": 2}


def ordenar_funcionarios():
    """Reordena a lista de funcionários no session_state por tipo e id."""
    funcs = st.session_state.get("funcionarios", [])
    funcs_ordenados = sorted(
        funcs,
        key=lambda f: (ORDEM_TIPO.get(f.get("tipo", ""), 99), f.get("id", 0)),
    )
    st.session_state["funcionarios"] = funcs_ordenados


# =====================================================================
# Configuração básica da página
# =====================================================================

st.set_page_config(
    page_title="Desafio da Escala Inteligente",
    layout="wide",
)

# Centraliza o título usando colunas
col_esq, col_centro, col_dir = st.columns([1, 1, 1])
with col_centro:
    st.title("Desafio de atendimento")


# =====================================================================
# Helpers internos do app
# =====================================================================

SCENARIO_LABEL_MAP = {
    "Antes da tempestade": "antes_tempestade",
    "Campanha de marketing": "campanha_mkt",
    "Black Friday": "black_friday",
    "Operação Black Swan": "black_swan",
}
SCENARIO_LABEL_MAP_INV = {v: k for k, v in SCENARIO_LABEL_MAP.items()}


def funcionarios_para_turnos(funcionarios, tipos, horas_inicio_turno):
    """
    Converte a lista de funcionários (cada um com tipo e hora de início)
    para o formato de 'turnos' usado no modelo.

    Retorno:
        {
            "junior": [{"inicio": h, "quantidade": q}, ...],
            "pleno":  [...],
            "senior": [...],
        }
    """
    mapa = {t: {} for t in tipos}
    for f in funcionarios:
        tipo = f["tipo"]
        h = int(f["inicio"])
        if tipo not in mapa:
            mapa[tipo] = {}
        mapa[tipo][h] = mapa[tipo].get(h, 0) + 1

    turnos = {}
    for tipo, por_hora in mapa.items():
        if por_hora:
            turnos[tipo] = [
                {"inicio": h, "quantidade": q}
                for h, q in sorted(por_hora.items())
                if h in horas_inicio_turno
            ]
    return turnos


# =====================================================================
# Barra lateral: escolha de cenário
# =====================================================================

st.sidebar.header("Configurações")

scenario_label = st.sidebar.selectbox(
    "Cenário:",
    list(SCENARIO_LABEL_MAP.keys()),
)
scenario_key = SCENARIO_LABEL_MAP[scenario_label]

# =====================================================================
# Carrega dados do cenário e soluções IA-1 / IA-2
# =====================================================================

dados = obter_dados_cenario(scenario_key)
horas = dados.horas  # ex.: [9, 10, ..., 17]
base_team = dados.config.base_team_por_tipo
extra_max_total = dados.config.extra_max_total

# =====================================================================
# Resumo do cenário na barra lateral
# =====================================================================

st.sidebar.markdown("---")

st.sidebar.markdown(f"_{dados.config.descricao}_")

st.sidebar.markdown("**Resumo do cenário**")
st.sidebar.markdown(
    f"- **Atendimento:** {int(horas[0]):02d}:00 às {int(horas[-1]) + 1:02d}:00  \n"
    f"- **Horas de pico:** {', '.join(str(h) + 'h' for h in dados.config.horas_pico)}  \n"
    f"- **Ticket médio:** R$ {dados.config.ticket_medio:,.2f}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Time base disponível e perfil dos funcionários**")

for t in TIPOS:
    qtd_base = base_team.get(t, 0)
    prod = PRODUTIVIDADE_TIPOS.get(t, 0.0)
    custo = CUSTO_HORA_TIPOS.get(t, 0.0)

    st.sidebar.markdown(
        f"- **{t.capitalize()}s**:  \n"
        f"  atende {prod:.1f} clientes/hora  \n"
        f"  custa R$ {custo:,.2f} por hora.  \n"
        f" {qtd_base} disponíveis"       
    )

st.sidebar.markdown(
    f"**Funcionários extra disponíveis:** {extra_max_total}"
)

st.sidebar.markdown(
    "Obs: Clientes não atendidos em até **10 minutos** vão embora sem comprar."
)

st.sidebar.markdown(
    "Obs: Clientes não atendidos até o final do turno vão embora sem comprar."
)

# Nota sobre como montar a escala
st.sidebar.markdown("---")
st.sidebar.markdown("**Como montar sua escala**")
st.sidebar.markdown(
    f"_Turnos fixos de **{DURACAO_TURNO_HORAS} horas**.  \n"
    "Use o slider de cada linha para escolher o horário de **início** do turno.  \n"
    "O fim é calculado automaticamente._"
)



# =====================================================================
# Construção dos horários válidos e carregamento das escalas otimizadas
# =====================================================================


# Horários válidos de início de turno (já respeitam 'fechamento - duração')
horas_inicio_turno = construir_horas_inicio_validas(
    horas,
    DURACAO_TURNO_HORAS,
)

# Carrega as duas soluções salvas
turnos_sa, res_sa, resumo_sa = obter_escala_otima(
    scenario_key=scenario_key,
    metodo_preferido="sa",
    tipos=TIPOS,
)
turnos_brkga, res_brkga, resumo_brkga = obter_escala_otima(
    scenario_key=scenario_key,
    metodo_preferido="brkga",
    tipos=TIPOS,
)

# =====================================================================
# Distribuição histórica da demanda
# =====================================================================

st.markdown("---")
st.markdown(
    "<h3 style='text-align: center;'>Distribuição histórica da demanda</h3>",
    unsafe_allow_html=True,
)

fig_dist = criar_figura_distribuicao_demanda(scenario_key)
fig_dist.set_size_inches(7, 2)

col_fig, col_texto = st.columns([3, 4])

with col_fig:
    st.pyplot(fig_dist, use_container_width=True)

with col_texto:
    horas_pico_str = ", ".join(f"{h}h" for h in dados.config.horas_pico)
    st.markdown(
        f"""
**Histórico de chegadas de clientes à empresa**

- Sua empresa registra, há bastante tempo, **quantos clientes chegam em cada hora** do dia.  
- Ao olhar esses registros, fica claro que existem **horas comuns** e **horas de pico**.  
- Nas horas comuns, o fluxo de chegada é menor e mais estável.  
  Já nas horas de pico, os clientes chegam em ritmo bem mais intenso.  
- Neste cenário específico, as horas de pico típicas são: **{horas_pico_str}**.  
- As curvas ao lado representam, de forma aproximada, a **probabilidade**
  de observar certo número de clientes chegando em **uma hora**:
  uma curva para as horas comuns e outra para as horas de pico.
        """
    )

# =====================================================================
# 2. Chegadas x Capacidade – storytelling + placeholder do gráfico
# =====================================================================

st.markdown("---")
st.markdown(
    "<h3 style='text-align: center;'>Chegadas x Capacidade de atendimento (sua escala, IA-1 e IA-2)</h3>",
    unsafe_allow_html=True,
)

col_esq, col_texto, col_dir = st.columns([1, 6, 2.6])
with col_texto:
    st.markdown(
        """
Conhecendo o histórico do seu negócio, vamos propor um **desafio simplificado**.

Nessa simplificação, você tem uma vantagem que quase ninguém tem na vida real:
já sabe, de antemão, **quantos clientes chegarão à sua empresa em cada hora do dia**.

O gráfico a seguir mostra essa demanda “revelada”.
A curva representa a **capacidade da sua escala**.

Com essa informação privilegiada, você seria capaz de escalar sua equipe
para aproveitar ao máximo essa oportunidade, evitando filas e ociosidade?

Importante:
- Clientes com mais de 10 minutos de espera vão embora sem comprar.
- Clientes não atendidos ao final do expediente vão embora sem comprar.

Agora é a sua vez...

**Ajuste os sliders, teste cenários e faça o seu melhor!**

Quando estiver pronto, clique em **“Mostrar resultados das IAs”**
e compare seu desempenho com o das duas IAs que convidamos para o mesmo desafio.
        """
    )

# Placeholder do gráfico Chegadas x Capacidade
col_esq, col_centro, col_dir = st.columns([1, 6, 2.6])
with col_centro:
    cap_placeholder = st.empty()

# =====================================================================
# 1. Sliders de início de turno + botões
# =====================================================================

# Reset da lista de funcionários quando muda o cenário
if (
    "scenario_key" not in st.session_state
    or st.session_state["scenario_key"] != scenario_key
):
    st.session_state["scenario_key"] = scenario_key
    st.session_state["funcionarios"] = []
    st.session_state["next_func_id"] = 1
    st.session_state["mostrar_comparacao"] = False

if "mostrar_comparacao" not in st.session_state:
    st.session_state["mostrar_comparacao"] = False

# Horários possíveis de início de turno
min_inicio = horas_inicio_turno[0]        # primeira hora válida de início
max_inicio = horas_inicio_turno[-1]       # última hora válida de início

# Limites do slider de intervalo (início, fim)
slider_min = min_inicio
slider_max = max_inicio + DURACAO_TURNO_HORAS  # ex.: último início 11 → fim 17

# Containers: primeiro o dos sliders, depois o dos botões de adicionar
sliders_container = st.container()
buttons_container = st.container()

# ---------------------------------------------------------
# BOTÕES DE ADIÇÃO (logo abaixo dos sliders, mas logicamente antes)
# ---------------------------------------------------------
with buttons_container:
    # Contagem atual por tipo
    qtd_por_tipo = {t: 0 for t in TIPOS}
    for f in st.session_state["funcionarios"]:
        tipo = f.get("tipo")
        if tipo in qtd_por_tipo:
            qtd_por_tipo[tipo] += 1

    # Extras já usados = tudo que passou do time base, somando todos os tipos
    extras_usados = 0
    for t in TIPOS:
        base_t = base_team.get(t, 0)
        qtd_t = qtd_por_tipo.get(t, 0)
        extras_usados += max(0, qtd_t - base_t)

    extras_restantes = max(0, extra_max_total - extras_usados)

    def capacidade_restante(tipo: str) -> int:
        """Quantos ainda posso adicionar desse tipo, respeitando base + extras globais."""
        base_t = base_team.get(tipo, 0)
        qtd_t = qtd_por_tipo.get(tipo, 0)
        max_para_tipo = base_t + extras_restantes
        return max(0, max_para_tipo - qtd_t)

    restantes_junior = capacidade_restante("junior")
    restantes_pleno = capacidade_restante("pleno")
    restantes_senior = capacidade_restante("senior")

    col_esq, col_b1, col_b2, col_b3, col_dir = st.columns([2, 1, 1, 1, 3])

    # Botão Júnior
    with col_b1:
        label_j = f"➕ Adicionar Júnior ({restantes_junior})"
        if st.button(label_j, disabled=(restantes_junior <= 0), key="btn_add_junior"):
            st.session_state["funcionarios"].append(
                {
                    "id": st.session_state["next_func_id"],
                    "tipo": "junior",
                    "inicio": min_inicio,
                }
            )
            st.session_state["next_func_id"] += 1
            ordenar_funcionarios()
            st.rerun()

    # Botão Pleno
    with col_b2:
        label_p = f"➕ Adicionar Pleno ({restantes_pleno})"
        if st.button(label_p, disabled=(restantes_pleno <= 0), key="btn_add_pleno"):
            st.session_state["funcionarios"].append(
                {
                    "id": st.session_state["next_func_id"],
                    "tipo": "pleno",
                    "inicio": min_inicio,
                }
            )
            st.session_state["next_func_id"] += 1
            ordenar_funcionarios()
            st.rerun()

    # Botão Sênior
    with col_b3:
        label_s = f"➕ Adicionar Sênior ({restantes_senior})"
        if st.button(label_s, disabled=(restantes_senior <= 0), key="btn_add_senior"):
            st.session_state["funcionarios"].append(
                {
                    "id": st.session_state["next_func_id"],
                    "tipo": "senior",
                    "inicio": min_inicio,
                }
            )
            st.session_state["next_func_id"] += 1
            ordenar_funcionarios()
            st.rerun()



# ---------------------------------------------------------
# SLIDERS (aparecem em cima dos botões)
# ---------------------------------------------------------
with sliders_container:
    # st.markdown(
    #     f"_Cada turno dura **{DURACAO_TURNO_HORAS} horas**. "
    #     f"Use o slider de cada linha para escolher o horário de **início** do turno. "
    #     f"O fim é calculado automaticamente._"
    # )

    funcionarios = st.session_state["funcionarios"]

    if not funcionarios:
        st.info(
            "Nenhum funcionário adicionado ainda. "
            "Use os botões a seguir para começar."
        )

    # índice da linha "central" para posicionar o botão de IA
    total_funcs = len(funcionarios)
    mid_idx = total_funcs // 2 if total_funcs > 0 else 0

    for idx, f in enumerate(funcionarios):
        # garante que sempre exista 'inicio' dentro da faixa válida
        if "inicio" not in f:
            f["inicio"] = min_inicio

        inicio_atual = int(f.get("inicio", min_inicio))
        if inicio_atual < min_inicio:
            inicio_atual = min_inicio
        if inicio_atual > max_inicio:
            inicio_atual = max_inicio
        f["inicio"] = inicio_atual

        # chave do slider no session_state
        interval_key = f"intervalo_{f['id']}"

        # Corrige o valor no session_state ANTES de criar o slider
        if interval_key in st.session_state:
            inicio_prev, _ = st.session_state[interval_key]
            if inicio_prev < min_inicio:
                inicio_prev = min_inicio
            if inicio_prev > max_inicio:
                inicio_prev = max_inicio
            st.session_state[interval_key] = (
                inicio_prev,
                inicio_prev + DURACAO_TURNO_HORAS,
            )
        else:
            st.session_state[interval_key] = (
                inicio_atual,
                inicio_atual + DURACAO_TURNO_HORAS,
            )

        # [label] [slider] [texto hora] [X] [botão IA] [espaço]
        col_label, col_slider, col_fim, col_rm, col_toggle, col_d = st.columns(
            [1.8, 5, 0.6, 0.3, 1.3, 0.6]
        )

        # Rótulo colorido de acordo com o tipo, alinhado à direita
        with col_label:
            tipo_label = f["tipo"].capitalize()
            cor = COR_TIPO.get(f["tipo"], "#ffffff")
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <span style="color:{cor}; font-weight:bold;">
                        {tipo_label} #{idx + 1}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Slider (início, fim) com duração fixa
        with col_slider:
            intervalo = st.slider(
                "Período do turno",
                min_value=slider_min,
                max_value=slider_max,
                value=st.session_state[interval_key],
                step=1,
                key=interval_key,
                label_visibility="collapsed",
            )

        # usamos só a primeira bolinha como início
        novo_inicio = int(intervalo[0])
        if novo_inicio < min_inicio:
            novo_inicio = min_inicio
        if novo_inicio > max_inicio:
            novo_inicio = max_inicio

        f["inicio"] = novo_inicio
        fim = f["inicio"] + DURACAO_TURNO_HORAS

        with col_fim:
            st.markdown(f"**{f['inicio']:02d}h → {fim:02d}h**")

        # Botão X (remove funcionário imediatamente)
        with col_rm:
            remove_clicked = st.button("X", key=f"rm_{f['id']}")
        if remove_clicked:
            st.session_state["funcionarios"] = [
                func
                for func in st.session_state["funcionarios"]
                if func["id"] != f["id"]
            ]
            ordenar_funcionarios()
            st.rerun()

        # Botão "Mostrar resultados IAs" na linha central
        if idx == mid_idx and total_funcs > 0:
            ia_ativa = st.session_state.get("mostrar_comparacao", False)
            with col_toggle:
                if ia_ativa:
                    if st.button(
                        "Mostrar resultados IAs",
                        type="primary",
                        key="btn_ias_on",
                    ):
                        st.session_state["mostrar_comparacao"] = False
                        st.rerun()
                else:
                    if st.button(
                        "Mostrar resultados IAs",
                        type="secondary",
                        key="btn_ias_off",
                    ):
                        st.session_state["mostrar_comparacao"] = True
                        st.rerun()

# ---------------------------------------------------------
# Monta turnos do usuário com lista ATUALIZADA
# ---------------------------------------------------------
funcionarios_atualizados = st.session_state["funcionarios"]
turnos_user = funcionarios_para_turnos(
    funcionarios_atualizados, TIPOS, horas_inicio_turno
)

# =====================================================================
# 2. Chegadas x Capacidade – sua escala vs IA-1 e IA-2
# =====================================================================

# Avalia a sua escala aqui para reaproveitar nos gráficos e na tabela
res_user, resumo_user = avaliar_escala_usuario(
    scenario_key=scenario_key,
    turnos_usuario=turnos_user,
    tipos=TIPOS,
)

fig_cap = criar_figura_capacidade_vs_demanda(
    dados,
    res_user,
    res_sa,
    res_brkga,
    mostrar_ias=st.session_state.get("mostrar_comparacao", False),
)
cap_placeholder.pyplot(fig_cap, use_container_width=True)

# =====================================================================
# Resultados numéricos da sua escala vs IA-1 / IA-2
# =====================================================================

if st.session_state.get("mostrar_comparacao", False):
    st.markdown("---")
    st.markdown(
        "<h3 style='text-align: center;'>Comparação de resultados</h3>",
        unsafe_allow_html=True,
    )

    linhas = []

    def add_linha(nome, val_user, val_ia1, val_ia2, tipo="beneficio"):
        """
        tipo:
          - 'beneficio': maior é melhor → ganho = IA - usuário
          - 'menor_melhor': menor é melhor → ganho = usuário - IA
        """
        if tipo == "beneficio":
            ganho1 = val_ia1 - val_user
            ganho2 = val_ia2 - val_user
        else:
            ganho1 = val_user - val_ia1
            ganho2 = val_user - val_ia2

        linhas.append(
            {
                "Métrica": nome,
                "Sua escala": float(val_user),
                "IA-1": float(val_ia1),
                "IA-2": float(val_ia2),
                "Ganho IA-1": float(ganho1),
                "Ganho IA-2": float(ganho2),
            }
        )

    # Métricas de clientes
    add_linha(
        "Clientes atendidos",
        resumo_user.atendidos,
        resumo_sa.atendidos,
        resumo_brkga.atendidos,
        tipo="beneficio",
    )
    add_linha(
        "Clientes perdidos (total)",
        resumo_user.perdidos,
        resumo_sa.perdidos,
        resumo_brkga.perdidos,
        tipo="menor_melhor",
    )
    add_linha(
        "Clientes atrasados (> limite)",
        resumo_user.num_clientes_atrasados,
        resumo_sa.num_clientes_atrasados,
        resumo_brkga.num_clientes_atrasados,
        tipo="menor_melhor",
    )
    add_linha(
        "Não atendidos até o fim do dia",
        resumo_user.num_clientes_nao_atendidos_final,
        resumo_sa.num_clientes_nao_atendidos_final,
        resumo_brkga.num_clientes_nao_atendidos_final,
        tipo="menor_melhor",
    )

    # Métricas financeiras
    add_linha(
        "Faturamento (R$)",
        resumo_user.faturamento,
        resumo_sa.faturamento,
        resumo_brkga.faturamento,
        tipo="beneficio",
    )
    add_linha(
        "Custo com funcionários (R$)",
        resumo_user.custo_funcionarios,
        resumo_sa.custo_funcionarios,
        resumo_brkga.custo_funcionarios,
        tipo="menor_melhor",
    )
    add_linha(
        "Caixa após folha (R$)",
        resumo_user.caixa_apos_folha,
        resumo_sa.caixa_apos_folha,
        resumo_brkga.caixa_apos_folha,
        tipo="beneficio",
    )

    df_comp = pd.DataFrame(linhas)

    # Renomeia colunas de ganho
    df_comp = df_comp.rename(
        columns={
            "Ganho IA-1": "Ganho usando IA-1",
            "Ganho IA-2": "Ganho usando IA-2",
        }
    )

    # Colunas numéricas
    num_cols = [
        "Sua escala",
        "IA-1",
        "IA-2",
        "Ganho usando IA-1",
        "Ganho usando IA-2",
    ]

    # ---- Destaques por linha (IA-1 / IA-2 / Usuário) ----
    def highlight_row(row: pd.Series) -> pd.Series:
        styles = [""] * len(row)
        col_idx = {col: i for i, col in enumerate(df_comp.columns)}

        # estilos
        style_ia1 = "background-color: #d4edda; color: #155724; font-weight:bold;"
        style_ia2 = "background-color: #cce5ff; color: #004085; font-weight:bold;"
        style_user = "background-color: #fff3cd; color: #856404; font-weight:bold;"

        try:
            g1 = float(row["Ganho usando IA-1"])
            g2 = float(row["Ganho usando IA-2"])
        except Exception:
            return pd.Series(styles, index=row.index)

        # Ambos não ajudaram (ganho <= 0) → mérito pro usuário
        if g1 <= 0 and g2 <= 0:
            styles[col_idx["Sua escala"]] = style_user
            return pd.Series(styles, index=row.index)

        # Alguém ajudou: decide quem destacar
        if g1 > 0 and g2 <= 0:
            # só IA-1 foi boa
            styles[col_idx["IA-1"]] = style_ia1
            styles[col_idx["Ganho usando IA-1"]] = style_ia1
        elif g2 > 0 and g1 <= 0:
            # só IA-2 foi boa
            styles[col_idx["IA-2"]] = style_ia2
            styles[col_idx["Ganho usando IA-2"]] = style_ia2
        elif g1 > 0 and g2 > 0:
            if g1 > g2:
                # IA-1 ganhou
                styles[col_idx["IA-1"]] = style_ia1
                styles[col_idx["Ganho usando IA-1"]] = style_ia1
            elif g2 > g1:
                # IA-2 ganhou
                styles[col_idx["IA-2"]] = style_ia2
                styles[col_idx["Ganho usando IA-2"]] = style_ia2
            else:
                # empate positivo → acende as duas com cores diferentes
                styles[col_idx["IA-1"]] = style_ia1
                styles[col_idx["Ganho usando IA-1"]] = style_ia1
                styles[col_idx["IA-2"]] = style_ia2
                styles[col_idx["Ganho usando IA-2"]] = style_ia2

        return pd.Series(styles, index=row.index)

    # ---- Styler: formatação + centralização ----
    df_style = df_comp.style

    # 4 primeiras linhas: inteiros
    df_style = df_style.format("{:,.0f}", subset=pd.IndexSlice[0:3, num_cols])

    # 3 últimas linhas: moeda
    df_style = df_style.format("R$ {:,.2f}", subset=pd.IndexSlice[4:6, num_cols])

    # aplica destaque por linha
    df_style = df_style.apply(highlight_row, axis=1)

    # esconde índice e centraliza tudo
    df_style = (
        df_style.hide(axis="index")
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
    )

    col_esq, col_meio, col_dir = st.columns([1, 6, 2.6])
    with col_meio:
        html_table = df_style.to_html()
        st.markdown(html_table, unsafe_allow_html=True)

        # -------------------------------------------------
        # Quem teve o melhor resultado final?
        # (usa a linha "Caixa após folha (R$)")
        # -------------------------------------------------
        try:
            linha_caixa = df_comp.loc[
                df_comp["Métrica"] == "Caixa após folha (R$)"
            ].iloc[0]

            val_user = float(linha_caixa["Sua escala"])
            val_ia1 = float(linha_caixa["IA-1"])
            val_ia2 = float(linha_caixa["IA-2"])

            max_val = max(val_user, val_ia1, val_ia2)
            ganhadores = []
            if val_user == max_val:
                ganhadores.append("Sua escala")
            if val_ia1 == max_val:
                ganhadores.append("IA-1")
            if val_ia2 == max_val:
                ganhadores.append("IA-2")

            if len(ganhadores) == 1:
                msg = ganhadores[0]
            elif len(ganhadores) == 2:
                msg = f"{ganhadores[0]} e {ganhadores[1]}"
            else:
                msg = "todas as escalas (empate)"

            st.markdown(
                f"**Melhor desempenho (com base em `Caixa após folha (R$)`):** {msg}."
            )
        except Exception:
            # Se por algum motivo não achar a linha, simplesmente não mostra o resumo
            pass



    # =================================================================
    # Gantt comparativo (Usuário vs IA-1 vs IA-2)
    # =================================================================

    st.markdown("---")
    # st.markdown(
    #     "<h3 style='text-align: center;'>Você vs IA-1 vs IA-2</h3>",
    #     unsafe_allow_html=True,
    # )

    fig_gantt_comp = criar_figura_gantt_triplo(
        horas=horas,
        turnos_user=turnos_user,
        turnos_ia1=turnos_sa,
        turnos_ia2=turnos_brkga,
        tipos=TIPOS,
        titulo_user="Sua escala",
        titulo_ia1="Escala da IA-1",
        titulo_ia2="Escala da IA-2",
        funcionarios_user_ordenados=st.session_state["funcionarios"],
    )

    fig_gantt_comp.set_size_inches(7, 5)

    col_esq, col_centro, col_dir = st.columns([1, 6, 2.6])
    with col_centro:
        st.pyplot(fig_gantt_comp, use_container_width=False)
