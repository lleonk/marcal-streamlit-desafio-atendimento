"""
Módulo para carregamento das soluções ótimas salvas em JSON.
"""

import json
from typing import Dict, Any, Optional


def carregar_turnos_otimos(cenario_key: str, metodo_preferido: str) -> Optional[Dict[str, Any]]:
    """
    Carrega os turnos ótimos salvos para um cenário e método específicos.
    
    Args:
        cenario_key: Chave do cenário (ex: "black_swan")
        metodo_preferido: Método preferido (ex: "sa", "brkga")
        
    Returns:
        Dicionário com os dados dos turnos ótimos ou None se não encontrado
    """
    try:
        with open("solucoes_otimas.json", "r", encoding="utf-8") as f:
            dados = json.load(f)
        
        if cenario_key not in dados:
            return None
            
        if metodo_preferido not in dados[cenario_key]:
            return None
            
        return dados[cenario_key][metodo_preferido]
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None