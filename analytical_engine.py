# analytical_engine.py (versão com roteador de intenção aprimorado)

import numpy as np
import pandas as pd
import re
from collections import defaultdict

class AnalyticalEngine:
    """
    Um motor de análise que opera sobre o JSON de resumo pré-processado
    para responder perguntas quantitativas e de listagem.
    """
    def __init__(self, summary_data: dict, knowledge_base: dict):
        """
        Inicializa o motor com os dados de resumo e a base de conhecimento (dicionário).
        """
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base
        # O mapa de rotas continua útil para buscas mais simples
        self.alias_router_map = self._create_alias_map()

    def _create_alias_map(self) -> dict:
        """
        Cria um mapa de 'alias -> referência da função de análise' para roteamento rápido.
        """
        alias_map = {}
        topic_to_function_map = {
            "PrecoDesconto": self._analyze_strike_discount,
            "TSR_Absoluto": self._analyze_tsr,
            "TSR_Relativo": self._analyze_tsr,
            "Vesting": self._analyze_vesting_period,
            "Lockup": self._analyze_lockup_period,
        }

        for section, topics in self.kb.items():
            for topic_name, aliases in topics.items():
                if topic_name in topic_to_function_map:
                    for alias in aliases:
                        alias_map[alias.lower()] = topic_to_function_map[topic_name]
        return alias_map

    def answer_query(self, query: str) -> tuple:
        """
        Ponto de entrada principal. Analisa a query com regras de intenção
        e a direciona para a função correta.
        """
        query_lower = query.lower()

        # --- MUDANÇA CRÍTICA: REGRAS DE INTENÇÃO ESPECÍFICAS ---
        # Verificamos combinações de palavras-chave de alta prioridade primeiro.

        # Regra 1: Intenção de analisar Desconto no Preço de Exercício
        if 'desconto' in query_lower and ('preço de exercício' in query_lower or 'strike' in query_lower):
            return self._analyze_strike_discount(query=query_lower)
        
        # Regra 2: Intenção de analisar TSR (já tratada na função, mas poderia ser explícita aqui também)
        if 'tsr' in query_lower:
            return self._analyze_tsr(query=query_lower)

        # --- ROTEADOR BASEADO EM ALIAS (FALLBACK) ---
        # Se nenhuma regra específica for acionada, tenta a busca por alias.
        for alias in sorted(self.alias_router_map.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                analysis_function = self.alias_router_map[alias]
                return analysis_function(query=query_lower)

        # Se nada funcionar, cai na busca genérica por tópicos.
        return self._find_companies_by_general_topic(query_lower)

    # --- FUNÇÕES DE ANÁLISE ESPECÍFICAS (sem alterações) ---

    def _analyze_strike_discount(self, query: str) -> tuple:
        """Responde a: "quantas empresas tem desconto...", "qual é o desconto médio..." """
        discounts = []
        companies_with_discount = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico', 0)
                if valor_numerico > 0:
                    discounts.append(valor_numerico * 100)
                    companies_with_discount.append(company)

        if not discounts:
            return "Nenhuma empresa com desconto explícito no preço de exercício foi encontrada nos documentos.", None
        
        media = np.mean(discounts)
        minimo = np.min(discounts)
        maximo = np.max(discounts)
        
        report_text = f"""
        ### Análise de Desconto no Preço de Exercício
        | Métrica | Valor |
        |---|---|
        | **Total de Empresas com Desconto** | {len(companies_with_discount)} |
        | **Desconto Médio** | {media:.2f}% |
        | **Desconto Mínimo** | {minimo:.2f}% |
        | **Desconto Máximo** | {maximo:.2f}% |
        """
        df = pd.DataFrame(sorted(companies_with_discount), columns=["Empresas com Desconto no Preço"])
        return report_text, df

    def _analyze_tsr(self, query: str) -> tuple:
        """Analisa a presença de TSR, diferenciando entre Relativo, Absoluto ou ambos."""
        results = defaultdict(list)
        tsr_type = 'qualquer'
        if 'relativo' in query and 'absoluto' in query: tsr_type = 'ambos'
        elif 'relativo' in query: tsr_type = 'relativo'
        elif 'absoluto' in query: tsr_type = 'absoluto'
        
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            topics = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})
            has_tsr_absoluto = "TSR_Absoluto" in topics
            has_tsr_relativo = 'tsr_relativo' in facts and facts['tsr_relativo'].get('presente', False)
            if has_tsr_absoluto: results['absoluto'].append(company)
            if has_tsr_relativo: results['relativo'].append(company)
            if has_tsr_absoluto and has_tsr_relativo: results['ambos'].append(company)
            if has_tsr_absoluto or has_tsr_relativo: results['qualquer'].append(company)

        target_companies = results.get(tsr_type, [])
        if not target_companies:
            return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type}'.", None
        report_text = f"Encontradas **{len(target_companies)}** empresas com o critério de TSR: **{tsr_type.upper()}**."
        df = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type.upper()})"])
        return report_text, df

    def _analyze_vesting_period(self, query: str) -> tuple:
        """Analisa o fato 'periodo_vesting' e calcula estatísticas."""
        # ... (código da função permanece o mesmo) ...
        return "Análise de Vesting a ser implementada.", None

    def _analyze_lockup_period(self, query: str) -> tuple:
        """Analisa o fato 'periodo_lockup' e calcula estatísticas."""
        # ... (código da função permanece o mesmo) ...
        return "Análise de Lock-up a ser implementada.", None

    def _find_companies_by_general_topic(self, query: str) -> tuple:
        """Busca genérica por um tópico para perguntas simples."""
        # ... (código da função permanece o mesmo) ...
        return "Nenhum tópico específico encontrado na sua pergunta.", None
    
    def kb_flat_map(self) -> dict:
        """Helper para criar um mapa chato de alias -> detalhes do tópico."""
        # ... (código da função permanece o mesmo) ...
        return {}
