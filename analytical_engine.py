# analytical_engine.py (versão com moda, mediana e desconto individual)

import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy import stats # MUDANÇA 1: Importando a biblioteca scipy para calcular a moda

class AnalyticalEngine:
    """
    Um motor de análise que opera sobre o JSON de resumo pré-processado
    para responder perguntas quantitativas e de listagem.
    """
    def __init__(self, summary_data: dict, knowledge_base: dict):
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base
        self.alias_router_map = self._create_alias_map()

    def _create_alias_map(self) -> dict:
        # ... (código da função permanece o mesmo) ...
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
        # ... (código da função permanece o mesmo, com as regras de intenção) ...
        query_lower = query.lower()
        if 'desconto' in query_lower and ('preço de exercício' in query_lower or 'strike' in query_lower):
            return self._analyze_strike_discount(query=query_lower)
        if 'tsr' in query_lower:
            return self._analyze_tsr(query=query_lower)
        for alias in sorted(self.alias_router_map.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                analysis_function = self.alias_router_map[alias]
                return analysis_function(query=query_lower)
        return self._find_companies_by_general_topic(query_lower)

    # --- FUNÇÃO DE ANÁLISE DE DESCONTO APRIMORADA ---
    def _analyze_strike_discount(self, query: str) -> tuple:
        """
        Analisa o fato 'desconto_strike_price', calcula as estatísticas (incluindo
        moda e mediana) e retorna a lista de empresas com seus respectivos descontos.
        """
        companies_and_discounts = []

        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico', 0)
                if valor_numerico > 0:
                    # Armazena uma tupla com o nome da empresa e o valor do desconto em %
                    companies_and_discounts.append((company, valor_numerico * 100))

        if not companies_and_discounts:
            return "Nenhuma empresa com desconto explícito no preço de exercício foi encontrada nos documentos.", None
        
        # Extrai apenas os valores de desconto para os cálculos estatísticos
        discounts = [item[1] for item in companies_and_discounts]
        
        # MUDANÇA 2: Cálculo das novas métricas
        media = np.mean(discounts)
        mediana = np.median(discounts)
        minimo = np.min(discounts)
        maximo = np.max(discounts)
        
        # O cálculo da moda pode retornar múltiplos valores se houver empate. Pegamos o primeiro.
        moda_result = stats.mode(discounts, keepdims=False)
        moda = moda_result.mode
        
        report_text = f"""
        ### Análise de Desconto no Preço de Exercício

        | Métrica | Valor |
        |---|---|
        | **Total de Empresas com Desconto** | {len(discounts)} |
        | **Desconto Médio** | {media:.2f}% |
        | **Desconto Mediano (valor central)** | {mediana:.2f}% |
        | **Desconto (Moda - mais comum)** | {moda:.2f}% |
        | **Desconto Mínimo** | {minimo:.2f}% |
        | **Desconto Máximo** | {maximo:.2f}% |
        """
        
        # MUDANÇA 3: Criação do DataFrame com as duas colunas e ordenação
        df = pd.DataFrame(companies_and_discounts, columns=["Empresa", "Desconto Aplicado (%)"])
        df_sorted = df.sort_values(by="Desconto Aplicado (%)", ascending=False).reset_index(drop=True)
        
        return report_text, df_sorted

    # --- OUTRAS FUNÇÕES DE ANÁLISE (sem alterações) ---
    def _analyze_tsr(self, query: str) -> tuple:
        # ... (código da função permanece o mesmo) ...
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
        if not target_companies: return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type}'.", None
        report_text = f"Encontradas **{len(target_companies)}** empresas com o critério de TSR: **{tsr_type.upper()}**."
        df = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type.upper()})"])
        return report_text, df

    def _analyze_vesting_period(self, query: str) -> tuple:
        # ... (código da função permanece o mesmo) ...
        return "Análise de Vesting a ser implementada.", None

    def _analyze_lockup_period(self, query: str) -> tuple:
        # ... (código da função permanece o mesmo) ...
        return "Análise de Lock-up a ser implementada.", None

    def _find_companies_by_general_topic(self, query: str) -> tuple:
        # ... (código da função permanece o mesmo) ...
        return "Nenhum tópico específico encontrado na sua pergunta.", None

    def kb_flat_map(self) -> dict:
        # ... (código da função permanece o mesmo) ...
        if hasattr(self, '_kb_flat_map_cache'): return self._kb_flat_map_cache
        flat_map = {}
        for section, topics in self.kb.items():
            for topic_name_raw, aliases in topics.items():
                topic_name_formatted = topic_name_raw.replace('_', ' ')
                details = (section, topic_name_formatted, topic_name_raw)
                flat_map[topic_name_formatted.lower()] = details
                for alias in aliases: flat_map[alias.lower()] = details
        self._kb_flat_map_cache = flat_map
        return flat_map
