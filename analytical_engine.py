# analytical_engine.py

"""
Módulo contendo o Motor de Análise Quantitativa (AnalyticalEngine).

Esta classe é projetada para operar sobre o arquivo de resumo estruturado
(resumo_fatos_e_topicos_final_enriquecido.json) para responder a perguntas
quantitativas e de listagem de forma rápida e precisa, sem a necessidade
de consultar o modelo de linguagem (LLM) ou os índices vetoriais (FAISS).
"""

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
        Inicializa o motor com os dados de resumo e a base de conhecimento.

        Args:
            summary_data (dict): O conteúdo carregado do 'resumo_fatos_e_topicos...json'.
            knowledge_base (dict): O DICIONARIO_UNIFICADO_HIERARQUICO importado de knowledge_base.py.
        """
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base
        
        # Cria o mapa de rotas na inicialização para máxima eficiência.
        self.alias_router_map = self._create_alias_map()

    def _create_alias_map(self) -> dict:
        """
        Cria um mapa de 'alias -> referência da função de análise' para roteamento rápido.
        Este mapa é a peça central da inteligência do roteador.
        """
        alias_map = {}
        
        # Mapeamento explícito de tópicos para funções de análise específicas.
        # A chave é o nome do tópico no DICIONARIO, o valor é a referência da função.
        topic_to_function_map = {
            "PrecoDesconto": self._analyze_strike_discount,
            "TSR_Absoluto": self._analyze_tsr,
            "TSR_Relativo": self._analyze_tsr,
            "Vesting": self._analyze_vesting_period,
            "Lockup": self._analyze_lockup_period,
        }

        # Itera sobre a base de conhecimento para popular o mapa de rotas.
        for section, topics in self.kb.items():
            for topic_name, aliases in topics.items():
                if topic_name in topic_to_function_map:
                    # Adiciona todos os sinônimos (aliases) ao mapa.
                    for alias in aliases:
                        alias_map[alias.lower()] = topic_to_function_map[topic_name]
        return alias_map

    def answer_query(self, query: str) -> tuple:
        """
        Ponto de entrada principal. Analisa a query e a direciona para a função correta.
        Retorna uma tupla (string_do_relatorio, dataframe_pandas).
        """
        query_lower = query.lower()
        
        # Itera sobre os aliases, dos mais longos aos mais curtos, para garantir o
        # match mais específico (ex: "opções de compra" antes de "opções").
        for alias in sorted(self.alias_router_map.keys(), key=len, reverse=True):
            # Usa regex com limites de palavra (\b) para evitar matches parciais.
            if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                # Encontrou uma rota! Chama a função de análise correspondente.
                analysis_function = self.alias_router_map[alias]
                return analysis_function(query=query_lower)

        # Se nenhuma rota específica foi encontrada, usa a busca genérica por tópicos.
        return self._find_companies_by_general_topic(query_lower)

    # --- FUNÇÕES DE ANÁLISE ESPECÍFICAS ---

    def _analyze_strike_discount(self, query: str) -> tuple:
        """
        Analisa o fato 'desconto_strike_price' e calcula as estatísticas.
        """
        discounts = []
        companies_with_discount = []

        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico', 0)
                if valor_numerico > 0:
                    discounts.append(valor_numerico * 100) # Converte para porcentagem
                    companies_with_discount.append(company)

        if not discounts:
            return "Nenhuma empresa com desconto explícito no preço de exercício foi encontrada.", None

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
        """
        Analisa a presença de TSR, diferenciando entre Relativo, Absoluto ou ambos.
        """
        results = defaultdict(list)
        tsr_type = 'qualquer' # Padrão
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
        periods_in_years = []
        companies = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_vesting' in facts and facts['periodo_vesting'].get('presente', False):
                valor = facts['periodo_vesting'].get('valor', 0)
                unidade = facts['periodo_vesting'].get('unidade', 'ano')
                if valor > 0:
                    # Normaliza tudo para anos para uma comparação justa.
                    valor_em_anos = valor if unidade == 'ano' else valor / 12
                    periods_in_years.append(valor_em_anos)
                    companies.append(company)

        if not periods_in_years:
            return "Nenhuma empresa com dados numéricos de período de vesting foi encontrada.", None

        report = f"""
        ### Análise de Período de Vesting
        | Métrica | Valor |
        |---|---|
        | **Total de Empresas com Vesting** | {len(companies)} |
        | **Período Médio** | {np.mean(periods_in_years):.1f} anos |
        | **Período Mínimo** | {np.min(periods_in_years):.1f} anos |
        | **Período Máximo** | {np.max(periods_in_years):.1f} anos |
        """
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Período de Vesting Definido"])
        return report, df

    def _analyze_lockup_period(self, query: str) -> tuple:
        """Analisa o fato 'periodo_lockup' e calcula estatísticas."""
        periods_in_years = []
        companies = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_lockup' in facts and facts['periodo_lockup'].get('presente', False):
                valor = facts['periodo_lockup'].get('valor', 0)
                unidade = facts['periodo_lockup'].get('unidade', 'ano')
                if valor > 0:
                    valor_em_anos = valor if unidade == 'ano' else valor / 12
                    periods_in_years.append(valor_em_anos)
                    companies.append(company)

        if not periods_in_years:
            return "Nenhuma empresa com dados numéricos de período de lock-up foi encontrada.", None

        report = f"""
        ### Análise de Período de Lock-up
        | Métrica | Valor |
        |---|---|
        | **Total de Empresas com Lock-up** | {len(companies)} |
        | **Período Médio** | {np.mean(periods_in_years):.1f} anos |
        | **Período Mínimo** | {np.min(periods_in_years):.1f} anos |
        | **Período Máximo** | {np.max(periods_in_years):.1f} anos |
        """
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Período de Lock-up Definido"])
        return report, df

    def _find_companies_by_general_topic(self, query: str) -> tuple:
        """
        Função de fallback. Busca por um tópico genérico na query para listar empresas
        que o mencionam, sem realizar análises numéricas.
        Ex: "Quais empresas têm Malus e Clawback?"
        """
        found_topic_details = None
        for alias in sorted(self.kb_flat_map().keys(), key=len, reverse=True):
             if re.search(r'\b' + re.escape(alias) + r'\b', query):
                found_topic_details = self.kb_flat_map()[alias]
                break
        
        if not found_topic_details:
            return "Não consegui identificar um tópico técnico conhecido na sua pergunta para realizar a busca agregada.", None

        section, topic_name_formatted, topic_name_raw = found_topic_details
        companies = [
            company_name for company_name, details in self.data.items()
            if section in details.get("topicos_encontrados", {}) and topic_name_raw in details["topicos_encontrados"][section]
        ]

        if not companies:
            return f"Nenhuma empresa encontrada com o tópico '{topic_name_formatted}'.", None

        report_text = f"Encontradas **{len(companies)}** empresas com o tópico: **{topic_name_formatted}**"
        df = pd.DataFrame(sorted(companies), columns=[f"Empresas com {topic_name_formatted}"])
        return report_text, df
    
    def kb_flat_map(self) -> dict:
        """Helper para criar um mapa chato de alias -> detalhes do tópico."""
        if hasattr(self, '_kb_flat_map_cache'):
            return self._kb_flat_map_cache
        
        flat_map = {}
        for section, topics in self.kb.items():
            for topic_name_raw, aliases in topics.items():
                topic_name_formatted = topic_name_raw.replace('_', ' ')
                details = (section, topic_name_formatted, topic_name_raw)
                flat_map[topic_name_formatted.lower()] = details
                for alias in aliases:
                    flat_map[alias.lower()] = details
        
        self._kb_flat_map_cache = flat_map
        return flat_map
