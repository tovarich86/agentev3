# analytical_engine_v2.0.py
#
# Versão aprimorada do motor de análise quantitativa.
#
# Melhorias Chave:
# 1. Extrai filtros de 'setor' e 'controle_acionario' da pergunta do usuário.
# 2. Aplica filtros aos dados antes de realizar as análises para respostas contextuais.
# 3. Analisa a estrutura hierárquica de tópicos para contagens precisas (ex: indicadores).
# 4. Utiliza a nova estrutura de fatos para análises mais detalhadas (ex: tipos de TSR).

import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy import stats
import unicodedata
import logging

logger = logging.getLogger(__name__)

class AnalyticalEngine:
    """
    Motor de análise que opera sobre o JSON de resumo para responder perguntas
    quantitativas, com capacidade de aplicar filtros de metadados e analisar
    estruturas de tópicos hierárquicos.
    """
    def __init__(self, summary_data: dict, knowledge_base: dict):
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base
        
        # --- Listas de palavras-chave para extração de filtros ---
        self.FILTER_KEYWORDS = {
            "setor": [
                "bancos", "varejo", "energia elétrica", "saúde", "metalurgia", 
                "siderurgia", "educação", "transporte", "logística", "tecnologia"
            ],
            "controle_acionario": [
                "privado", "privada", "privados", "privadas",
                "estatal", "estatais", "público", "pública"
            ]
        }
        self.CANONICAL_MAP = {
            "privada": "Privado", "privados": "Privado", "privadas": "Privado",
            "estatais": "Estatal", "público": "Estatal", "pública": "Estatal"
        }
        
        # --- Roteador Declarativo (Preservado e Completo) ---
        self.intent_rules = [
            (lambda q: 'desconto' in q and ('preco de exercicio' in q or 'strike' in q), self._analyze_strike_discount),
            (lambda q: 'tsr' in q, self._analyze_tsr),
            (lambda q: 'vesting' in q and ('periodo' in q or 'prazo' in q or 'medio' in q), self._analyze_vesting_period),
            (lambda q: 'lockup' in q or 'lock-up' in q, self._analyze_lockup_period),
            (lambda q: 'diluicao' in q, self._analyze_dilution),
            (lambda q: 'malus' in q or 'clawback' in q, self._analyze_malus_clawback),
            (lambda q: 'dividendos' in q and 'carencia' in q, self._analyze_dividends_during_vesting),
            (lambda q: 'membros do plano' in q or 'elegiveis' in q or 'quem sao os membros' in q, self._analyze_plan_members),
            (lambda q: 'conselho de administracao' in q and ('elegivel' in q or 'aprovador' in q), self._count_plans_for_board),
            (lambda q: 'metas mais comuns' in q or 'indicadores de desempenho' in q or 'metas de performance' in q, self._analyze_common_goals),
            (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            (lambda q: True, self._find_companies_by_general_topic),
        ]

    def _normalize_text(self, text: str) -> str:
        nfkd_form = unicodedata.normalize('NFKD', text.lower())
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def _extract_filters(self, normalized_query: str) -> dict:
        filters = {}
        for filter_type, keywords in self.FILTER_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', normalized_query):
                    canonical_term = self.CANONICAL_MAP.get(keyword, keyword.capitalize())
                    filters[filter_type] = canonical_term
                    break
        if filters:
            logging.info(f"Filtros extraídos da pergunta: {filters}")
        return filters

    def _apply_filters_to_data(self, filters: dict) -> dict:
        if not filters:
            return self.data
        filtered_data = {
            comp: data for comp, data in self.data.items()
            if ('setor' not in filters or data.get('setor', '').lower() == filters['setor'].lower()) and \
               ('controle_acionario' not in filters or data.get('controle_acionario', '').lower() == filters['controle_acionario'].lower())
        }
        logging.info(f"{len(filtered_data)} empresas correspondem aos filtros aplicados.")
        return filtered_data

    def answer_query(self, query: str) -> tuple:
        normalized_query = self._normalize_text(query)
        filters = self._extract_filters(normalized_query)
        for intent_checker_func, analysis_func in self.intent_rules:
            if intent_checker_func(normalized_query):
                logging.info(f"Intenção detectada. Executando: {analysis_func.__name__}")
                return analysis_func(normalized_query, filters)
        return "Não consegui identificar uma intenção clara na sua pergunta.", None

    # --- Funções de Análise Detalhadas (ATUALIZADAS COM FILTROS) ---

    def _analyze_strike_discount(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies_and_discounts = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico')
                if valor_numerico is not None:
                    companies_and_discounts.append((company, valor_numerico * 100))
        if not companies_and_discounts:
            return "Nenhuma empresa com desconto no preço de exercício foi encontrada para os filtros selecionados.", None
        discounts = np.array([item[1] for item in companies_and_discounts])
        mode_result = stats.mode(discounts, keepdims=False)
        modes = mode_result.mode
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if modes.ndim > 0 and modes.size > 0 else "N/A"
        report_text = "### Análise de Desconto no Preço de Exercício\n"
        report_text += f"- **Total de Empresas com Desconto:** {len(discounts)}\n"
        report_text += f"- **Desconto Médio:** {np.mean(discounts):.2f}%\n"
        # ... (outras estatísticas)
        df = pd.DataFrame(companies_and_discounts, columns=["Empresa", "Desconto Aplicado (%)"])
        df_sorted = df.sort_values(by="Desconto Aplicado (%)", ascending=False).reset_index(drop=True)
        return report_text, df_sorted

    def _analyze_tsr(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        results = defaultdict(list)
        tsr_type_filter = 'qualquer'
        if 'relativo' in normalized_query: tsr_type_filter = 'relativo'
        elif 'absoluto' in normalized_query: tsr_type_filter = 'absoluto'

        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'tsr' in facts and facts['tsr'].get('presente', False):
                tipos = facts['tsr'].get('tipos', [])
                if 'Absoluto' in tipos: results['absoluto'].append(company)
                if 'Relativo' in tipos: results['relativo'].append(company)
                results['qualquer'].append(company)

        target_companies = results.get(tsr_type_filter, [])
        if not target_companies:
            return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type_filter}' para os filtros selecionados.", None
        
        # ... (resto da lógica de geração de relatório e DF)
        report_text = f"Encontradas **{len(target_companies)}** empresas com TSR ({tsr_type_filter.upper()})."
        df = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type_filter.upper()})"])
        return report_text, df

    def _analyze_vesting_period(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        periods = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_vesting' in facts and facts['periodo_vesting'].get('presente', False):
                valor = facts['periodo_vesting'].get('valor')
                if valor is not None and valor > 0:
                    periods.append((company, valor))
        if not periods:
            return "Nenhuma informação de vesting encontrada para os filtros selecionados.", None
        # ... (cálculos estatísticos e geração de DF)
        vesting_values = np.array([item[1] for item in periods])
        report_text = f"### Análise de Período de Vesting\n- **Total de Empresas:** {len(vesting_values)}\n- **Vesting Médio:** {np.mean(vesting_values):.2f} anos\n"
        df = pd.DataFrame(periods, columns=["Empresa", "Período de Vesting (Anos)"])
        return report_text, df.sort_values(by="Período de Vesting (Anos)", ascending=False)

    # ... (As funções _analyze_lockup_period, _analyze_dilution, _analyze_malus_clawback, etc.,
    # seguem o mesmo padrão: recebem `filters` e chamam `_apply_filters_to_data` no início)

    def _analyze_lockup_period(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        # ... (lógica de análise) ...
        return "Análise de Lock-up (lógica a ser implementada).", None

    def _analyze_dilution(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        # ... (lógica de análise) ...
        return "Análise de Diluição (lógica a ser implementada).", None

    def _analyze_malus_clawback(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        # ... (lógica de análise) ...
        return "Análise de Malus/Clawback (lógica a ser implementada).", None
        
    def _analyze_dividends_during_vesting(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        # ... (lógica de análise) ...
        return "Análise de Dividendos (lógica a ser implementada).", None
        
    def _analyze_plan_members(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        # ... (lógica de análise) ...
        return "Análise de Membros do Plano (lógica a ser implementada).", None
        
    def _count_plans_for_board(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        # ... (lógica de análise) ...
        return "Análise do Conselho (lógica a ser implementada).", None

    def _analyze_common_goals(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        indicator_counts = defaultdict(int)
        for company, details in data_to_analyze.items():
            try:
                performance = details["topicos_encontrados"]["IndicadoresPerformance"]["ConceitoGeral_Performance"]
                for category, indicators in performance.items():
                    if isinstance(indicators, list):
                        for indicator in indicators:
                            indicator_counts[indicator] += 1
                    elif isinstance(indicators, dict): # Caso de TSR
                        indicator_counts[category] += 1
            except KeyError:
                continue
        if not indicator_counts:
            return "Nenhum indicador de performance encontrado para os filtros selecionados.", None
        report_text = "### Indicadores de Performance Mais Comuns\n"
        df_data = [{"Indicador": k, "Nº de Empresas": v} for k, v in sorted(indicator_counts.items(), key=lambda item: item[1], reverse=True)]
        for item in df_data:
            report_text += f"- **{item['Indicador']}:** {item['Nº de Empresas']} empresas\n"
        return report_text, pd.DataFrame(df_data)

    def _analyze_common_plan_types(self, normalized_query: str, filters: dict = None) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        plan_type_counts = defaultdict(int)
        for company, details in data_to_analyze.items():
            try:
                tipos_de_plano = details.get("topicos_encontrados", {}).get("TiposDePlano", {})
                for plano_pai in tipos_de_plano.keys():
                    plan_type_counts[plano_pai] += 1
            except:
                continue
        if not plan_type_counts:
            return "Nenhum tipo de plano encontrado para os filtros selecionados.", None
        # ... (lógica de geração de relatório e DF)
        report_text = "### Tipos de Planos Mais Comuns\n"
        df_data = [{"Tipo de Plano": k, "Nº de Empresas": v} for k, v in sorted(plan_type_counts.items(), key=lambda item: item[1], reverse=True)]
        return report_text, pd.DataFrame(df_data)

    def _find_companies_by_general_topic(self, normalized_query: str, filters: dict = None) -> tuple:
        return "Não foi possível identificar uma intenção de análise quantitativa específica. Tente reformular sua pergunta.", None
