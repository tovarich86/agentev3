# analytical_engine.py (versão final aprimorada)

import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy import stats
import unicodedata

class AnalyticalEngine:
    """
    Um motor de análise que opera sobre o JSON de resumo pré-processado
    para responder perguntas quantitativas e de listagem.
    Versão com análise estatística aprofundada e roteador declarativo.
    """
    def __init__(self, summary_data: dict, knowledge_base: dict):
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base
        
        # --- MELHORIA 2: ROTEADOR DECLARATIVO ---
        # Mapeamento de funções de análise para suas regras de ativação.
        # Cada regra é uma função lambda que retorna True se a intenção for detectada.
        self.intent_rules = [
            (lambda q: 'desconto' in q and ('preco de exercicio' in q or 'strike' in q), self._analyze_strike_discount),
            (lambda q: 'tsr' in q, self._analyze_tsr),
            (lambda q: 'vesting' in q, self._analyze_vesting_period),
            (lambda q: 'lockup' in q or 'lock-up' in q, self._analyze_lockup_period),
        ]
        
    def _normalize_text(self, text: str) -> str:
        nfkd_form = unicodedata.normalize('NFKD', text.lower())
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def answer_query(self, query: str) -> tuple:
        """
        Ponto de entrada principal. Usa o roteador declarativo para encontrar
        e executar a ferramenta de análise correta.
        """
        normalized_query = self._normalize_text(query)

        # Itera sobre as regras de intenção. A primeira que corresponder é executada.
        for intent_checker_func, analysis_func in self.intent_rules:
            if intent_checker_func(normalized_query):
                return analysis_func(normalized_query)

        # Se nenhuma regra específica for acionada, cai na busca genérica por tópicos.
        return self._find_companies_by_general_topic(normalized_query)


    # --- FUNÇÃO DE ANÁLISE DE DESCONTO COM ESTATÍSTICAS APROFUNDADAS ---
    def _analyze_strike_discount(self, normalized_query: str) -> tuple:
        """
        Analisa o fato 'desconto_strike_price' com estatísticas detalhadas
        e retorna a lista de empresas com seus respectivos descontos.
        """
        companies_and_discounts = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico', 0)
                if valor_numerico > 0:
                    companies_and_discounts.append((company, valor_numerico * 100))

        if not companies_and_discounts:
            return "Nenhuma empresa com desconto explícito no preço de exercício foi encontrada.", None
        
        discounts = np.array([item[1] for item in companies_and_discounts])
        
        # --- MELHORIA 1: CÁLCULO DE ESTATÍSTICAS DETALHADAS ---
        stats_data = {
            "Total de Empresas": len(discounts),
            "Desconto Médio": f"{np.mean(discounts):.2f}%",
            "Desvio Padrão": f"{np.std(discounts):.2f}%",
            "Mínimo": f"{np.min(discounts):.2f}%",
            "25º Percentil": f"{np.percentile(discounts, 25):.2f}%",
            "Mediana (50º)": f"{np.median(discounts):.2f}%",
            "75º Percentil": f"{np.percentile(discounts, 75):.2f}%",
            "Máximo": f"{np.max(discounts):.2f}%",
        }
        
        # Formata o relatório para exibição
        report_text = "### Análise de Desconto no Preço de Exercício\n"
        for key, value in stats_data.items():
            report_text += f"- **{key}:** {value}\n"
        
        df = pd.DataFrame(companies_and_discounts, columns=["Empresa", "Desconto Aplicado (%)"])
        df_sorted = df.sort_values(by="Desconto Aplicado (%)", ascending=False).reset_index(drop=True)
        
        return report_text, df_sorted

    # --- Outras funções de análise (podem ser aprimoradas da mesma forma no futuro) ---

    def _analyze_tsr(self, normalized_query: str) -> tuple:
        # ... (código da função permanece o mesmo) ...
        results = defaultdict(list)
        tsr_type = 'qualquer'
        if 'relativo' in normalized_query and 'absoluto' in normalized_query: tsr_type = 'ambos'
        elif 'relativo' in normalized_query: tsr_type = 'relativo'
        elif 'absoluto' in normalized_query: tsr_type = 'absoluto'
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

    def _analyze_vesting_period(self, normalized_query: str) -> tuple:
        # ... (implementar análise detalhada similar à de desconto) ...
        return "Análise de Vesting a ser implementada.", pd.DataFrame()

    def _analyze_lockup_period(self, normalized_query: str) -> tuple:
        # ... (implementar análise detalhada similar à de desconto) ...
        return "Análise de Lock-up a ser implementada.", pd.DataFrame()

    def _find_companies_by_general_topic(self, normalized_query: str) -> tuple:
        """Função de fallback que busca por qualquer tópico do dicionário."""
        flat_map = self.kb_flat_map()
        found_topic_details = None
        for alias in sorted(flat_map.keys(), key=len, reverse=True):
             if re.search(r'\b' + re.escape(self._normalize_text(alias)) + r'\b', normalized_query):
                found_topic_details = flat_map[alias]
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
