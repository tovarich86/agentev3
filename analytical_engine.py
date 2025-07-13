# analytical_engine.py

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

        Args:
            summary_data (dict): O conteúdo carregado do 'resumo_fatos_e_topicos...json'.
            knowledge_base (dict): O DICIONARIO_UNIFICADO_HIERARQUICO.
        """
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base

    def answer_query(self, query: str):
        """
        Ponto de entrada principal. Analisa a query e a direciona para a função correta.
        """
        query_lower = query.lower()

        # Rota 1: Análise de Desconto no Strike Price (a mais complexa)
        if re.search(r'desconto.*(strike|preço.*exercício)', query_lower):
            return self._analyze_strike_discount()

        # Rota 2: Análise de TSR (com subtipos)
        if 'tsr' in query_lower:
            tsr_type = 'qualquer'
            if 'relativo' in query_lower and 'absoluto' in query_lower:
                tsr_type = 'ambos'
            elif 'relativo' in query_lower:
                tsr_type = 'relativo'
            elif 'absoluto' in query_lower:
                tsr_type = 'absoluto'
            return self._analyze_tsr(tsr_type)

        # Rota 3: Busca genérica por tópicos (para perguntas simples de "quais/quantas")
        # Ex: "Quais empresas têm clawback?"
        return self._find_companies_by_general_topic(query_lower)

    def _analyze_strike_discount(self):
        """
        Responde a: "quantas empresas tem desconto...", "qual é o desconto médio..."
        """
        discounts = []
        companies_with_discount = []

        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            # Acessamos o fato estruturado extraído pelo seu script de resumo
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico', 0)
                if valor_numerico > 0:
                    # Armazenamos o valor do desconto (ex: 0.2 para 20%)
                    discounts.append(valor_numerico * 100) # Convertendo para %
                    companies_with_discount.append(company)

        if not discounts:
            return "Nenhuma empresa com desconto explícito no preço de exercício foi encontrada nos documentos.", None

        # Usamos numpy para as estatísticas
        media = np.mean(discounts)
        minimo = np.min(discounts)
        maximo = np.max(discounts)
        
        # Montamos um relatório em texto
        report_text = f"""
        ### Análise de Desconto no Preço de Exercício

        | Métrica | Valor |
        |---|---|
        | **Total de Empresas com Desconto** | {len(companies_with_discount)} |
        | **Desconto Médio** | {media:.2f}% |
        | **Desconto Mínimo** | {minimo:.2f}% |
        | **Desconto Máximo** | {maximo:.2f}% |
        """
        
        # Criamos um DataFrame para a lista de empresas
        df = pd.DataFrame(sorted(companies_with_discount), columns=["Empresas com Desconto no Preço"])
        
        return report_text, df

    def _analyze_tsr(self, tsr_type: str = 'qualquer'):
        """
        Responde a: "quais empresas tem tsr?", "...tsr relativo?", "...tsr absoluto?"
        """
        results = defaultdict(list)
        
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            topics = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})

            # Verificamos as duas condições possíveis para TSR
            has_tsr_absoluto = "TSR_Absoluto" in topics
            has_tsr_relativo = 'tsr_relativo' in facts and facts['tsr_relativo'].get('presente', False)

            if has_tsr_absoluto:
                results['absoluto'].append(company)
            if has_tsr_relativo:
                results['relativo'].append(company)
            if has_tsr_absoluto and has_tsr_relativo:
                results['ambos'].append(company)
            if has_tsr_absoluto or has_tsr_relativo:
                results['qualquer'].append(company)

        target_companies = results.get(tsr_type, [])

        if not target_companies:
            return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type}'.", None
            
        report_text = f"Encontradas **{len(target_companies)}** empresas com o critério de TSR: **{tsr_type.upper()}**."
        df = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type.upper()})"])

        return report_text, df

    def _find_companies_by_general_topic(self, query: str):
        """
        Busca genérica por um tópico para perguntas simples como "Quais empresas têm Malus e Clawback?".
        """
        found_topic = None
        # Itera sobre nossa base de conhecimento para encontrar o tópico mencionado na query
        for section_name, topics in self.kb.items():
            for topic_name, aliases in topics.items():
                # Inclui o próprio nome do tópico na busca
                search_terms = aliases + [topic_name.replace('_', ' ')]
                for term in search_terms:
                    if re.search(r'\b' + re.escape(term.lower()) + r'\b', query):
                        found_topic = (section_name, topic_name)
                        break
                if found_topic: break
            if found_topic: break
        
        if not found_topic:
            return "Não consegui identificar um tópico técnico conhecido na sua pergunta para realizar a busca agregada.", None

        section, topic = found_topic
        companies = []
        for company_name, details in self.data.items():
            if section in details.get("topicos_encontrados", {}) and topic in details["topicos_encontrados"][section]:
                companies.append(company_name)

        if not companies:
            return f"Nenhuma empresa encontrada com o tópico '{topic.replace('_', ' ')}'.", None

        report_text = f"Encontradas **{len(companies)}** empresas com o tópico: **{topic.replace('_', ' ')}**"
        df = pd.DataFrame(sorted(companies), columns=[f"Empresas com {topic.replace('_', ' ')}"])

        return report_text, df|
