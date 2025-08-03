# analytical_engine.py (v4 - Versão Final com Sumarização de Tópicos)

import numpy as np
import pandas as pd
import re
from collections import defaultdict
import unicodedata
import logging

logger = logging.getLogger(__name__)

# --- PONTO CENTRAL DA OTIMIZAÇÃO: Mapa de Fatos Quantitativos ---
# Mapeia aliases (termos de busca do usuário) para a chave exata no JSON 'fatos_extraidos'.
FACT_MAP = {
    "diluicao_maxima_percentual": {
        "aliases": ["diluição máxima", "diluicao maxima", "diluição"],
        "unit": "percentual"
    },
    "periodo_lockup": {
        "aliases": ["lock-up", "lockup", "período de lockup", "periodo de lockup", "restrição de venda"],
        "unit": "ano"
    },
    "periodo_vesting": {
        "aliases": ["vesting", "período de vesting", "prazo de vesting", "periodo de vesting", "carência"],
        "unit": "ano"
    },
    "desconto_strike_price": {
        "aliases": ["desconto no preço de exercício", "desconto strike", "desconto preco exercicio"],
        "unit": "percentual"
    },
}


class AnalyticalEngine:
    """
    Motor de análise com sumarização, unificação de buscas e análises comparativas.
    """
    def __init__(self, summary_data: dict, knowledge_base: dict):
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base
        
        self.STATISTIC_KEYWORDS = {alias: key for key, details in FACT_MAP.items() for alias in details["aliases"]}
        self.TOPIC_KEYWORDS = self._flatten_topics(knowledge_base)

    def _normalize_text(self, text: str) -> str:
        """Normaliza o texto para comparação (minúsculas, sem acentos)."""
        if not isinstance(text, str): return ""
        return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower().strip()

    def _flatten_topics(self, node: dict) -> dict:
        """
        Cria um mapa plano de 'alias_normalizado -> nome_canônico_do_tópico'
        a partir da base de conhecimento hierárquica.
        """
        flat_map = {}
        for key, value in node.items():
            if not isinstance(value, dict): continue
            
            # O nome canônico é a chave original, para manter a formatação
            canonical_name = key 
            
            # Adiciona a própria chave normalizada como um alias
            flat_map[self._normalize_text(key)] = canonical_name
            
            # Adiciona os aliases definidos
            for alias in value.get("_aliases", []):
                flat_map[self._normalize_text(alias)] = canonical_name
            
            # Busca recursivamente nos subtopicos
            if "subtopicos" in value:
                flat_map.update(self._flatten_topics(value["subtopicos"]))
        return flat_map

    def _parse_query(self, query: str) -> (str, dict, list):
        """
        Identifica intenção (estatística, listagem, comparação, sumarização) e múltiplos alvos.
        """
        normalized_query = self._normalize_text(query)
        intent, targets = None, []
        search_space = {**self.TOPIC_KEYWORDS, **self.STATISTIC_KEYWORDS}

        for keyword, canonical_name in search_space.items():
            # Usar limites de palavra para evitar correspondências parciais (ex: 'meta' em 'metade')
            if re.search(r'\b' + re.escape(keyword) + r'\b', normalized_query):
                targets.append({"name": canonical_name, "type": "fact" if canonical_name in FACT_MAP else "topic"})

        if not targets: return None, {}, []

        # Remove alvos duplicados que podem surgir de múltiplos aliases
        unique_targets = [dict(t) for t in {tuple(d.items()) for d in targets}]
        targets = unique_targets

        # Lógica para definir a intenção
        is_listing = any(word in normalized_query for word in ["quais", "liste", "lista de", "empresas com", "que tem"])
        is_comparison = "compare" in normalized_query or "vs" in normalized_query
        is_summarization = any(word in normalized_query for word in ["resumo de", "principais", "mais comuns", "quais sao os", "faça um resumo"])

        if is_summarization and len(targets) == 1 and targets[0]['type'] == 'topic':
            intent = "summarize_topics"
        elif is_listing:
            intent = "listing"
        elif is_comparison and len(targets) > 1:
            intent = "comparison"
        elif len(targets) == 1 and targets[0]['type'] == 'fact':
            intent = "statistic"
        else: # Se não for claro, o padrão é listar
            intent = "listing"

        filters = {} # A lógica de filtros pode ser adicionada aqui, se necessário
        return intent, filters, targets

    def _extract_numerical_facts(self, companies_data: dict, fact_key: str) -> list:
        """Função genérica para extrair valores de um fato quantitativo específico."""
        values = []
        for company_data in companies_data.values():
            for plan in company_data.get("planos_identificados", {}).values():
                facts = plan.get("fatos_extraidos", {})
                fact_info = facts.get(fact_key, {})
                if fact_info.get("presente"):
                    value = fact_info.get("valor")
                    if isinstance(value, (int, float)):
                        values.append(value)
        return values

    def _calculate_statistics(self, values: list, fact_key: str) -> (str, pd.DataFrame):
        """Calcula e formata estatísticas descritivas para uma lista de valores."""
        if not values: return f"Nenhuma informação de '{fact_key.replace('_', ' ')}' encontrada.", None
        df_data = {"Métrica": ["Contagem", "Média", "Mediana", "Mínimo", "Máximo", "Desvio Padrão"], "Valor": [len(values), np.mean(values), np.median(values), np.min(values), np.max(values), np.std(values)]}
        df = pd.DataFrame(df_data)
        unit = FACT_MAP.get(fact_key, {}).get("unit", "")
        if unit == "percentual":
            for metric in ["Média", "Mediana", "Mínimo", "Máximo", "Desvio Padrão"]:
                df.loc[df["Métrica"] == metric, "Valor"] = df.loc[df["Métrica"] == metric, "Valor"].apply(lambda x: f"{x:.2%}")
        else:
            df["Valor"] = df["Valor"].round(2)
        return f"Análise estatística para **{fact_key.replace('_', ' ').title()}**:", df

    def _find_companies_by_target(self, companies_data: dict, target: dict) -> (str, pd.DataFrame):
        """Função unificada para listar empresas por fato ou por tópico."""
        companies = set()
        target_name, target_type = target['name'], target['type']
        for name, company_data in companies_data.items():
            found = False
            for plan in company_data.get("planos_identificados", {}).values():
                if target_type == 'fact':
                    if target_name in plan.get("fatos_extraidos", {}) and plan["fatos_extraidos"][target_name].get("presente"):
                        found = True; break
                else: # type == 'topic'
                    if self._find_topic_in_node(plan.get("topicos_encontrados", {}), self._normalize_text(target_name)):
                        found = True; break
            if found: companies.add(name)
        target_display = target_name.replace('_', ' ').title()
        if not companies: return f"Nenhuma empresa encontrada com '{target_display}'.", None
        return f"Encontradas **{len(companies)}** empresas com: **{target_display}**", pd.DataFrame(sorted(list(companies)), columns=[f"Empresas com {target_display}"])

    def _find_topic_in_node(self, node: dict, topic_normalized: str) -> bool:
        """Função recursiva para encontrar um tópico na estrutura aninhada."""
        if isinstance(node, dict):
            for key, value in node.items():
                if self._normalize_text(key) == topic_normalized: return True
                if self._find_topic_in_node(value, topic_normalized): return True
        return False

    def _summarize_topic_usage(self, companies_data: dict, parent_topic_name: str) -> (str, pd.DataFrame):
        """
        Conta a ocorrência de todos os sub-tópicos dentro de uma categoria principal
        em todo o conjunto de dados.
        """
        indicator_counts = defaultdict(int)
        
        def find_parent_node(node, target_key_normalized):
            if not isinstance(node, dict): return None
            for key, value in node.items():
                if self._normalize_text(key) == target_key_normalized: return value
                found = find_parent_node(value.get("subtopicos", {}), target_key_normalized)
                if found is not None: return found
            return None

        def collect_indicators(node):
            indicators = []
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "_aliases": continue
                    # Adiciona a chave como um indicador se ela não tiver mais sub-tópicos
                    # ou se for uma folha da árvore.
                    indicators.append(key)
                    # Continua a busca recursiva
                    if isinstance(value, dict) and "subtopicos" in value:
                        indicators.extend(collect_indicators(value["subtopicos"]))
            return indicators

        for company_data in companies_data.values():
            for plan in company_data.get("planos_identificados", {}).values():
                parent_node = find_parent_node(plan.get("topicos_encontrados", {}), self._normalize_text(parent_topic_name))
                if parent_node:
                    found_indicators = collect_indicators(parent_node.get("subtopicos", {}))
                    for indicator in set(found_indicators):
                        indicator_counts[indicator] += 1
        
        if not indicator_counts:
            return f"Nenhum sub-tópico encontrado para a categoria '{parent_topic_name}'.", None

        df = pd.DataFrame(indicator_counts.items(), columns=['Indicador', 'Nº de Planos'])
        df = df.sort_values(by='Nº de Planos', ascending=False).reset_index(drop=True)
        
        report_text = f"Resumo de frequência para a categoria **{parent_topic_name.replace('_', ' ').title()}**:"
        return report_text, df

    def answer_question(self, query: str) -> list:
        """
        Ponto de entrada principal. Roteia a pergunta para a função correta
        e retorna uma lista de resultados (tuplas de texto e dataframe).
        """
        try:
            intent, filters, targets = self._parse_query(query)
            if not intent or not targets:
                return [("Não foi possível identificar um alvo claro (fato ou tópico) na sua pergunta.", None)]

            filtered_data = self.data
            results = []
            
            if intent == "summarize_topics":
                results.append(self._summarize_topic_usage(filtered_data, targets[0]['name']))
            else:
                for target in targets:
                    if intent == "statistic" and target['type'] == 'fact':
                        values = self._extract_numerical_facts(filtered_data, target['name'])
                        results.append(self._calculate_statistics(values, target['name']))
                    else: # "listing" or "comparison"
                        results.append(self._find_companies_by_target(filtered_data, target))
            return results
        except Exception as e:
            logger.error(f"Erro no AnalyticalEngine ao processar a query '{query}': {e}")
            return [(f"Ocorreu um erro ao processar sua pergunta: {e}", None)]
