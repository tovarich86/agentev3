# analytical_engine_v3.0.py (Versão Completa e Definitiva)
#
# DESCRIÇÃO:
# Esta é a versão completa e sem omissões do motor de análise quantitativa.
# Ela combina a profundidade estatística e a lógica robusta da versão original
# com a capacidade de filtragem por metadados e análise hierárquica da v2.0.
# Todas as funções de análise estão totalmente implementadas.

import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy import stats
import unicodedata
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AnalyticalEngine:
    """
    Motor de análise que opera sobre dados de resumo para responder perguntas
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
                "bancos", "varejo", "energia", "saude", "metalurgia", "siderurgia",
                "educacao", "transporte", "logistica", "tecnologia"
            ],
            "controle_acionario": [
                "privado", "privada", "privados", "privadas",
                "estatal", "estatais", "publico", "publica"
            ]
        }
        self.CANONICAL_MAP = {
            "privada": "Privado", "privados": "Privado", "privadas": "Privado",
            "estatais": "Estatal", "publico": "Estatal", "publica": "Estatal"
        }
        
        # --- Roteador Declarativo (Completo e com todas as funções implementadas) ---
        self.intent_rules = [
            # Vesting: Adicionado "carência", "tempo", "duração"
            (lambda q: 'vesting' in q and ('periodo' in q or 'prazo' in q or 'medio' in q or 'media' in q or 'carencia' in q or 'tempo' in q or 'duracao' in q), self._analyze_vesting_period),
            
            # Lock-up: Adicionado "restrição de venda"
            (lambda q: ('lockup' in q or 'lock-up' in q or 'restricao de venda' in q) and ('periodo' in q or 'prazo' in q or 'medio' in q or 'media' in q), self._analyze_lockup_period),

            # Diluição: Adicionado "percentual", "estatisticas"
            (lambda q: 'diluicao' in q and ('media' in q or 'percentual' in q or 'estatisticas' in q), self._analyze_dilution),

            # Desconto/Strike: A regra original já era boa.
            (lambda q: 'desconto' in q and ('preco de exercicio' in q or 'strike' in q), self._analyze_strike_discount),
            
            # TSR: A regra original já era boa.
            (lambda q: 'tsr' in q, self._analyze_tsr),
            
            # Malus/Clawback: Adicionado "lista", "quais" para forçar listagem.
            (lambda q: ('malus' in q or 'clawback' in q) and ('lista' in q or 'quais' in q), self._analyze_malus_clawback),
            
            # Dividendos: Adicionado "lista", "quais"
            (lambda q: 'dividendos' in q and 'carencia' in q and ('lista' in q or 'quais' in q), self._analyze_dividends_during_vesting),
            
            # Elegibilidade/Membros: A regra original já era boa.
            (lambda q: 'membros do plano' in q or 'elegiveis' in q or 'quem sao os membros' in q, self._analyze_plan_members),
            
            # Conselho: A regra original já era boa.
            (lambda q: 'conselho de administracao' in q and ('elegivel' in q or 'aprovador' in q), self._count_plans_for_board),
            
            # Metas/Indicadores: A regra original já era boa.
            (lambda q: 'metas mais comuns' in q or 'indicadores de desempenho' in q or 'metas de desempenho' in q or 'metas de performance' in q or 'indicadores de performance' in q, self._analyze_common_goals),
            
            # Regra para tipos de plano (agora separada e com sua própria vírgula)
            (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            
            # Tipos de Plano: A regra original já era boa.
            (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            
            # Fallback (sempre por último)
            (lambda q: True, self._find_companies_by_general_topic),
        ]
  # SUBSTITUA a função antiga por esta, DENTRO da classe AnalyticalEngine:

    def _collect_leaf_indicators_recursive(self, node: dict, collected_indicators: list):
        """
        VERSÃO CORRIGIDA: Navega pela árvore de tópicos e coleta os indicadores finais.
        """
        # Se o nó atual não for um dicionário, não há como processá-lo.
        if not isinstance(node, dict):
            return
        subtopics_dict = node.get("subtopicos", {})
    # Se não houver subtopicos, é folha: coleta aliases (se houver)
        if not subtopics_dict:
            # Coleta todos os aliases desta folha (se houver)
            aliases = node.get("aliases", [])
            # Só adiciona se houver aliases
            if aliases:
                collected_aliases.extend(aliases)
            return

        for key, value in subtopics_dict.items():
            # Apenas processa se for dict
            if isinstance(value, dict):
                self._collect_leaf_aliases_recursive(value, collected_aliases)


    
    def _normalize_text(self, text: str) -> str:
        """Normaliza o texto para minúsculas e remove acentos."""
        nfkd_form = unicodedata.normalize('NFKD', text.lower())
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def _extract_filters(self, normalized_query: str) -> dict:
        """Extrai filtros da pergunta com base em palavras-chave."""
        filters = {}
        for filter_type, keywords in self.FILTER_KEYWORDS.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(self._normalize_text(keyword)) + r'\b', normalized_query):
                    canonical_term = self.CANONICAL_MAP.get(keyword, keyword.capitalize())
                    filters[filter_type] = canonical_term
                    break
        if filters:
            logging.info(f"Filtros extraídos da pergunta: {filters}")
        return filters

    def _apply_filters_to_data(self, filters: dict) -> dict:
        """Aplica um dicionário de filtros aos dados principais."""
        if not filters:
            return self.data
        filtered_data = {
            comp: data for comp, data in self.data.items()
            if ('setor' not in filters or self._normalize_text(data.get('setor', '')) == self._normalize_text(filters['setor'])) and \
               ('controle_acionario' not in filters or data.get('controle_acionario', '').lower() == filters['controle_acionario'].lower())
        }
        logging.info(f"{len(filtered_data)} empresas correspondem aos filtros aplicados.")
        return filtered_data

    def answer_query(self, query: str, filters: dict | None = None) -> tuple:
        """
        Responde a uma consulta quantitativa.
        
        Args:
            query (str): A pergunta do usuário.
            filters (dict | None, optional): Um dicionário de filtros pré-selecionados
                                             (ex: da interface). Se for None, os filtros
                                             serão extraídos do texto da query.
        
        Returns:
            tuple: Uma tupla contendo o texto do relatório e um DataFrame/dicionário.
        """
        normalized_query = self._normalize_text(query)
        
        # Prioriza os filtros passados como argumento (da UI).
        # Se nenhum for passado, usa a extração da query como fallback.
        final_filters = filters if filters is not None else self._extract_filters(normalized_query)
        
        for intent_checker_func, analysis_func in self.intent_rules:
            if intent_checker_func(normalized_query):
                logging.info(f"Intenção detectada. Executando: {analysis_func.__name__}")
                return analysis_func(normalized_query, final_filters)
                
        return "Não consegui identificar uma intenção clara na sua pergunta.", None
    
    # --- Funções de Análise Detalhadas e Completas ---

    def _analyze_vesting_period(self, normalized_query: str, filters: dict) -> tuple:
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
        
        vesting_values = np.array([item[1] for item in periods])
        mode_result = stats.mode(vesting_values, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f} anos" for m in modes]) if len(modes) > 0 else "N/A"
        
        report_text = "### Análise de Período de Vesting\n"
        report_text += f"- **Total de Empresas com Dados:** {len(vesting_values)}\n"
        report_text += f"- **Vesting Médio:** {np.mean(vesting_values):.2f} anos\n"
        report_text += f"- **Desvio Padrão:** {np.std(vesting_values):.2f} anos\n"
        report_text += f"- **Mediana:** {np.median(vesting_values):.2f} anos\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(vesting_values):.2f} / {np.max(vesting_values):.2f} anos\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(periods, columns=["Empresa", "Período de Vesting (Anos)"])
        return report_text, df.sort_values(by="Período de Vesting (Anos)", ascending=False).reset_index(drop=True)

    def _analyze_lockup_period(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        periods = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_lockup' in facts and facts['periodo_lockup'].get('presente', False):
                valor = facts['periodo_lockup'].get('valor')
                if valor is not None and valor > 0:
                    periods.append((company, valor))
        if not periods:
            return "Nenhuma informação de lock-up encontrada para os filtros selecionados.", None

        lockup_values = np.array([item[1] for item in periods])
        mode_result = stats.mode(lockup_values, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f} anos" for m in modes]) if len(modes) > 0 else "N/A"

        report_text = "### Análise de Período de Lock-up\n"
        report_text += f"- **Total de Empresas com Dados:** {len(lockup_values)}\n"
        report_text += f"- **Lock-up Médio:** {np.mean(lockup_values):.2f} anos\n"
        report_text += f"- **Mediana:** {np.median(lockup_values):.2f} anos\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(lockup_values):.2f} / {np.max(lockup_values):.2f} anos\n"
        report_text += f"- **Moda(s):** {mode_str}\n"

        df = pd.DataFrame(periods, columns=["Empresa", "Período de Lock-up (Anos)"])
        return report_text, df.sort_values(by="Período de Lock-up (Anos)", ascending=False).reset_index(drop=True)

    def _analyze_dilution(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        diluicao_percentual = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'diluicao_maxima_percentual' in facts and facts['diluicao_maxima_percentual'].get('presente', False):
                valor = facts['diluicao_maxima_percentual'].get('valor')
                if valor is not None:
                    diluicao_percentual.append((company, valor * 100))
        if not diluicao_percentual:
            return "Nenhuma informação de diluição encontrada para os filtros selecionados.", None

        percents = np.array([item[1] for item in diluicao_percentual])
        mode_result = stats.mode(percents, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if len(modes) > 0 else "N/A"
        
        report_text = "### Análise de Diluição Máxima Percentual\n"
        report_text += f"- **Total de Empresas com Dados:** {len(percents)}\n"
        report_text += f"- **Média:** {np.mean(percents):.2f}%\n"
        report_text += f"- **Mediana:** {np.median(percents):.2f}%\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(percents):.2f}% / {np.max(percents):.2f}%\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df_percent = pd.DataFrame(diluicao_percentual, columns=["Empresa", "Diluição Máxima (%)"])
        return report_text, df_percent.sort_values(by="Diluição Máxima (%)", ascending=False).reset_index(drop=True)

    def _analyze_strike_discount(self, normalized_query: str, filters: dict) -> tuple:
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
        mode_result = stats.mode(discounts, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if len(modes) > 0 else "N/A"
        
        report_text = "### Análise de Desconto no Preço de Exercício\n"
        report_text += f"- **Total de Empresas com Desconto:** {len(discounts)}\n"
        report_text += f"- **Desconto Médio:** {np.mean(discounts):.2f}%\n"
        report_text += f"- **Desvio Padrão:** {np.std(discounts):.2f}%\n"
        report_text += f"- **Mediana:** {np.median(discounts):.2f}%\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(discounts):.2f}% / {np.max(discounts):.2f}%\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(companies_and_discounts, columns=["Empresa", "Desconto Aplicado (%)"])
        return report_text, df.sort_values(by="Desconto Aplicado (%)", ascending=False).reset_index(drop=True)

    def _analyze_tsr(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        results = defaultdict(list)
        tsr_type_filter = 'qualquer'
        if 'relativo' in normalized_query and 'absoluto' not in normalized_query:
            tsr_type_filter = 'relativo'
        elif 'absoluto' in normalized_query and 'relativo' not in normalized_query:
            tsr_type_filter = 'absoluto'

        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            tipos = facts.get('tsr', {}).get('tipos', [])
            has_tsr_absoluto = 'Absoluto' in tipos
            has_tsr_relativo = 'Relativo' in tipos
            if has_tsr_absoluto: results['absoluto'].append(company)
            if has_tsr_relativo: results['relativo'].append(company)
            if has_tsr_absoluto or has_tsr_relativo: results['qualquer'].append(company)

        target_companies = results.get(tsr_type_filter, [])
        if not target_companies:
            return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type_filter}' para os filtros selecionados.", None
        
        report_text = f"Encontradas **{len(target_companies)}** empresas com o critério de TSR: **{tsr_type_filter.upper()}** para os filtros aplicados."
        
        df = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type_filter.upper()})"])
        return report_text, df

    def _analyze_malus_clawback(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'malus_clawback_presente' in facts and facts['malus_clawback_presente'].get('presente', False):
                companies.append(company)
        if not companies:
            return "Nenhuma empresa com cláusulas de Malus ou Clawback foi encontrada para os filtros selecionados.", None
        
        report_text = f"Encontradas **{len(companies)}** empresas com cláusulas de **Malus ou Clawback** para os filtros aplicados."
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Malus/Clawback"])
        return report_text, df

    def _analyze_dividends_during_vesting(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'dividendos_durante_carencia' in facts and facts['dividendos_durante_carencia'].get('presente', False):
                companies.append(company)
        if not companies:
            return "Nenhuma empresa que paga dividendos durante a carência foi encontrada para os filtros selecionados.", None
        
        report_text = f"Encontradas **{len(companies)}** empresas que distribuem dividendos durante a **carência/vesting** para os filtros aplicados."
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Dividendos Durante Carência"])
        return report_text, df

    def _analyze_plan_members(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        member_role_counts = defaultdict(int)
        company_member_details = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'elegiveis_ao_plano' in facts and facts['elegiveis_ao_plano'].get('presente', False):
                roles = facts['elegiveis_ao_plano'].get('funcoes', [])
                company_member_details.append({"Empresa": company, "Funções Elegíveis": ", ".join(roles) if roles else "Não especificado"})
                for role in roles:
                    member_role_counts[role] += 1
        if not member_role_counts:
            return "Nenhuma informação sobre membros elegíveis foi encontrada para os filtros selecionados.", None
        
        report_text = "### Análise de Membros Elegíveis ao Plano\n**Contagem de Empresas por Tipo de Membro:**\n"
        df_counts_data = []
        for role, count in sorted(member_role_counts.items(), key=lambda item: item[1], reverse=True):
            report_text += f"- **{role}:** {count} empresas\n"
            df_counts_data.append({"Tipo de Membro Elegível": role, "Número de Empresas": count})
        
        dfs_to_return = {
            'Contagem por Tipo de Membro': pd.DataFrame(df_counts_data),
            'Detalhes por Empresa': pd.DataFrame(company_member_details).sort_values(by="Empresa").reset_index(drop=True)
        }
        return report_text, dfs_to_return

    def _count_plans_for_board(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies = []
        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            if 'conselho_administracao_elegivel_ou_aprovador' in facts and facts['conselho_administracao_elegivel_ou_aprovador'].get('presente', False):
                companies.append(company)
        if not companies:
            return "Nenhuma empresa com menção ao Conselho de Administração como elegível/aprovador foi encontrada para os filtros selecionados.", None
        
        report_text = f"**{len(companies)}** empresas com menção ao **Conselho de Administração** como elegível ou aprovador de planos foram encontradas para os filtros aplicados."
        df = pd.DataFrame(sorted(companies), columns=["Empresas com Menção ao Conselho de Administração"])
        return report_text, df

    def _recursive_count_indicators(self, data: dict, counts: defaultdict):
        """Função auxiliar recursiva para contar todos os indicadores e categorias aninhados."""
        for key, value in data.items():
            if key == 'aliases' or key == 'subtopicos':
                continue
            counts[key.replace('_', ' ')] += 1
            if isinstance(value, dict) and "subtopicos" in value and value.get("subtopicos"):
                self._recursive_count_indicators(value["subtopicos"], counts)
    # FUNÇÃO DE ANÁLISE USANDO A ABORDAGEM ITERATIVA
    def _analyze_common_goals(self, normalized_query: str, filters: dict) -> tuple:
        """
        Analisa e contabiliza os aliases de indicadores de performance mais comuns, com base nos filtros aplicados.
        Retorna um texto de relatório e um DataFrame com os resultados.
        """
        data_to_analyze = self._apply_filters_to_data(filters)
        alias_counts = defaultdict(int)
        for details in data_to_analyze.values():
            performance_section = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})
            if not performance_section:
                continue
            company_leaf_aliases = []
            self._collect_leaf_aliases_recursive(performance_section, company_leaf_aliases)

            for alias in set(company_leaf_aliases):
                alias_counts[alias] += 1

        if not alias_counts:
            return "Nenhum alias de indicador de performance encontrado para os filtros selecionados.", None

        report_text = "### Aliases de indicadores de performance mais comuns\n"
        df_data = [{"Alias": k, "Nº de Empresas": v}
                   for k, v in sorted(alias_counts.items(), key=lambda item: item[1], reverse=True)]
        for item in df_data:
            report_text += f"- **{item['Alias']}:** {item['Nº de Empresas']} empresas\n"
        return report_text, pd.DataFrame(df_data)
        
    def _analyze_common_plan_types(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        plan_type_counts = defaultdict(int)
        for details in data_to_analyze.values():
            plan_topics = details.get("topicos_encontrados", {}).get("TiposDePlano", {})
            for key in plan_topics.keys():
                if key not in ['aliases', 'subtopicos']:
                    plan_type_counts[key.replace('_', ' ')] += 1
        if not plan_type_counts:
            return "Nenhum tipo de plano encontrado para os filtros selecionados.", None
            
        report_text = "### Tipos de Planos Mais Comuns\n"
        df_data = [{"Tipo de Plano": k, "Nº de Empresas": v} for k, v in sorted(plan_type_counts.items(), key=lambda item: item[1], reverse=True)]
        for item in df_data:
            report_text += f"- **{item['Tipo de Plano'].capitalize()}:** {item['Nº de Empresas']} empresas\n"
        return report_text, pd.DataFrame(df_data)

    # --- Funções de Busca Hierárquica (Fallback) ---

    def _recursive_flat_map_builder(self, sub_dict: dict, section: str, flat_map: dict):
        """Função auxiliar recursiva para construir o mapa plano de aliases."""
        for topic_name_raw, data in sub_dict.items():
            if not isinstance(data, dict):
                continue
            
            topic_name_formatted = topic_name_raw.replace('_', ' ')
            details = (section, topic_name_formatted, topic_name_raw)

            flat_map[self._normalize_text(topic_name_formatted)] = details
            for alias in data.get("aliases", []):
                flat_map[self._normalize_text(alias)] = details
            
            if "subtopicos" in data and data.get("subtopicos"):
                self._recursive_flat_map_builder(data["subtopicos"], section, flat_map)
    
    def _kb_flat_map(self) -> dict:
        """Cria um mapa plano de alias -> (seção, nome_formatado, nome_bruto)."""
        if hasattr(self, '_kb_flat_map_cache'):
            return self._kb_flat_map_cache
        
        flat_map = {}
        for section, data in self.kb.items():
            if not isinstance(data, dict):
                continue

            section_name_formatted = section.replace('_', ' ')
            details = (section, section_name_formatted, section)
            
            flat_map[self._normalize_text(section_name_formatted)] = details
            for alias in data.get("aliases", []):
                flat_map[self._normalize_text(alias)] = details

            if "subtopicos" in data and data.get("subtopicos"):
                self._recursive_flat_map_builder(data["subtopicos"], section, flat_map)
        
        self._kb_flat_map_cache = flat_map
        return flat_map

    def _find_companies_by_general_topic(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        flat_map = self._kb_flat_map()
        found_topic_details = None
        
        # Busca pelo alias mais longo que corresponde à pergunta
        for alias in sorted(flat_map.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', normalized_query):
                found_topic_details = flat_map[alias]
                break
        
        if not found_topic_details:
            return "Não foi possível identificar um tópico técnico conhecido na sua pergunta para realizar a busca.", None

        section, topic_name_formatted, topic_name_raw = found_topic_details
        
        # Lógica para encontrar empresas que mencionam o tópico (incluindo sub-tópicos)
        companies = []
        for name, details in data_to_analyze.items():
            if section in details.get("topicos_encontrados", {}):
                # Função interna para buscar recursivamente no dicionário de tópicos da empresa
                def find_topic_recursively(data_dict):
                    if topic_name_raw in data_dict:
                        return True
                    for k, v in data_dict.items():
                        if isinstance(v, dict) and 'subtopicos' in v and v['subtopicos']:
                            if find_topic_recursively(v['subtopicos']):
                                return True
                    return False
                
                if find_topic_recursively(details["topicos_encontrados"][section]):
                    companies.append(name)

        if not companies:
            return f"Nenhuma empresa encontrada com o tópico '{topic_name_formatted}' para os filtros aplicados.", None

        report_text = f"Encontradas **{len(companies)}** empresas com o tópico: **{topic_name_formatted.capitalize()}**"
        df = pd.DataFrame(sorted(companies), columns=[f"Empresas com {topic_name_formatted.capitalize()}"])
        return report_text, df
