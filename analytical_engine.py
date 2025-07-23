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
            (lambda q: 'metas mais comuns' in q or 'indicadores de desempenho' in q or 'metas de desempenho' in q or 'metas de performance' in q or
             'indicadores de performance' in q, self._analyze_common_goals), (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            
            # Tipos de Plano: A regra original já era boa.
            (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            
            # Fallback (sempre por último)
            (lambda q: True, self._find_companies_by_general_topic),
        ]
    def _recursive_flat_map_builder(self, sub_dict: dict, section: str, flat_map: dict):
        """Função auxiliar recursiva para construir o mapa plano de aliases."""
        for topic_name_raw, data in sub_dict.items():
            if not isinstance(data, dict):
                continue
            
            topic_name_formatted = topic_name_raw.replace('_', ' ')
            details = (section, topic_name_formatted, topic_name_raw)

            # Mapeia o nome canônico do tópico
            flat_map[self._normalize_text(topic_name_formatted)] = details
            # Mapeia todos os aliases definidos
            for alias in data.get("aliases", []):
                flat_map[self._normalize_text(alias)] = details
            
            # Continua a recursão para os sub-tópicos
            if "subtopicos" in data and data.get("subtopicos"):
                self._recursive_flat_map_builder(data["subtopicos"], section, flat_map)


    def _recursive_count_indicators(self, data: dict, counts: defaultdict):
        """
        Função auxiliar recursiva para contar todos os indicadores e categorias aninhados,
        navegando pela estrutura de 'subtopicos'.
        """
        for key, value in data.items():
            # Conta o indicador ou categoria atual
            counts[key.replace('_', ' ')] += 1
            
            # Se o valor for um dicionário e contiver 'subtopicos', continua a recursão
            if isinstance(value, dict) and "subtopicos" in value and value["subtopicos"]:
                self._recursive_count_indicators(value["subtopicos"], counts)
                
    def _normalize_text(self, text: str) -> str:
        nfkd_form = unicodedata.normalize('NFKD', text.lower())
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def _extract_filters(self, normalized_query: str) -> dict:
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
        
        # --- LÓGICA DE FILTRO CORRIGIDA ---
        # Prioriza os filtros passados como argumento (da UI).
        # Se nenhum for passado, usa a extração da query como fallback.
        final_filters = filters if filters is not None else self._extract_filters(normalized_query)
        
        for intent_checker_func, analysis_func in self.intent_rules:
            if intent_checker_func(normalized_query):
                logging.info(f"Intenção detectada. Executando: {analysis_func.__name__}")
                # Passa os filtros corretos para a função de análise
                return analysis_func(normalized_query, final_filters)
                
        return "Não consegui identificar uma intenção clara na sua pergunta.", None
      
    # --- Funções de Análise Detalhadas (Lógica Completa Restaurada e Adaptada com Filtros) ---

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
         # Garante que 'modes' seja sempre uma lista/array iterável
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
        if 'relativo' in normalized_query and 'absoluto' in normalized_query: tsr_type_filter = 'ambos'
        elif 'relativo' in normalized_query: tsr_type_filter = 'relativo'
        elif 'absoluto' in normalized_query: tsr_type_filter = 'absoluto'

        for company, details in data_to_analyze.items():
            facts = details.get("fatos_extraidos", {})
            tipos = facts.get('tsr', {}).get('tipos', [])
            has_tsr_absoluto = 'Absoluto' in tipos
            has_tsr_relativo = 'Relativo' in tipos
            if has_tsr_absoluto: results['absoluto'].append(company)
            if has_tsr_relativo: results['relativo'].append(company)
            if has_tsr_absoluto and has_tsr_relativo: results['ambos'].append(company)
            if has_tsr_absoluto or has_tsr_relativo: results['qualquer'].append(company)

        target_companies = results.get(tsr_type_filter, [])
        if not target_companies:
            return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type_filter}' para os filtros selecionados.", None
        
        report_text = f"Encontradas **{len(target_companies)}** empresas com o critério de TSR: **{tsr_type_filter.upper()}** para os filtros aplicados."
        
        dfs_to_return = {}
        df_companies = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type_filter.upper()})"])
        dfs_to_return['Empresas'] = df_companies

        if tsr_type_filter in ['relativo', 'ambos', 'qualquer']:
            tsr_details_list = []
            for company in target_companies:
                facts = data_to_analyze[company].get("fatos_extraidos", {})
                if 'Relativo' in facts.get('tsr', {}).get('tipos', []):
                    peer_group = ', '.join(facts['tsr'].get('peer_group', [])) or 'Não especificado'
                    indice = facts['tsr'].get('indice_comparacao', 'Não especificado')
                    tsr_details_list.append({"Empresa": company, "Peer Group": peer_group, "Índice de Comparação": indice})
            
            if tsr_details_list:
                df_details = pd.DataFrame(tsr_details_list)
                dfs_to_return['Detalhes TSR Relativo'] = df_details
        
        return report_text, dfs_to_return if len(dfs_to_return) > 1 else list(dfs_to_return.values())[0]

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
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if len(modes) > 0 else "N/A"
    
        

        report_text = "### Análise de Período de Vesting\n"
        report_text += f"- **Total de Empresas:** {len(vesting_values)}\n"
        report_text += f"- **Vesting Médio:** {np.mean(vesting_values):.2f} anos\n"
        report_text += f"- **Desvio Padrão:** {np.std(vesting_values):.2f} anos\n"
        report_text += f"- **Mediana:** {np.median(vesting_values):.2f} anos\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(vesting_values):.2f} / {np.max(vesting_values):.2f} anos\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(periods, columns=["Empresa", "Período de Vesting (Anos)"])
        return report_text, df.sort_values(by="Período de Vesting (Anos)", ascending=False)

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
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if len(modes) > 0 else "N/A"

        report_text = "### Análise de Período de Lock-up\n"
        report_text += f"- **Total de Empresas:** {len(lockup_values)}\n"
        report_text += f"- **Lock-up Médio:** {np.mean(lockup_values):.2f} anos\n"
        report_text += f"- **Mediana:** {np.median(lockup_values):.2f} anos\n"
        report_text += f"- **Mínimo / Máximo:** {np.min(lockup_values):.2f} / {np.max(lockup_values):.2f} anos\n"
        report_text += f"- **Moda(s):** {mode_str}\n"

        df = pd.DataFrame(periods, columns=["Empresa", "Período de Lock-up (Anos)"])
        return report_text, df.sort_values(by="Período de Lock-up (Anos)", ascending=False)

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

        report_text = "### Análise de Diluição Máxima Percentual\n"
        percents = np.array([item[1] for item in diluicao_percentual])
        mode_result = stats.mode(percents, keepdims=True)
        modes = mode_result.mode
        if not isinstance(modes, (list, np.ndarray)):
            modes = [modes]
        mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if len(modes) > 0 else "N/A"
        report_text += f"- **Total de Empresas:** {len(percents)}\n"
        report_text += f"- **Média:** {np.mean(percents):.2f}%\n"
        report_text += f"- **Mediana:** {np.median(percents):.2f}%\n"
        report_text += f"- **Moda(s):** {mode_str}\n"
        
        df_percent = pd.DataFrame(diluicao_percentual, columns=["Empresa", "Diluição Máxima (%)"])
        return report_text, df_percent.sort_values(by="Diluição Máxima (%)", ascending=False).reset_index(drop=True)

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

    def _analyze_common_goals(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        indicator_counts = defaultdict(int)
        for details in data_to_analyze.values():
            # Pega a seção de performance completa
            performance_section = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})
            if performance_section:
                # Inicia a contagem recursiva a partir da seção principal
                self._recursive_count_indicators(performance_section, indicator_counts)
        for details in data_to_analyze.values():
            performance_section = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})
            for key in performance_section.keys():
                indicator_counts[key.replace('_', ' ')] += 1
        if not indicator_counts:
            return "Nenhum indicador de performance encontrado para os filtros selecionados.", None
        
        report_text = "### Indicadores de Performance Mais Comuns\n"
        df_data = [{"Indicador": k, "Nº de Empresas": v} for k, v in sorted(indicator_counts.items(), key=lambda item: item[1], reverse=True)]
        for item in df_data:
            report_text += f"- **{item['Indicador']}:** {item['Nº de Empresas']} empresas\n"
        return report_text, pd.DataFrame(df_data)

    def _analyze_common_plan_types(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        plan_type_counts = defaultdict(int)
        for details in data_to_analyze.values():
            plan_topics = details.get("topicos_encontrados", {}).get("TiposDePlano", {})
            for key in plan_topics.keys():
                plan_type_counts[key.replace('_', ' ')] += 1
        if not plan_type_counts:
            return "Nenhum tipo de plano encontrado para os filtros selecionados.", None
            
        report_text = "### Tipos de Planos Mais Comuns\n"
        df_data = [{"Tipo de Plano": k, "Nº de Empresas": v} for k, v in sorted(plan_type_counts.items(), key=lambda item: item[1], reverse=True)]
        for item in df_data:
            report_text += f"- **{item['Tipo de Plano']}:** {item['Nº de Empresas']} empresas\n"
        return report_text, pd.DataFrame(df_data)

    def _find_companies_by_general_topic(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        flat_map = self._kb_flat_map()
        found_topic_details = None
        for alias in sorted(flat_map.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(alias) + r'\b', normalized_query):
                found_topic_details = flat_map[alias]
                break
        
        if not found_topic_details:
            return "Não foi possível identificar um tópico técnico conhecido na sua pergunta para realizar a busca.", None

        section, topic_name_formatted, topic_name_raw = found_topic_details
        companies = [
            name for name, details in data_to_analyze.items()
            if section in details.get("topicos_encontrados", {}) and topic_name_raw in details["topicos_encontrados"][section]
        ]
        if not companies:
            return f"Nenhuma empresa encontrada com o tópico '{topic_name_formatted}' para os filtros aplicados.", None

        report_text = f"Encontradas **{len(companies)}** empresas com o tópico: **{topic_name_formatted}**"
        df = pd.DataFrame(sorted(companies), columns=[f"Empresas com {topic_name_formatted}"])
        return report_text, df
        
    def _kb_flat_map(self) -> dict:
        """
        Cria um mapa plano de alias -> (seção, nome_formatado, nome_bruto)
        compatível com a nova estrutura de dicionário hierárquico.
        """
        if hasattr(self, '_kb_flat_map_cache'):
            return self._kb_flat_map_cache
        
        flat_map = {}
        for section, data in self.kb.items(): # ex: section = "IndicadoresPerformance"
            if not isinstance(data, dict):
                continue

            # Trata a própria seção principal como um tópico pesquisável
            section_name_formatted = section.replace('_', ' ')
            details = (section, section_name_formatted, section)
            
            # Adiciona o nome canônico da seção (ex: "indicadores performance")
            flat_map[self._normalize_text(section_name_formatted)] = details
            # Adiciona os aliases da seção (ex: "kpis", "metas")
            for alias in data.get("aliases", []):
                flat_map[self._normalize_text(alias)] = details

            # Inicia a recursão para os sub-tópicos, se existirem
            if "subtopicos" in data and data.get("subtopicos"):
                self._recursive_flat_map_builder(data["subtopicos"], section, flat_map)
        
        self._kb_flat_map_cache = flat_map
        return flat_map
