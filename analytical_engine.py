# analytical_engine.py
# Este script contém o motor de análise para perguntas quantitativas.

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
    Um motor de análise que opera sobre o JSON de resumo pré-processado
    para responder perguntas quantitativas e de listagem.
    Versão com análise estatística aprofundada e roteador declarativo.
    """
    # --- Métodos Auxiliares e de Análise (definidos ANTES de __init__) ---

    def _normalize_text(self, text: str) -> str:
        """Normaliza o texto para comparação (minúsculas, sem acentos)."""
        nfkd_form = unicodedata.normalize('NFKD', text.lower())
        return "".join([c for c in nfkode_form if not unicodedata.combining(c)])

    def _analyze_strike_discount(self, normalized_query: str) -> tuple:
        """
        Analisa o desconto no preço de exercício, fornecendo estatísticas e uma lista de empresas.
        """
        companies_and_discounts = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'desconto_strike_price' in facts and facts['desconto_strike_price'].get('presente', False):
                valor_numerico = facts['desconto_strike_price'].get('valor_numerico')
                if valor_numerico is not None:
                    companies_and_discounts.append((company, valor_numerico * 100)) # Converte para percentual

        if not companies_and_discounts:
            return "Nenhuma empresa com desconto explícito ou implícito no preço de exercício foi encontrada nas fontes analisadas.", None
            
        discounts = np.array([item[1] for item in companies_and_discounts])
        
        # Moda removida daqui
        # Removida: mode_result = stats.mode(discounts, keepdims=False)
        # Removida: modes = mode_result.mode
        # Removida: mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if modes.ndim > 0 and modes.size > 0 else "N/A"

        report_text = "### Análise de Desconto no Preço de Exercício\n"
        report_text += f"- **Total de Empresas com Desconto:** {len(discounts)}\n"
        report_text += f"- **Desconto Médio:** {np.mean(discounts):.2f}%\n"
        report_text += f"- **Desvio Padrão:** {np.std(discounts):.2f}%\n"
        report_text += f"- **Mínimo:** {np.min(discounts):.2f}%\n"
        report_text += f"- **25º Percentil:** {np.percentile(discounts, 25):.2f}%\n"
        report_text += f"- **Mediana (50º Percentil):** {np.median(discounts):.2f}%\n"
        report_text += f"- **75º Percentil:** {np.percentile(discounts, 75):.2f}%\n"
        report_text += f"- **Máximo:** {np.max(discounts):.2f}%\n"
        # Removida: report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(companies_and_discounts, columns=["Empresa", "Desconto Aplicado (%)"])
        df_sorted = df.sort_values(by="Desconto Aplicado (%)", ascending=False).reset_index(drop=True)
        
        return report_text, df_sorted

    def _analyze_tsr(self, normalized_query: str) -> tuple:
        """
        Analisa empresas com base em critérios de TSR (Absoluto, Relativo, Ambos).
        Retorna também detalhes do peer group e índice de comparação se disponíveis.
        """
        results = defaultdict(list)
        tsr_type_filter = 'qualquer' # Padrão: qualquer tipo de TSR
        if 'relativo' in normalized_query and 'absoluto' in normalized_query: tsr_type_filter = 'ambos'
        elif 'relativo' in normalized_query: tsr_type_filter = 'relativo'
        elif 'absoluto' in normalized_query: tsr_type_filter = 'absoluto'

        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            topics_perf = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})

            has_tsr_absoluto = "TSR_Absoluto" in topics_perf # Verifica se o tópico está presente
            has_tsr_relativo = 'tsr_relativo' in facts and facts['tsr_relativo'].get('presente', False)

            if has_tsr_absoluto: results['absoluto'].append(company)
            if has_tsr_relativo: results['relativo'].append(company)
            if has_tsr_absoluto and has_tsr_relativo: results['ambos'].append(company)
            if has_tsr_absoluto or has_tsr_relativo: results['qualquer'].append(company)

        target_companies = results.get(tsr_type_filter, [])
        if not target_companies: 
            return f"Nenhuma empresa encontrada com o critério de TSR '{tsr_type_filter}'.", None
        
        report_text = f"Encontradas **{len(target_companies)}** empresas com o critério de TSR: **{tsr_type_filter.upper()}**."
        
        dfs_to_return = {}
        df_companies = pd.DataFrame(sorted(target_companies), columns=[f"Empresas com TSR ({tsr_type_filter.upper()})"])
        dfs_to_return['Empresas'] = df_companies

        # Adiciona detalhes para TSR Relativo se aplicável e se houver dados
        if tsr_type_filter in ['relativo', 'ambos']:
            tsr_details_list = []
            for company in target_companies:
                facts = self.data[company].get("fatos_extraidos", {})
                if 'tsr_relativo' in facts and facts['tsr_relativo'].get('presente', False):
                    peer_group = ', '.join(facts['tsr_relativo'].get('peer_group', [])) if facts['tsr_relativo'].get('peer_group') else 'Não especificado'
                    indice = facts['tsr_relativo'].get('indice_comparacao', 'Não especificado')
                    tsr_details_list.append({"Empresa": company, "Peer Group": peer_group, "Índice de Comparação": indice})
            
            if tsr_details_list:
                df_details = pd.DataFrame(tsr_details_list)
                dfs_to_return['Detalhes TSR Relativo'] = df_details
        
        # Se apenas um DF for gerado, retorna-o diretamente. Se forem múltiplos, retorna o dicionário.
        return report_text, dfs_to_return if len(dfs_to_return) > 1 else (list(dfs_to_return.values())[0] if dfs_to_return else None)

    def _analyze_vesting_period(self, normalized_query: str) -> tuple:
        """
        Analisa o período de vesting médio e outras estatísticas (em anos).
        """
        periods = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_vesting' in facts and facts['periodo_vesting'].get('presente', False):
                valor = facts['periodo_vesting'].get('valor') # Já deve estar em anos
                if valor is not None and valor > 0:
                    periods.append((company, valor))

        if not periods:
            return "Nenhuma informação sobre período de vesting foi encontrada nas empresas analisadas.", None
        
        vesting_values = np.array([item[1] for item in periods])
        
        # Moda removida daqui
        # Removida: mode_result = stats.mode(vesting_values, keepdims=False)
        # Removida: modes = mode_result.mode
        # Removida: mode_str = ", ".join([f"{m:.2f} anos" for m in modes]) if modes.ndim > 0 and modes.size > 0 else "N/A"

        report_text = "### Análise de Período de Vesting\n"
        report_text += f"- **Total de Empresas com Vesting Mapeado:** {len(vesting_values)}\n"
        report_text += f"- **Vesting Médio:** {np.mean(vesting_values):.2f} anos\n"
        report_text += f"- **Desvio Padrão:** {np.std(vesting_values):.2f} anos\n"
        report_text += f"- **Mínimo:** {np.min(vesting_values):.2f} anos\n"
        report_text += f"- **25º Percentil:** {np.percentile(vesting_values, 25):.2f} anos\n"
        report_text += f"- **Mediana (50º Percentil):** {np.median(vesting_values):.2f} anos\n"
        report_text += f"- **75º Percentil:** {np.percentile(vesting_values, 75):.2f} anos\n"
        report_text += f"- **Máximo:** {np.max(vesting_values):.2f} anos\n"
        # Removida: report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(periods, columns=["Empresa", "Período de Vesting (Anos)"])
        df_sorted = df.sort_values(by="Período de Vesting (Anos)", ascending=False).reset_index(drop=True)
        
        return report_text, df_sorted

    def _analyze_lockup_period(self, normalized_query: str) -> tuple:
        """
        Analisa o período de lock-up médio e outras estatísticas (em anos).
        """
        periods = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'periodo_lockup' in facts and facts['periodo_lockup'].get('presente', False):
                valor = facts['periodo_lockup'].get('valor') # Já deve estar em anos
                if valor is not None and valor > 0:
                    periods.append((company, valor))

        if not periods:
            return "Nenhuma informação sobre período de lock-up foi encontrada nas empresas analisadas.", None
        
        lockup_values = np.array([item[1] for item in periods])
        
        # Moda removida daqui
        # Removida: mode_result = stats.mode(lockup_values, keepdims=False)
        # Removida: modes = mode_result.mode
        # Removida: mode_str = ", ".join([f"{m:.2f} anos" for m in modes]) if modes.ndim > 0 and modes.size > 0 else "N/A"

        report_text = "### Análise de Período de Lock-up\n"
        report_text += f"- **Total de Empresas com Lock-up Mapeado:** {len(lockup_values)}\n"
        report_text += f"- **Lock-up Médio:** {np.mean(lockup_values):.2f} anos\n"
        report_text += f"- **Desvio Padrão:** {np.std(lockup_values):.2f} anos\n"
        report_text += f"- **Mínimo:** {np.min(lockup_values):.2f} anos\n"
        report_text += f"- **25º Percentil:** {np.percentile(lockup_values, 25):.2f} anos\n"
        report_text += f"- **Mediana (50º Percentil):** {np.median(lockup_values):.2f} anos\n"
        report_text += f"- **75º Percentil:** {np.percentile(lockup_values, 75):.2f} anos\n"
        report_text += f"- **Máximo:** {np.max(lockup_values):.2f} anos\n"
        # Removida: report_text += f"- **Moda(s):** {mode_str}\n"
        
        df = pd.DataFrame(periods, columns=["Empresa", "Período de Lock-up (Anos)"])
        df_sorted = df.sort_values(by="Período de Lock-up (Anos)", ascending=False).reset_index(drop=True)
        
        return report_text, df_sorted

    def _analyze_dilution(self, normalized_query: str) -> tuple:
        """
        Analisa a diluição máxima em percentual e quantidade de ações, fornecendo estatísticas.
        """
        diluicao_percentual = []
        diluicao_quantidade = []

        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'diluicao_maxima_percentual' in facts and facts['diluicao_maxima_percentual'].get('presente', False):
                valor = facts['diluicao_maxima_percentual'].get('valor')
                if valor is not None:
                    diluicao_percentual.append((company, valor * 100)) # Converte para %
            if 'diluicao_maxima_quantidade_acoes' in facts and facts['diluicao_maxima_quantidade_acoes'].get('presente', False):
                valor = facts['diluicao_maxima_quantidade_acoes'].get('valor')
                if valor is not None:
                    diluicao_quantidade.append((company, valor))

        report_text = "### Análise de Diluição Máxima\n"
        dfs_to_return = {} # Para retornar múltiplos DataFrames no Streamlit

        if diluicao_percentual:
            percents = np.array([item[1] for item in diluicao_percentual])
            
            # Moda removida daqui
            # Removida: mode_result = stats.mode(percents, keepdims=False)
            # Removida: modes = mode_result.mode
            # Removida: mode_str = ", ".join([f"{m:.2f}%" for m in modes]) if modes.ndim > 0 and modes.size > 0 else "N/A"

            report_text += "\n#### Diluição Percentual\n"
            report_text += f"- **Total de Empresas com Diluição % Mapeada:** {len(percents)}\n"
            report_text += f"- **Média:** {np.mean(percents):.2f}%\n"
            report_text += f"- **Máximo:** {np.max(percents):.2f}%\n"
            # Removida: report_text += f"- **Moda(s):** {mode_str}\n"
            df_percent = pd.DataFrame(diluicao_percentual, columns=["Empresa", "Diluição Máxima (%)"])
            dfs_to_return['Diluição Percentual'] = df_percent.sort_values(by="Diluição Máxima (%)", ascending=False).reset_index(drop=True)
        else:
            report_text += "\n#### Diluição Percentual\n- Nenhuma informação de diluição percentual encontrada.\n"

        if diluicao_quantidade:
            quantities = np.array([item[1] for item in diluicao_quantidade])

            # Moda removida daqui
            # Removida: mode_result = stats.mode(quantities, keepdims=False)
            # Removida: modes = mode_result.mode
            # Removida: mode_str = ", ".join([f"{m:,.0f} ações" for m in modes]) if modes.ndim > 0 and modes.size > 0 else "N/A"

            report_text += "\n#### Diluição em Quantidade de Ações\n"
            report_text += f"- **Total de Empresas com Diluição por Qtd. de Ações Mapeada:** {len(quantities)}\n"
            report_text += f"- **Média:** {np.mean(quantities):,.0f} ações\n"
            report_text += f"- **Máximo:** {np.max(quantities):,.0f} ações\n"
            # Removida: report_text += f"- **Moda(s):** {mode_str}\n"
            df_quantity = pd.DataFrame(diluicao_quantidade, columns=["Empresa", "Diluição Máxima (Ações)"])
            dfs_to_return['Diluição Quantidade'] = df_quantity.sort_values(by="Diluição Máxima (Ações)", ascending=False).reset_index(drop=True)
        else:
            report_text += "\n#### Diluição em Quantidade de Ações\n- Nenhuma informação de diluição por quantidade de ações encontrada.\n"
        
        if not diluicao_percentual and not diluicao_quantidade:
            return "Nenhuma informação de diluição máxima encontrada nas empresas analisadas.", None

        # Retorna um dicionário de DataFrames se houver mais de um, caso contrário, retorna o DF único ou None
        return report_text, dfs_to_return if len(dfs_to_return) > 1 else (list(dfs_to_return.values())[0] if dfs_to_return else None)

    def _analyze_malus_clawback(self, normalized_query: str) -> tuple:
        """
        Conta quantas empresas possuem cláusulas de Malus ou Clawback.
        """
        companies_with_clause = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'malus_clawback_presente' in facts and facts['malus_clawback_presente'].get('presente', False):
                companies_with_clause.append(company)
        
        if not companies_with_clause:
            return "Nenhuma empresa mencionou cláusulas de Malus ou Clawback nos documentos analisados.", None
        
        report_text = f"Encontradas **{len(companies_with_clause)}** empresas que mencionam cláusulas de **Malus ou Clawback**."
        df = pd.DataFrame(sorted(companies_with_clause), columns=["Empresas com Malus/Clawback"])
        return report_text, df

    def _analyze_dividends_during_vesting(self, normalized_query: str) -> tuple:
        """
        Conta quantas empresas distribuem dividendos durante o período de carência/vesting.
        """
        companies_with_dividends_during_vesting = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'dividendos_durante_carencia' in facts and facts['dividendos_durante_carencia'].get('presente', False):
                companies_with_dividends_during_vesting.append(company)
        
        if not companies_with_dividends_during_vesting:
            return "Nenhuma empresa mencionou distribuição de dividendos durante o período de carência/vesting nos documentos analisados.", None
        
        report_text = f"Encontradas **{len(companies_with_dividends_during_vesting)}** empresas que distribuem dividendos durante o período de **carência/vesting**."
        df = pd.DataFrame(sorted(companies_with_dividends_during_vesting), columns=["Empresas com Dividendos Durante Carência"])
        return report_text, df

    def _analyze_plan_members(self, normalized_query: str) -> tuple:
        """
        Lista os tipos de membros elegíveis e a contagem de empresas para cada tipo.
        Fornece detalhes por empresa.
        """
        member_role_counts = defaultdict(int)
        company_member_details = [] # Para armazenar Empresa e Funções Elegíveis

        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            if 'elegiveis_ao_plano' in facts and facts['elegiveis_ao_plano'].get('presente', False):
                roles = facts['elegiveis_ao_plano'].get('funcoes', [])
                
                # Adiciona detalhes para o DataFrame de detalhes por empresa
                company_member_details.append({"Empresa": company, "Funções Elegíveis": ", ".join(roles) if roles else "Não especificado"})
                
                # Conta a ocorrência de cada função entre as empresas
                for role in roles:
                    member_role_counts[role] += 1 # Conta a ocorrência de cada função entre as empresas
        
        if not member_role_counts:
            return "Nenhuma informação sobre membros elegíveis ao plano foi encontrada nas empresas analisadas.", None
        
        report_text = "### Análise de Membros Elegíveis ao Plano\n"
        
        # Relatório de Contagem por Tipo de Membro
        report_text += "**Contagem de Empresas por Tipo de Membro Elegível:**\n"
        df_counts_data = []
        sorted_members = sorted(member_role_counts.items(), key=lambda item: item[1], reverse=True)
        for role, count in sorted_members:
            report_text += f"- **{role}:** {count} empresas\n"
            df_counts_data.append({"Tipo de Membro Elegível": role, "Número de Empresas": count})
        
        df_member_counts = pd.DataFrame(df_counts_data)
        
        # DataFrame de Detalhes por Empresa
        df_company_details = pd.DataFrame(company_member_details)
        df_company_details_sorted = df_company_details.sort_values(by="Empresa").reset_index(drop=True)
        
        # Retorna um dicionário de DataFrames para exibição no Streamlit
        return report_text, {'Contagem por Tipo de Membro': df_member_counts, 'Detalhes por Empresa': df_company_details_sorted}

    def _count_plans_for_board(self, normalized_query: str) -> tuple:
        """
        Identifica empresas onde o Conselho de Administração é explicitamente mencionado
        como elegível ou aprovador de planos no Item 8.4 (ou similar).
        """
        companies_with_board_plans_mention = []
        for company, details in self.data.items():
            facts = details.get("fatos_extraidos", {})
            # Verifica a presença do fato 'conselho_administracao_elegivel_ou_aprovador'
            if 'conselho_administracao_elegivel_ou_aprovador' in facts and facts['conselho_administracao_elegivel_ou_aprovador'].get('presente', False):
                companies_with_board_plans_mention.append(company)
        
        if not companies_with_board_plans_mention:
            return "Nenhuma empresa foi identificada com informações explícitas sobre o Conselho de Administração como elegível ou aprovador de planos nos documentos analisados.", None
        
        total_companies = len(companies_with_board_plans_mention)
        report_text = f"**{total_companies}** empresas foram identificadas com menção ao **Conselho de Administração** como elegível ou aprovador de planos de incentivo.\n\n"
        report_text += "É importante notar que esta análise reflete a *presença* dessa menção nos documentos, e não uma contagem exata do *número de planos* por empresa aprovados pelo Conselho, o que exigiria um nível de extração de dados mais granular e estruturado por plano individual.\n\n"
        
        df = pd.DataFrame(sorted(companies_with_board_plans_mention), columns=["Empresas com Menção ao Conselho de Administração em Planos"])
        return report_text, df

    def _analyze_common_goals(self, normalized_query: str) -> tuple:
        """
        Lista os indicadores de performance mais comuns entre as empresas.
        Esta versão simplificada conta todos os tópicos encontrados sob
        "IndicadoresPerformance" e os lista por frequência.
        """
        goal_counts = defaultdict(int)
        for company, details in self.data.items():
            performance_topics = details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})
            for topic_raw in performance_topics.keys():
                # Normaliza o nome do tópico para contagem (ex: 'TSR_Absoluto' -> 'TSR Absoluto')
                canonical_goal_name = topic_raw.replace('_', ' ')
                goal_counts[canonical_goal_name] += 1
        
        if not goal_counts:
            return "Nenhum indicador de performance foi encontrado nos planos analisados.", None
        
        report_text = "### Indicadores de Performance Mais Comuns\n"
        report_text += "Esta análise lista todos os indicadores de performance encontrados, ordenados por frequência.\n\n"
        
        df_data = []
        sorted_goals = sorted(goal_counts.items(), key=lambda item: item[1], reverse=True)
        for goal, count in sorted_goals:
            report_text += f"- **{goal}:** {count} empresas\n"
            df_data.append({"Indicador de Performance": goal, "Número de Empresas": count})
        
        df = pd.DataFrame(df_data)
        return report_text, df


    def _analyze_common_plan_types(self, normalized_query: str) -> tuple:
        """
        Lista os tipos de planos de incentivo mais comuns entre as empresas.
        """
        plan_type_counts = defaultdict(int)
        for company, details in self.data.items():
            plan_topics = details.get("topicos_encontrados", {}).get("TiposDePlano", {})
            for topic_raw in plan_topics.keys():
                # Normaliza o nome do tópico para contagem
                canonical_plan_name = topic_raw.replace('_', ' ')
                plan_type_counts[canonical_plan_name] += 1
        
        if not plan_type_counts:
            return "Nenhum tipo de plano de incentivo foi encontrado nos documentos analisados.", None
        
        report_text = "### Tipos de Planos de Incentivo Mais Comuns\n"
        report_text += "Esta análise considera os tipos de planos encontrados nos documentos.\n\n"
        
        df_data = []
        sorted_plan_types = sorted(plan_type_counts.items(), key=lambda item: item[1], reverse=True)
        for plan_type, count in sorted_plan_types:
            report_text += f"- **{plan_type}:** {count} empresas\n"
            df_data.append({"Tipo de Plano": plan_type, "Número de Empresas": count})
        
        df = pd.DataFrame(df_data)
        return report_text, df

    def _find_companies_by_general_topic(self, normalized_query: str) -> tuple:
        """
        Função de fallback que busca empresas associadas a qualquer tópico do dicionário
        se nenhuma intenção específica for detectada.
        """
        flat_map = self.kb_flat_map()
        found_topic_details = None
        # Prioriza matches mais longos e específicos
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
        """
        Cria um mapeamento plano de aliases para nomes canônicos de tópicos para busca rápida.
        Armazena em cache para performance.
        """
        if hasattr(self, '_kb_flat_map_cache'): return self._kb_flat_map_cache
        flat_map = {}
        for section, topics in self.kb.items():
            for topic_name_raw, aliases in topics.items():
                topic_name_formatted = topic_name_raw.replace('_', ' ')
                details = (section, topic_name_formatted, topic_name_raw)
                # Adiciona o nome canônico e todos os aliases ao mapeamento
                flat_map[topic_name_formatted.lower()] = details
                for alias in aliases: 
                    # Normaliza o alias antes de adicionar ao mapa para garantir correspondência consistente
                    flat_map[self._normalize_text(alias)] = details 
        self._kb_flat_map_cache = flat_map
        return flat_map

    # O construtor __init__ deve ser definido depois de todos os métodos que ele referencia
    def __init__(self, summary_data: dict, knowledge_base: dict):
        if not summary_data:
            raise ValueError("Dados de resumo (summary_data) não podem ser nulos.")
        self.data = summary_data
        self.kb = knowledge_base 
        
        # --- ROTEADOR DECLARATIVO ---
        # Mapeamento de funções de análise para suas regras de ativação.
        # As regras são verificadas em ordem. A primeira que corresponder é executada.
        self.intent_rules = [
            (lambda q: 'desconto' in q and ('preco de exercicio' in q or 'strike' in q), self._analyze_strike_discount),
            (lambda q: 'tsr' in q, self._analyze_tsr),
            (lambda q: 'vesting' in q and ('periodo' in q or 'prazo' in q or 'medio' in q), self._analyze_vesting_period),
            (lambda q: 'lockup' in q or 'lock-up' in q, self._analyze_lockup_period),
            (lambda q: 'diluicao' in q, self._analyze_dilution),
            (lambda q: 'malus' in q or 'clawback' in q, self._analyze_malus_clawback),
            (lambda q: 'dividendos' in q and 'carencia' in q, self._analyze_dividends_during_vesting),
            (lambda q: 'membros do plano' in q or 'elegiveis' in q or 'quem sao os membros' in q, self._analyze_plan_members),
            (lambda q: 'quantos planos tem o conselho de administracao elegivel' in q or 'conselho de administracao elegivel' in q, self._count_plans_for_board),
            (lambda q: 'metas mais comuns' in q or 'indicadores de desempenho' in q or 'metas de performance' in q, self._analyze_common_goals),
            (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            # Fallback para busca genérica por tópico se nenhuma regra específica for acionada
            (lambda q: True, self._find_companies_by_general_topic),
        ]
