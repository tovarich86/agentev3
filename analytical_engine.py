# analytical_engine.py (VERSÃO CORRIGIDA E FORMATADA)

import numpy as np
import pandas as pd
import re
from collections import defaultdict, deque
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
                "educacao", "transporte", "logistica", "tecnologia", "alimentos",
                "farmaceutico e higiene", "construcao civil", "telecomunicacoes",
                "intermediacao financeira", "seguradoras e corretoras",
                "extracaomineral", "textil e vestuario", "embalagens", "brinquedos e lazer",
                "hospedagem e turismo", "saneamento", "servicos agua e gas",
                "maquinas, equipamentos, veiculos e pecas", "petroleo e gas", "papel e celulose",
                "securitizacao de recebiveis", "reflorestamento", "arrendamento mercantil"
            ],
            "controle_acionario": [
                "privado", "privada", "privados", "privadas",
                "estatal", "estatais", "publico", "publica", "estrangeiro"
            ]
        }
        self.CANONICAL_MAP = {
            "privada": "Privado", "privados": "Privado", "privadas": "Privado",
            "estatais": "Estatal", "publico": "Estatal", "publica": "Estatal",
            "bancos": "Bancos", "varejo": "Comércio (Atacado e Varejo)", 
            "energia": "Energia Elétrica", "saude": "Serviços médicos", 
            "metalurgia": "Metalurgia e Siderurgia", "siderurgia": "Metalurgia e Siderurgia",
            "educacao": "Educação", "transporte": "Serviços Transporte e Logística",
            "logistica": "Serviços Transporte e Logística", "tecnologia": "Comunicação e Informática",
            "alimentos": "Alimentos", "farmaceutico e higiene": "Farmacêutico e Higiene",
            "construcao civil": "Construção Civil, Mat. Constr. e Decoração", "telecomunicacoes": "Telecomunicações",
            "intermediacao financeira": "Intermediação Financeira", "seguradoras e corretoras": "Seguradoras e Corretoras",
            "extracaomineral": "Extração Mineral", "textil e vestuario": "Têxtil e Vestuário", 
            "embalagens": "Embalagens", "brinquedos e lazer": "Brinquedos e Lazer",
            "hospedagem e turismo": "Hospedagem e Turismo", "saneamento": "Saneamento, Serv. Água e Gás",
            "servicos agua e gas": "Saneamento, Serv. Água e Gás",
            "maquinas, equipamentos, veiculos e pecas": "Máquinas, Equipamentos, Veículos e Peças",
            "petroleo e gas": "Petróleo e Gás", "papel e celulose": "Papel e Celulose",
            "securitizacao de recebiveis": "Securitização de Recebíveis", "reflorestamento": "Reflorestamento",
            "arrendamento mercantil": "Arrendamento Mercantil"
        }

        # --- Mapeamento Canônico de Indicadores e Categorias ---
        self.TERMOS_A_IGNORAR = [
            "performance",
            "indicadores de performance",
            "outros/genéricos",
            "grupos de comparação"
        ]

        self.INDICATOR_CANONICAL_MAP = {
            # Financeiro - Unificação
            'lucro': 'Lucro (Geral)',
            'lucro líquido': 'Lucro (Geral)',
            'ebitda': 'EBITDA',
            'adjusted ebitda': 'EBITDA',
            'fluxo de caixa': 'Fluxo de Caixa / FCF',
            'fcf': 'Fluxo de Caixa / FCF',
            'fluxo de caixa livre': 'Fluxo de Caixa / FCF',
            'roic': 'ROIC (Retorno sobre Capital Investido)',
            'retorno sobre o capital investido': 'ROIC (Retorno sobre Capital Investido)',
            'cagr': 'CAGR (Taxa de Crescimento Anual)',
            'cagr ebitda per share': 'CAGR (Taxa de Crescimento Anual)',
            'receita líquida': 'Receita',
            'receita operacional líquida': 'Receita',
            'receita operacional': 'Receita',
            'capital de giro': 'Capital de Giro',
            'margem bruta': 'Margens',
            'margem operacional': 'Margens',
            'eva': 'EVA (Valor Econômico Agregado)',
            'economic value added': 'EVA (Valor Econômico Agregado)',
            'valor econômico agregado': 'EVA (Valor Econômico Agregado)',
            'redução de dívida': 'Redução de Dívida / Alavancagem',
            'dívida financeira bruta': 'Redução de Dívida / Alavancagem',
            'wacc': 'WACC (Custo de Capital)',
            'weighted average capital cost': 'WACC (Custo de Capital)',
            'custo de capital': 'WACC (Custo de Capital)',
            'ev': 'EV (Enterprise Value)',
            'enterprise value': 'EV (Enterprise Value)',
            'nopat': 'NOPAT (Lucro Operacional s/ Impostos)',
            'net operating profit after tax': 'NOPAT (Lucro Operacional s/ Impostos)',
            'rentabilidade': 'Rentabilidade (Geral)',
            'retorno sobre ativo': 'Rentabilidade (Geral)',
            'despesas de capital': 'CAPEX',
        
            # Mercado - Unificação
            'ipca': 'IPCA (Inflação)',
            'tsr': 'TSR (Retorno Total ao Acionista)',
            'cdi': 'CDI (Taxa Interbancária)',
            'selic': 'Selic (Taxa Básica de Juros)',
            'equity value': 'Valor de Mercado / Equity',
            'valor teórico da companhia': 'Valor de Mercado / Equity',
            'valor teórico unitário da ação': 'Valor de Mercado / Equity',

            # ESG - Unificação
            'esg': 'ESG (Geral)',
            'esg (objetivos de desenvolvimento sustentável)': 'ESG (Geral)',
            'esg (meio ambiente)': 'ESG (Geral)',
            'esg (inclusão/diversidade)': 'ESG (Geral)',

            # Operacional - Unificação
            'qualidade': 'Qualidade (Operacional)',
            'crescimento': 'Crescimento (Operacional)',
            'produtividade': 'Produtividade (Operacional)',
            'desempenho de entrega': 'Desempenho de Entrega',
            'expansão comercial': 'Expansão Comercial',
            'conclusão de aquisições': 'M&A e Expansão',
            'desempenho de segurança': 'Segurança (Operacional)',
            'nps': 'NPS (Net Promoter Score)',
            'rotatividade do estoque': 'Eficiência de Ativos',
            'rotatividade de ativos líquidos': 'Eficiência de Ativos'
        }

        self.INDICATOR_CATEGORIES = {
            "Financeiro": [
                'Lucro (Geral)', 'EBITDA', 'Fluxo de Caixa / FCF', 'ROIC (Retorno sobre Capital Investido)',
                'CAGR (Taxa de Crescimento Anual)', 'Receita', 'Capital de Giro', 'Margens',
                'EVA (Valor Econômico Agregado)', 'Redução de Dívida / Alavancagem', 'WACC (Custo de Capital)',
                'EV (Enterprise Value)', 'NOPAT (Lucro Operacional s/ Impostos)', 'Rentabilidade (Geral)', 'CAPEX'
            ],
            "Mercado": [
                'IPCA (Inflação)', 'TSR (Retorno Total ao Acionista)', 'CDI (Taxa Interbancária)',
                'Selic (Taxa Básica de Juros)', 'Valor de Mercado / Equity'
            ],
            "ESG": [
                'ESG (Geral)'
            ],
            "Operacional": [
                'Qualidade (Operacional)', 'Crescimento (Operacional)', 'Produtividade (Operacional)',
                'Desempenho de Entrega', 'Expansão Comercial', 'M&A e Expansão',
                'Segurança (Operacional)', 'NPS (Net Promoter Score)', 'Eficiência de Ativos'
            ]
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
            (lambda q: 'metas mais comuns' in q or 'indicadores de desempenho' in q or 'metas de desempenho' in q or 'metas de performance' in q or 'indicadores de performance' in q or 'quais os indicadores mais comuns' in q, self._analyze_common_goals),
            
            # Regra para tipos de plano (agora separada e com sua própria vírgula)
            (lambda q: 'planos mais comuns' in q or 'tipos de plano mais comuns' in q, self._analyze_common_plan_types),
            
            # Fallback (sempre por último)
            (lambda q: True, self._find_companies_by_general_topic),
        ]

    def _recursive_flat_map_builder(self, sub_dict: dict, section: str, flat_map: dict):
        """Função auxiliar recursiva para construir o mapa plano de aliases."""
        for topic_name_raw, data in sub_dict.items():
            # Pula chaves de controle como '_aliases'
            if not isinstance(data, dict):
                continue
            
            topic_name_formatted = topic_name_raw.replace('_', ' ')
            details = (section, topic_name_formatted, topic_name_raw)

            # Mapeia o nome canônico do tópico e seus aliases
            flat_map[self._normalize_text(topic_name_formatted)] = details
            for alias in data.get("aliases", []):
                flat_map[self._normalize_text(alias)] = details
            
            # Continua a recursão para 'subtopicos', se existirem
            if "subtopicos" in data and data.get("subtopicos"):
                self._recursive_flat_map_builder(data["subtopicos"], section, flat_map)


    def _collect_leaf_aliases_recursive(self, node: dict or list, collected_aliases: list):
        """
        Versão Definitiva: Percorre a estrutura de dados e coleta tanto os aliases
        explícitos (em '_aliases') quanto os nomes dos indicadores que são usados
        como chaves de dicionário (ex: "TSR": {...}), garantindo uma contagem precisa
        e consistente em todas as análises.
        """
        if isinstance(node, list):
            for item in node:
                # Se o item da lista for um alias (string), coleta
                if isinstance(item, str):
                    collected_aliases.append(item)
                # Se for outro dicionário ou lista, continua a busca
                elif isinstance(item, (dict, list)):
                    self._collect_leaf_aliases_recursive(item, collected_aliases)
        
        elif isinstance(node, dict):
            # Itera sobre as chaves e valores do dicionário
            for key, value in node.items():
                # A própria CHAVE é um indicador, exceto as chaves de controle
                if key not in ["_aliases", "subtopicos"]:
                    collected_aliases.append(key)
                
                # Se o valor for a lista de aliases, adiciona todos
                if key == "_aliases" and isinstance(value, list):
                    collected_aliases.extend(value)
                # Se o valor for outro dicionário ou lista, continua a recursão
                elif isinstance(value, (dict, list)):
                    self._collect_leaf_aliases_recursive(value, collected_aliases)

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
        [VERSÃO DE DEPURAÇÃO] Responde a uma consulta quantitativa e mostra o processo 
        de decisão do roteador de intenção.
        """
        # Adicione 'import streamlit as st' no topo do seu arquivo analytical_engine.py
        import streamlit as st

        normalized_query = self._normalize_text(query)
        final_filters = filters if filters is not None else self._extract_filters(normalized_query)
        
        st.info("--- INICIANDO MODO DE DEPURAÇÃO: Roteador `answer_query` ---")
        st.write(f"**Query Original:** '{query}'")
        st.write(f"**Query Normalizada (usada para regras):** '{normalized_query}'")
        st.write("--- Verificando Regras de Intenção em Ordem ---")

        for i, (intent_checker_func, analysis_func) in enumerate(self.intent_rules):
            match_found = intent_checker_func(normalized_query)
            
            # Mostra o status de verificação de cada regra
            if match_found:
                st.success(f"**REGRA {i+1} (MATCH!):** A query correspondeu a esta regra. Executando `{analysis_func.__name__}`.")
                # Se a regra correta for encontrada, removemos os prints de debug e executamos a função
                st.empty() # Limpa a tela de debug para mostrar apenas o resultado
                return analysis_func(normalized_query, final_filters)
            else:
                st.warning(f"**REGRA {i+1} (NO MATCH):** A query não correspondeu à regra para `{analysis_func.__name__}`.")
                
        # Este retorno só acontecerá se nenhuma regra corresponder (o que é impossível devido ao fallback)
        return "Não consegui identificar uma intenção clara na sua pergunta.", None

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
        def find_tsr_recursively(node: dict or list) -> bool:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == 'TSR':
                        return True
                    if isinstance(value, (dict, list)) and find_tsr_recursively(value):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, (dict, list)) and find_tsr_recursively(item):
                        return True
            return False

        data_to_analyze = self._apply_filters_to_data(filters)
        companies_with_tsr = []

        for company, details in data_to_analyze.items():
            for plan_details in details.get("planos_identificados", {}).values():
                performance_section = plan_details.get("topicos_encontrados", {}).get("IndicadoresPerformance")
                
                if performance_section and find_tsr_recursively(performance_section):
                    companies_with_tsr.append(company)
                    break
        
        if not companies_with_tsr:
            return "Nenhuma empresa encontrada com o critério de TSR para os filtros selecionados.", None

        unique_companies = sorted(list(set(companies_with_tsr)))
        report_text = f"Encontradas **{len(unique_companies)}** empresas com o critério de TSR para os filtros aplicados."
        df = pd.DataFrame(unique_companies, columns=["Empresas com TSR"])
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
        company_member_details = defaultdict(set) # Usar set para evitar duplicatas

        for company, details in data_to_analyze.items():
            # Itera sobre cada plano identificado para a empresa
            for plan_name, plan_details in details.get("planos_identificados", {}).items():
                # Busca a seção de elegibilidade DENTRO de cada plano
                elegibility_section = plan_details.get("topicos_encontrados", {}).get("ParticipantesCondicoes", {}).get("Elegibilidade", [])
                
                if elegibility_section:
                    for role in elegibility_section:
                        # Adiciona a função elegível ao set da empresa
                        company_member_details[company].add(role)

        # Após coletar de todas as empresas, faz a contagem
        for company, roles in company_member_details.items():
            for role in roles:
                member_role_counts[role] += 1
        
        if not member_role_counts:
            return "Nenhuma informação sobre membros elegíveis foi encontrada para os filtros selecionados.", None
        
        report_text = "### Análise de Membros Elegíveis ao Plano\n**Contagem de Empresas por Tipo de Membro:**\n"
        df_counts_data = []
        for role, count in sorted(member_role_counts.items(), key=lambda item: item[1], reverse=True):
            report_text += f"- **{role}:** {count} empresas\n"
            df_counts_data.append({"Tipo de Membro Elegível": role, "Nº de Empresas": count})
        
        # Formata o DataFrame de detalhes
        df_details_data = [{"Empresa": company, "Funções Elegíveis": ", ".join(sorted(list(roles)))} for company, roles in company_member_details.items()]

        dfs_to_return = {
            'Contagem por Tipo de Membro': pd.DataFrame(df_counts_data),
            'Detalhes por Empresa': pd.DataFrame(df_details_data).sort_values(by="Empresa").reset_index(drop=True)
        }
        return report_text, dfs_to_return

    def _count_plans_for_board(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        companies = set() # Usar set para evitar contagem dupla

        for company, details in data_to_analyze.items():
            # Itera sobre cada plano da empresa
            for plan_name, plan_details in details.get("planos_identificados", {}).items():
                governance_section = plan_details.get("topicos_encontrados", {}).get("GovernancaRisco", {})
                
                if "OrgaoDeliberativo" in governance_section:
                    deliberative_organs = governance_section["OrgaoDeliberativo"]
                    normalized_deliberative_organs = [self._normalize_text(org) for org in deliberative_organs]
                    
                    if "conselho de administracao" in normalized_deliberative_organs:
                        companies.add(company)
                        break # Encontrou na empresa, pode ir para a próxima

        if not companies:
            return "Nenhuma empresa com menção ao Conselho de Administração como elegível/aprovador foi encontrada para os filtros selecionados.", None
        
        companies_list = sorted(list(companies))
        report_text = f"**{len(companies_list)}** empresas com menção ao **Conselho de Administração** como elegível ou aprovador de planos foram encontradas para os filtros aplicados."
        df = pd.DataFrame(companies_list, columns=["Empresas com Menção ao Conselho de Administração"])
        return report_text, df

    def _analyze_common_goals(self, normalized_query: str, filters: dict) -> tuple:
        data_to_analyze = self._apply_filters_to_data(filters)
        canonical_indicator_companies = defaultdict(set)

        for company, details in data_to_analyze.items():
            if "planos_identificados" in details:
                for plan_name, plan_details in details["planos_identificados"].items():
                    performance_section = plan_details.get("topicos_encontrados", {}).get("IndicadoresPerformance", {})
                    if not performance_section:
                        continue
                    
                    company_leaf_aliases = []
                    self._collect_leaf_aliases_recursive(performance_section, company_leaf_aliases)

                    for alias in set(company_leaf_aliases):
                        canonical_alias = self.INDICATOR_CANONICAL_MAP.get(alias.lower(), alias)
                        canonical_indicator_companies[canonical_alias].add(company)
    
        canonical_alias_counts = {
            indicator: len(companies_set)
            for indicator, companies_set in canonical_indicator_companies.items()
        }

        filtered_counts = {
            k: v for k, v in canonical_alias_counts.items()
            if k.lower() not in self.TERMOS_A_IGNORAR
        }

        if not filtered_counts:
            return "Nenhum indicador de performance relevante encontrado para os filtros selecionados.", None

        categorized_indicators = defaultdict(list)
        uncategorized_list = []
    
        for indicator, count in filtered_counts.items():
            found_category = None
            for category, indicators_list in self.INDICATOR_CATEGORIES.items():
                if indicator in indicators_list:
                    found_category = category
                    break
            
            if found_category:
                categorized_indicators[found_category].append((indicator, count))
            else:
                uncategorized_list.append((indicator, count))

        report_text = "### Indicadores de Performance Mais Comuns\n\n"
        df_overall_data = []
    
        ordered_categories = ["Financeiro", "Mercado", "Operacional", "ESG"]
    
        for category in ordered_categories:
            if category in categorized_indicators:
                sorted_indicators = sorted(categorized_indicators[category], key=lambda item: item[1], reverse=True)
                
                report_text += f"#### {category}\n"
                for indicator, count in sorted_indicators:
                    report_text += f"- **{indicator}:** {count} empresas\n"
                    df_overall_data.append({"Indicador": indicator, "Categoria": category, "Nº de Empresas": count})
                report_text += "\n"
    
        if uncategorized_list:
            report_text += "#### Outros Indicadores\n"
            sorted_uncategorized = sorted(uncategorized_list, key=lambda item: item[1], reverse=True)
            for indicator, count in sorted_uncategorized:
                report_text += f"- **{indicator}:** {count} empresas\n"
                df_overall_data.append({"Indicador": indicator, "Categoria": "Outros", "Nº de Empresas": count})
            report_text += "\n"

        df = pd.DataFrame(df_overall_data).sort_values(by="Nº de Empresas", ascending=False).reset_index(drop=True)
    
        return report_text, df
        
    def _analyze_common_plan_types(self, normalized_query: str, filters: dict) -> tuple:
        """
        [VERSÃO CORRIGIDA] Analisa e conta a prevalência de cada tipo de plano
        de incentivo, baseando-se nas chaves do dicionário 'planos_identificados'.
        """
        data_to_analyze = self._apply_filters_to_data(filters)
        plan_type_counts = defaultdict(int)

        for company, details in data_to_analyze.items():
            identified_plans = details.get("planos_identificados", {})
            unique_plan_types_for_company = set(identified_plans.keys())

            for plan_type in unique_plan_types_for_company:
                plan_type_counts[plan_type.replace('_', ' ')] += 1

        if not plan_type_counts:
            return "Nenhum tipo de plano encontrado para os filtros selecionados.", None
            
        report_text = "### Tipos de Planos Mais Comuns\n"
        df_data = [{"Tipo de Plano": k, "Nº de Empresas": v} for k, v in sorted(plan_type_counts.items(), key=lambda item: item[1], reverse=True)]
        
        for item in df_data:
            report_text += f"- **{item['Tipo de Plano'].capitalize()}:** {item['Nº de Empresas']} empresas\n"
            
        return report_text, pd.DataFrame(df_data)
    
    def _kb_flat_map(self) -> dict:
        """[VERSÃO DE DEPURAÇÃO] Cria um mapa plano de alias -> (seção, nome_formatado, nome_bruto)."""
        # Adicione 'import streamlit as st' no topo do seu arquivo analytical_engine.py
        import streamlit as st

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

            self._recursive_flat_map_builder(data, section, flat_map)
        
        # --- DEBUG PRINT ---
        with st.expander("--- DEBUG: Conteúdo do `flat_map` gerado ---"):
             # Filtra o mapa para encontrar chaves relacionadas a "matching"
            debug_info = {k: v for k, v in flat_map.items() if "matching" in k or "coinvestimento" in k}
            st.write("Verificando se os aliases para 'Matching Coinvestimento' foram mapeados:")
            st.json(debug_info if debug_info else {"status": "Nenhum alias relevante encontrado no mapa."})
            st.write("--- Mapa Completo (primeiros 300 itens) ---")
            st.json({k: v for i, (k, v) in enumerate(flat_map.items()) if i < 300})
        # --- FIM DEBUG ---

        self._kb_flat_map_cache = flat_map
        return flat_map
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
        
        def _collect_all_topic_keys(node, collected_keys: set):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key not in ["_aliases", "subtopicos"]: collected_keys.add(key)
                    if isinstance(value, (dict, list)): _collect_all_topic_keys(value, collected_keys)
            elif isinstance(node, list):
                for item in node: _collect_all_topic_keys(item, collected_keys)

        companies_found = set()
        for company_name, details in data_to_analyze.items():
            identified_plans = details.get("planos_identificados", {})
            for plan_name, plan_details in identified_plans.items():
                topics_for_plan = plan_details.get("topicos_encontrados", {})
                all_plan_topic_keys = set()
                _collect_all_topic_keys(topics_for_plan, all_plan_topic_keys)
                if topic_name_raw in all_plan_topic_keys:
                    companies_found.add(company_name)
                    break 
        
        companies_list = sorted(list(companies_found))
        if not companies_list:
            return f"Nenhuma empresa encontrada com o tópico '{topic_name_formatted}' para os filtros aplicados.", None

        report_text = f"Encontradas **{len(companies_list)}** empresas com o tópico: **{topic_name_formatted.capitalize()}**"
        df = pd.DataFrame(companies_list, columns=[f"Empresas com {topic_name_formatted.capitalize()}"])
        return report_text, df
