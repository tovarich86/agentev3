# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP - VERSÃO STREAMLIT (HÍBRIDO)
Aplicação web para análise de planos de incentivo de longo prazo, com
capacidades de busca profunda (RAG) e análise agregada (resumo).
"""

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import glob
import os
import re
import unicodedata
import logging
from functools import lru_cache

# --- CONFIGURAÇÕES GERAIS ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
DADOS_PATH = "dados" # Centraliza o caminho para a pasta de dados

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DICIONÁRIOS DE CONHECIMENTO ---

# Dicionário principal para tradução de termos e busca de tópicos
TERMOS_TECNICOS_LTIP = {
    "Ações Restritas": ["Restricted Shares", "Plano de Ações Restritas", "Outorga de Ações", "ações restritas", "RSU"],
    "Opções de Compra de Ações": ["Stock Options", "ESOP", "Plano de Opção de Compra", "Outorga de Opções", "opções", "Plano de Opção", "Plano de Opções", "SOP"],
    "Ações Fantasmas": ["Phantom Shares", "Ações Virtuais"],
    "Opções Fantasmas (SAR)": ["Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
    "Bônus Diferido": ["Staying Bonus", "Retention Bonus", "Bônus de Permanência", "Bônus de Retenção", "bônus"],
    "Planos com Condição de Performance": ["Performance Shares", "Performance Stock Options", "Plano de Desempenho", "Metas de Performance", "performance", "desempenho"],
    "Vesting": ["Vesting", "Período de Carência", "Condições de Carência", "Aquisição de Direitos", "carência", "cronograma de vesting"],
    "Antecipação de Vesting": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação de carência", "antecipação do vesting", "antecipação"],
    "Tranche / Lote": ["Tranche", "Lote", "Parcela do Vesting"],
    "Cliff": ["Cliff Period", "Período de Cliff", "Carência Inicial"],
    "Matching": ["Matching", "Contrapartida", "Co-investimento", "Plano de Matching", "investimento"],
    "Lockup": ["Lockup", "Período de Lockup", "Restrição de Venda"],
    "Estrutura do Plano/Programa": ["Plano", "Planos", "Programa", "Programas", "termos e condições gerais"],
    "Ciclo de Vida do Exercício": ["pagamento", "liquidação", "vencimento", "expiração", "forma de liquidação"],
    "Eventos Corporativos": ["IPO", "grupamento", "desdobramento", "bonificações", "bonificação"],
    "Dividendos": ["Dividendos", "Dividendo", "JCP", "Juros sobre capital próprio", "Tratamento de Dividendos", "equivalente em dividendos", "proventos"],
    "Encargos": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"],
}

# Tópicos para o fallback do LLM na análise profunda (RAG)
AVAILABLE_TOPICS = list(TERMOS_TECNICOS_LTIP.keys()) + [
    "data de aprovação e órgão responsável", "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas",
    "critérios para fixação do preço de aquisição ou exercício", "preço de exercício", "strike price", "restrições à transferência das ações",
    "critérios e eventos de suspensão/extinção", "efeitos da saída do administrador"
]

# --- CARREGAMENTO DE DADOS E CACHING ---

@st.cache_resource
def load_all_artifacts():
    """
    Carrega todos os artefatos necessários para a aplicação:
    - Modelo de embedding
    - Índices FAISS e chunks (para RAG)
    - Resumo de características (para buscas agregadas)
    """
    # 1. Carregar Modelo de Embedding
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Carregar Artefatos do RAG (FAISS e Chunks)
    artifacts = {}
    index_files = glob.glob(os.path.join(DADOS_PATH, '*_faiss_index.bin'))
    if not index_files:
        logger.error("Nenhum arquivo de índice FAISS encontrado na pasta 'dados'. O RAG não funcionará.")
    else:
        for index_file in index_files:
            category = os.path.basename(index_file).replace('_faiss_index.bin', '')
            chunks_file = os.path.join(DADOS_PATH, f"{category}_chunks_map.json")
            try:
                index = faiss.read_index(index_file)
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                artifacts[category] = {'index': index, 'chunks': chunk_data}
            except FileNotFoundError:
                logger.warning(f"Arquivo de chunks para a categoria '{category}' não encontrado. Pulando.")
                continue
    
    # 3. Carregar Resumo de Características
    summary_data = None
    summary_file_path = os.path.join(DADOS_PATH, 'resumo_caracteristicas.json')
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        logger.error("Arquivo 'resumo_caracteristicas.json' não encontrado. Buscas agregadas não funcionarão.")
        
    return model, artifacts, summary_data

@st.cache_data
def criar_mapa_de_alias():
    """
    Cria um dicionário que mapeia cada apelido ao seu tópico canônico para buscas rápidas.
    Ex: {'performance': 'Planos com Condição de Performance'}
    """
    alias_to_canonical = {}
    for canonical_name, aliases in TERMOS_TECNICOS_LTIP.items():
        for alias in aliases:
            alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

# --- FUNÇÕES DE LÓGICA DE NEGÓCIO (ROTEADOR E MANIPULADORES) ---

def handle_aggregate_query(query, summary_data, alias_map):
    """
    Lida com perguntas agregadas ("quais", "quantas").
    Retorna a resposta formatada como uma string Markdown.
    """
    query_lower = query.lower()
    
    # 1. Extrair o tópico da pergunta usando o mapa de alias
    topico_canonico_encontrado = None
    # Iterar pelas chaves ordenadas pela mais longa primeiro para evitar correspondências parciais
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    
    for alias in sorted_aliases:
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            topico_canonico_encontrado = alias_map[alias]
            break

    if not topico_canonico_encontrado:
        return "Não consegui identificar um tópico conhecido (como 'performance', 'matching', 'opções') na sua pergunta para fazer a busca. Por favor, tente novamente."

    # 2. Buscar as empresas no JSON
    empresas_encontradas = []
    if summary_data:
        for empresa, dados in summary_data.items():
            if topico_canonico_encontrado in dados.get("topicos_encontrados", []):
                empresas_encontradas.append(empresa)
    
    empresas_encontradas.sort()

    # 3. Formatar a resposta
    if not empresas_encontradas:
        return f"Nenhuma empresa foi encontrada com planos sobre **'{topico_canonico_encontrado}'** nos documentos analisados."

    num_empresas = len(empresas_encontradas)
    
    if "quantas" in query_lower:
        return f"✅ **{num_empresas} empresa(s)** encontrada(s) com planos sobre **'{topico_canonico_encontrado}'**."

    resposta_md = f"✅ **{num_empresas} empresa(s)** encontrada(s) com planos sobre **'{topico_canonico_encontrado}'**:\n\n"
    
    if num_empresas > 0:
        # Apresenta em até 3 colunas para melhor visualização
        num_cols = min(3, num_empresas)
        cols = st.columns(num_cols)
        for i, empresa in enumerate(empresas_encontradas):
            with cols[i % num_cols]:
                st.markdown(f"- {empresa}")
    
    # Retorna a parte textual, as colunas são renderizadas diretamente
    return resposta_md


def handle_rag_query(query, artifacts, model, company_catalog_rich):
    """
    Lida com perguntas detalhadas e comparativas usando o fluxo RAG completo.
    """
    # ETAPA 1: GERAÇÃO DO PLANO
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        # Nota: Idealmente, company_catalog_rich seria carregado uma vez fora.
        # Por simplicidade, mantemos aqui.
        plan_response = create_dynamic_analysis_plan_v2(query, company_catalog_rich, list(artifacts.keys()))
        if plan_response['status'] != "success" or not plan_response['plan']['empresas']:
            st.error("❌ Não consegui identificar empresas na sua pergunta. Tente usar nomes conhecidos (ex: Magalu, Vivo, Itaú).")
            return "Análise abortada.", set()
        
        plan = plan_response['plan']
        empresas = plan.get('empresas', [])
        st.write(f"**🏢 Empresas identificadas:** {', '.join(empresas)}")
        st.write(f"**📝 Tópicos a analisar:** {len(plan.get('topicos', []))}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    # ETAPA 2: LÓGICA DE EXECUÇÃO (com tratamento para comparações)
    final_answer = ""
    sources = set()

    # MODO COMPARATIVO
    if len(empresas) > 1:
        st.info(f"Modo de comparação ativado para {len(empresas)} empresas. Analisando sequencialmente...")
        summaries = []
        for i, empresa in enumerate(empresas):
            with st.status(f"Analisando {i+1}/{len(empresas)}: {empresa}...", expanded=True):
                single_company_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
                query_intent = 'item_8_4_query' if any(term in query.lower() for term in ['8.4', 'formulário']) else 'general_query'
                retrieved_context, retrieved_sources = execute_dynamic_plan(single_company_plan, query_intent, artifacts, model)
                sources.update(retrieved_sources)

                if "Nenhuma informação" in retrieved_context or not retrieved_context.strip():
                    summary = f"## Análise para {empresa.upper()}\n\nNenhuma informação encontrada nos documentos para os tópicos solicitados."
                else:
                    summary_prompt = f"Com base no contexto a seguir sobre a empresa {empresa}, resuma os pontos principais sobre os seguintes tópicos: {', '.join(plan['topicos'])}. Contexto: {retrieved_context}"
                    summary = get_final_unified_answer(summary_prompt, retrieved_context)
                
                summaries.append(f"--- RESUMO PARA {empresa.upper()} ---\n\n{summary}")

        with st.status("Gerando relatório comparativo final...", expanded=True):
            comparison_prompt = f"Com base nos resumos individuais a seguir, crie um relatório comparativo detalhado e bem estruturado entre as empresas, focando nos pontos levantados na pergunta original do usuário.\n\nPergunta original do usuário: '{query}'\n\n" + "\n\n".join(summaries)
            final_answer = get_final_unified_answer(comparison_prompt, "\n\n".join(summaries))
            status.update(label="✅ Relatório comparativo gerado!", state="complete")

    # MODO DE ANÁLISE ÚNICA
    else:
        with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
            query_intent = 'item_8_4_query' if any(term in query.lower() for term in ['8.4', 'formulário']) else 'general_query'
            st.write(f"**🎯 Estratégia detectada:** {'Item 8.4 completo' if query_intent == 'item_8_4_query' else 'Busca geral'}")
            
            retrieved_context, retrieved_sources = execute_dynamic_plan(plan, query_intent, artifacts, model)
            sources.update(retrieved_sources)
            
            if not retrieved_context.strip() or "Nenhuma informação encontrada" in retrieved_context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", set()
            
            st.write(f"**📄 Contexto recuperado de:** {len(sources)} documento(s)")
            status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
        
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, retrieved_context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, sources

# --- FUNÇÕES DE BACKEND (RAG) - sem alterações ---

# Mantidas as funções originais para o fluxo RAG
def create_dynamic_analysis_plan_v2(query, company_catalog_rich, available_indices):
    # Esta função agora é chamada apenas pelo `handle_rag_query`
    api_key = GEMINI_API_KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    query_lower = query.lower().strip()
    
    # Identificação de Empresas
    mentioned_companies = []
    companies_found_by_alias = {}
    if company_catalog_rich:
        for company_data in company_catalog_rich:
            for alias in company_data.get("aliases", []):
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                    score = len(alias.split())
                    canonical_name = company_data["canonical_name"]
                    if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                        companies_found_by_alias[canonical_name] = score
        if companies_found_by_alias:
            sorted_companies = sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)
            mentioned_companies = [company for company, score in sorted_companies]
    
    if not mentioned_companies:
        return {"status": "error", "plan": {}}
    
    # Identificação de Tópicos
    topics = []
    found_topics = set()
    alias_map = criar_mapa_de_alias() # Reutiliza o mapa de alias
    for alias, canonical_name in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            found_topics.add(canonical_name)

    if found_topics:
        topics = list(found_topics)
    else:
        # Fallback para LLM se nenhum tópico for encontrado
        prompt = f"""Você é um consultor de ILP. Identifique os TÓPICOS CENTRAIS da pergunta: "{query}".
        Retorne APENAS uma lista JSON com os tópicos mais relevantes de: {json.dumps(AVAILABLE_TOPICS)}.
        Se for genérica, selecione tópicos para uma análise geral. Formato: ["Tópico 1", "Tópico 2"]"""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            response.raise_for_status()
            text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
            json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
            if json_match:
                topics = json.loads(json_match.group(0))
            else:
                topics = ["Estrutura do Plano/Programa", "Vesting", "Opções de Compra de Ações"]
        except Exception as e:
            logger.error(f"Falha ao chamar LLM para tópicos: {e}")
            topics = ["Estrutura do Plano/Programa", "Vesting", "Opções de Compra de Ações"]
            
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}


def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """
    Executa o plano de busca com controle robusto de tokens e deduplicação.
    (Esta função permanece exatamente como no seu código original)
    """
    # ... (Cole aqui o corpo inteiro da sua função `execute_dynamic_plan` original)
    # Nenhuma alteração é necessária nesta função.
    # Por brevidade, o corpo foi omitido aqui, mas deve ser colado integralmente.
    full_context = "Contexto recuperado pela função execute_dynamic_plan." # Placeholder
    all_retrieved_docs = {"doc1.pdf", "doc2.pdf"} # Placeholder
    return full_context, all_retrieved_docs


def get_final_unified_answer(query, context):
    """
    Gera a resposta final usando o contexto recuperado.
    (Esta função permanece exatamente como no seu código original)
    """
    # ... (Cole aqui o corpo inteiro da sua função `get_final_unified_answer` original)
    # Nenhuma alteração é necessária nesta função.
    # Por brevidade, o corpo foi omitido aqui, mas deve ser colado integralmente.
    return f"Resposta final gerada por LLM para a query: '{query}' com base no contexto." # Placeholder


# --- INTERFACE STREAMLIT (Aplicação Principal) ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    st.markdown("---")

    # Carregamento centralizado dos artefatos
    model, artifacts, summary_data = load_all_artifacts()
    ALIAS_MAP = criar_mapa_de_alias()

    # Tenta carregar o catálogo de empresas, mas não quebra se não encontrar
    try:
        from catalog_data import company_catalog_rich
    except ImportError:
        company_catalog_rich = []
        logger.warning("`catalog_data.py` não encontrado. A identificação de empresas por apelidos será limitada.")

    # Validação dos dados carregados
    if not artifacts:
        st.error("❌ Erro crítico: Nenhum artefato de busca (índices FAISS) foi carregado. A análise profunda está desativada.")
    if not summary_data:
        st.warning("⚠️ Aviso: O arquivo `resumo_caracteristicas.json` não foi encontrado. Análises de 'quais/quantas empresas' estão desativadas.")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Fontes de Documentos (RAG)", len(artifacts) if artifacts else 0)
        st.metric("Empresas no Resumo", len(summary_data) if summary_data else 0)
        
        if summary_data:
            with st.expander("empresas com características identificadas"):
                st.dataframe(sorted(list(summary_data.keys())), use_container_width=True)
        
        st.success("✅ Sistema pronto para análise")
        st.info(f"Modelo de embedding: `{MODEL_NAME}`")

    # --- Corpo Principal ---
    st.header("💬 Faça sua pergunta")
    
    # Colunas para exemplos de perguntas
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Experimente uma análise agregada:**")
        st.code("Quais empresas possuem planos com matching?")
        st.code("Quantas empresas têm vesting acelerado?")
    with col2:
        st.info("**Ou uma análise profunda (RAG):**")
        st.code("Compare o vesting da Vale com a Petrobras")
        st.code("Como funciona o lockup da Magazine Luiza?")

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quantas empresas oferecem ações restritas?")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return

        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
        
        # --- O ROTEADOR DE INTENÇÃO EM AÇÃO ---
        final_answer = ""
        sources = set()
        
        query_lower = user_query.lower()
        aggregate_keywords = ["quais", "quantas", "liste", "qual a lista de"]

        # Rota 1: Pergunta agregada
        if any(keyword in query_lower for keyword in aggregate_keywords):
            if not summary_data:
                st.error("A funcionalidade de busca agregada está desativada pois o arquivo `resumo_caracteristicas.json` não foi encontrado.")
            else:
                st.info("Detectada uma pergunta agregada. Buscando no resumo de características...")
                with st.spinner("Analisando resumo..."):
                    # A função `handle_aggregate_query` agora pode renderizar colunas diretamente
                    # e retornar o texto principal.
                    final_answer_text_part = handle_aggregate_query(user_query, summary_data, ALIAS_MAP)
                    st.markdown(final_answer_text_part) # Renderiza o texto e as colunas (se houver)

        # Rota 2: Pergunta profunda (RAG)
        else:
            if not artifacts:
                st.error("A funcionalidade de análise profunda está desativada pois os índices de busca não foram encontrados.")
            elif not company_catalog_rich:
                 st.error("A funcionalidade de análise profunda está desativada pois o `catalog_data.py` não foi encontrado.")
            else:
                st.info("Detectada uma pergunta detalhada. Acionando análise profunda (RAG)...")
                final_answer, sources = handle_rag_query(user_query, artifacts, model, company_catalog_rich)
                st.markdown(final_answer) # Renderiza a resposta do RAG

        # Fontes consultadas (apenas para o RAG)
        if sources:
            st.markdown("---")
            with st.expander(f"📚 Documentos consultados na análise profunda ({len(sources)})", expanded=False):
                for i, source in enumerate(sorted(list(sources)), 1):
                    st.write(f"{i}. {source}")

if __name__ == "__main__":
    main()
