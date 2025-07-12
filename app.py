# -*- coding: utf-8 -*-
"""
AGENTE DE CONSULTA COM LÓGICA ORIGINAL RESTAURADA (V5)
Aplicação web para análise de planos de incentivo de longo prazo, otimizada
para ser executada na Streamlit Community Cloud.

Esta versão restaura a robustez e a inteligência de orquestração do agente
original, aplicando-as à nova e eficiente estrutura de dados V7.
"""

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import re
import pandas as pd
import logging
import unicodedata
import requests

# --- CONFIGURAÇÕES GERAIS ---
# O BASE_PATH agora aponta para uma pasta 'dados' relativa.
# Esta estrutura deve existir no seu repositório do GitHub.
BASE_PATH = 'dados'

# Caminhos para os novos artefactos V7
FAISS_INDEX_PATH = os.path.join(BASE_PATH, 'faiss_index_contextual_v7.bin')
CHUNKS_MAP_PATH = os.path.join(BASE_PATH, 'chunks_com_metadata_contextual_v7.json')
CONSOLIDATED_TABLE_PATH = os.path.join(BASE_PATH, 'tabela_consolidada_v7.csv')

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 20

# Configurações da API do Gemini
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash-latest"

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DICIONÁRIOS DE CONHECIMENTO (do agente original) ---
TERMOS_TECNICOS_LTIP = {
    "Ações Restritas": ["Restricted Shares", "Plano de Ações Restritas", "Outorga de Ações", "ações restritas", "RSU", "Restricted Stock Units"],
    "Opções de Compra de Ações": ["Stock Options", "ESOP", "Plano de Opção de Compra", "Outorga de Opções", "opções", "Plano de Opção", "Plano de Opções", "SOP"],
    "Ações Fantasmas": ["Phantom Shares", "Ações Virtuais"],
    "Opções Fantasmas (SAR)": ["Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
    "Planos com Condição de Performance": ["Performance Shares", "Performance Units", "PSU", "Plano de Desempenho", "Metas de Performance", "performance", "desempenho"],
    "Plano de Compra de Ações (ESPP)": ["Plano de Compra de Ações", "Employee Stock Purchase Plan", "ESPP", "Ações com Desconto"],
    "Bônus Diferido": ["Staying Bonus", "Retention Bonus", "Bônus de Permanência", "Bônus de Retenção", "bônus", "Deferred Bonus"],
    "Matching": ["Matching", "Contrapartida", "Co-investimento", "Plano de Matching", "investimento"],
    "Outorga": ["Outorga", "Concessão", "Grant", "Grant Date", "Data da Outorga", "Aprovação"],
    "Vesting": ["Vesting", "Período de Carência", "Condições de Carência", "Aquisição de Direitos", "carência", "cronograma de vesting"],
    "Antecipação de Vesting": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação de carência", "antecipação do vesting", "antecipação"],
    "Tranche / Lote": ["Tranche", "Lote", "Parcela do Vesting"],
    "Cliff": ["Cliff Period", "Período de Cliff", "Carência Inicial"],
    "Preço": ["Preço", "Preço de Exercício", "Strike", "Strike Price"],
    "Ciclo de Vida do Exercício": ["Exercício", "Período de Exercício", "pagamento", "liquidação", "vencimento", "expiração", "forma de liquidação"],
    "Lockup": ["Lockup", "Período de Lockup", "Restrição de Venda", "período de restrição"],
    "Governança e Documentos": ["Regulamento", "Regulamento do Plano", "Contrato de Adesão", "Termo de Outorga", "Comitê de Remuneração", "Comitê de Pessoas", "Deliberação"],
    "Malus e Clawback": ["Malus", "Clawback", "Redução", "Devolução", "Cláusula de Recuperação", "Forfeiture", "Cancelamento", "Perda do Direito"],
    "Estrutura do Plano/Programa": ["Plano", "Planos", "Programa", "Programas", "termos e condições gerais"],
    "Diluição": ["Diluição", "Dilution", "Capital Social"],
    "Elegíveis": ["Participantes", "Beneficiários", "Elegíveis", "Empregados", "Administradores", "Executivos", "Colaboradores", "Conselheiros"],
    "Condição de Saída": ["Desligamento", "Saída", "Término do Contrato", "Rescisão", "Demissão", "Good Leaver", "Bad Leaver"],
    "Tratamento em Casos Especiais": ["Aposentadoria", "Morte", "Invalidez", "Reforma", "Afastamento"],
    "Indicadores": ["TSR", "Total Shareholder Return", "Retorno Total ao Acionista", "CDI", "IPCA", "Selic", "ROIC", "EBITDA", "LAIR", "Lucro", "CAGR", "Metas ESG", "Receita Líquida"],
    "Eventos Corporativos": ["IPO", "grupamento", "desdobramento", "cisão", "fusão", "incorporação", "bonificações", "bonificação"],
    "Mudança de Controle": ["Mudança de Controle", "Change of Control", "Evento de Liquidez"],
    "Dividendos": ["Dividendos", "Dividendo", "JCP", "Juros sobre capital próprio", "Tratamento de Dividendos", "dividend equivalent", "proventos"],
    "Encargos": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"],
    "Contabilidade e Normas": ["IFRS 2", "CPC 10", "Valor Justo", "Fair Value", "Black-Scholes", "Despesa Contábil", "Volatilidade"]
}
AVAILABLE_TOPICS = list(TERMOS_TECNICOS_LTIP.keys())

# --- CARREGAMENTO DE DADOS OTIMIZADO ---

@st.cache_resource
def load_all_artifacts():
    """
    Carrega todos os artefactos da nova estrutura de dados V7.
    """
    artifacts = {
        "model": None, "index": None, "chunks_dict": None, 
        "consolidated_df": None, "company_catalog": None
    }
    try:
        logger.info("A carregar o modelo de embedding...")
        artifacts["model"] = SentenceTransformer(MODEL_NAME)
        
        logger.info("A carregar o índice FAISS unificado...")
        artifacts["index"] = faiss.read_index(FAISS_INDEX_PATH)
        
        logger.info("A carregar o mapa de chunks com metadados...")
        with open(CHUNKS_MAP_PATH, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        artifacts["chunks_dict"] = {chunk['id']: chunk for chunk in chunks_data}
        
        logger.info("A carregar a tabela consolidada...")
        artifacts["consolidated_df"] = pd.read_csv(CONSOLIDATED_TABLE_PATH)

        try:
            from catalog_data import company_catalog_rich
            artifacts["company_catalog"] = company_catalog_rich
            logger.info("✅ Catálogo de empresas carregado com sucesso.")
        except ImportError:
            logger.warning("`catalog_data.py` não encontrado. A identificação de empresas por apelidos será limitada.")
            artifacts["company_catalog"] = []

        logger.info("✅ Todos os artefactos foram carregados com sucesso.")
        return artifacts
    except Exception as e:
        st.error(f"ERRO CRÍTICO AO CARREGAR ARTEFACTOS: {e}")
        st.error("Verifique se os ficheiros de índice e de dados (gerados pelo script de indexação V7) existem na pasta 'dados' do seu repositório GitHub e não estão corrompidos.")
        return artifacts


# --- LÓGICA DE BUSCA, ANÁLISE E GERAÇÃO DE RESPOSTA ---

def create_analysis_plan_with_llm(query, company_catalog, all_known_companies):
    """
    Usa o Gemini para interpretar a pergunta e criar um plano de análise robusto,
    identificando tanto empresas quanto tópicos.
    """
    if not GEMINI_API_KEY:
        st.error("Chave de API do Gemini não configurada.")
        return None

    # 1. Usar LLM para identificar as empresas mencionadas na query
    #    Fornecemos a lista de empresas que temos nos dados como contexto para o LLM.
    company_prompt = f"""
    Dada a lista de empresas conhecidas: {json.dumps(all_known_companies)}.
    Analise a seguinte pergunta do utilizador: "{query}".
    Identifique TODAS as empresas da lista conhecida que são mencionadas na pergunta.
    Se a pergunta mencionar um apelido (ex: Magalu), associe-o ao nome completo (ex: Magazine Luiza).
    Retorne APENAS uma lista JSON com os nomes canónicos das empresas encontradas.
    Formato da resposta: ["Empresa A", "Empresa B"]
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": company_prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    mentioned_companies = []
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            mentioned_companies = json.loads(json_match.group(0))
    except Exception as e:
        logger.error(f"Falha ao chamar LLM para identificar empresas: {e}")
        # Se o LLM falhar, recorremos ao catálogo como fallback
        if company_catalog:
            companies_found_by_alias = {}
            for company_data in company_catalog:
                for alias in company_data.get("aliases", []):
                    if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query.lower()):
                        companies_found_by_alias[company_data["canonical_name"]] = len(alias.split())
            if companies_found_by_alias:
                mentioned_companies = [c for c, s in sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)]

    if not mentioned_companies:
        return None # Nenhuma empresa encontrada

    # 2. Usar LLM para identificar os tópicos de interesse
    topic_prompt = f"""Você é um consultor de ILP. Identifique os TÓPICOS CENTRAIS da pergunta: "{query}".
    Retorne APENAS uma lista JSON com os tópicos mais relevantes da seguinte lista: {json.dumps(AVAILABLE_TOPICS)}.
    Se a pergunta for genérica sobre uma empresa, selecione tópicos para uma análise geral como ["Estrutura do Plano/Programa", "Vesting", "Elegíveis"].
    Formato da resposta: ["Tópico 1", "Tópico 2"]"""
    
    payload = {"contents": [{"parts": [{"text": topic_prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            topics = json.loads(json_match.group(0))
        else:
            topics = ["Estrutura do Plano/Programa", "Vesting", "Elegíveis"] # Fallback
    except Exception as e:
        logger.error(f"Falha ao chamar LLM para tópicos: {e}")
        topics = ["Estrutura do Plano/Programa", "Vesting", "Elegíveis"] # Fallback

    plan = {
        "empresas": mentioned_companies,
        "topicos": topics,
        "tipo_analise": "comparativa" if len(mentioned_companies) > 1 else "unica"
    }
    return plan


def execute_rag_analysis(plan, query, artifacts):
    """
    Executa o plano de análise RAG, buscando e construindo o contexto.
    """
    model = artifacts["model"]
    index = artifacts["index"]
    chunks_dict = artifacts["chunks_dict"]
    
    all_context = ""
    all_sources = set()

    for company in plan['empresas']:
        logger.info(f"A executar a busca para a empresa: {company}")
        
        query_vector = model.encode([query], normalize_embeddings=True).astype('float32')
        distances, ids = index.search(query_vector, TOP_K_SEARCH)
        
        company_context = ""
        company_sources = set()
        
        if ids.size > 0:
            for chunk_id in ids[0]:
                if chunk_id != -1:
                    chunk_info = chunks_dict.get(chunk_id)
                    if chunk_info and company.lower() in chunk_info['metadata']['empresa'].lower():
                        if any(topic.lower() in (ct.lower() for ct in chunk_info['metadata']['chunk_topics']) for topic in plan['topicos']):
                            metadata = chunk_info['metadata']
                            company_context += f"--- Contexto (Fonte: {metadata['arquivo_origem']}) ---\n"
                            company_context += f"Secção: {metadata['section_title']}\n"
                            company_context += f"Tópicos no Trecho: {', '.join(metadata['chunk_topics'])}\n"
                            company_context += f"Conteúdo: {chunk_info['content']}\n\n"
                            company_sources.add(chunk_info['source'])

        if company_context:
            all_context += f"--- ANÁLISE PARA {company.upper()} ---\n\n{company_context}"
            all_sources.update(company_sources)

    return all_context, all_sources


def get_llm_response(prompt):
    """
    Gera a resposta final usando o contexto recuperado e a API do Gemini.
    """
    if not GEMINI_API_KEY:
        st.error("Chave de API do Gemini não configurada.")
        return "ERRO: Chave de API não configurada."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        st.error(f"Falha na comunicação com a API do Gemini: {e}")
        return f"ERRO ao gerar resposta final."


# --- INTERFACE STREAMLIT (Aplicação Principal) ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP (V5)", page_icon="🔍", layout="wide")
    st.title("🤖 Agente de Análise Híbrido e Comparativo (V5)")
    st.markdown("---")

    artifacts = load_all_artifacts()
    if not artifacts["model"]: return

    with st.sidebar:
        st.header("📊 Informações do Sistema")
        if artifacts["consolidated_df"] is not None:
            st.metric("Documentos na Tabela", len(artifacts["consolidated_df"]['caminho_completo'].unique()))
        if artifacts["chunks_dict"] is not None:
            st.metric("Total de Chunks Indexados", len(artifacts["chunks_dict"]))
        if artifacts["company_catalog"]:
            st.success("Catálogo de empresas carregado.")
        else:
            st.warning("Catálogo de empresas não encontrado.")
        if artifacts["consolidated_df"] is not None:
            with st.expander("Empresas na Base de Dados"):
                st.dataframe(sorted(artifacts["consolidated_df"]['empresa'].unique()), use_container_width=True)
        st.success("✅ Sistema pronto para análise")

    st.header("💬 Faça a sua pergunta")
    st.info("**Exemplos:** `Como é o plano de vesting da Vale?` ou `Compare o tratamento de dividendos da Petrobras com a Gerdau.`")
    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Digite aqui...")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return

        with st.status("1️⃣ A criar plano de análise...", expanded=True) as status:
            # Passa a lista de empresas conhecidas para a função de criação do plano
            known_companies = artifacts["consolidated_df"]['empresa'].unique().tolist()
            plan = create_analysis_plan_with_llm(user_query, artifacts["company_catalog"], known_companies)
            
            if not plan:
                st.error("❌ Nenhuma empresa conhecida foi identificada na sua pergunta.")
                status.update(label="Falha ao criar plano.", state="error")
                return
            st.write(f"**Tipo de Análise:** {plan['tipo_analise'].title()}")
            st.write(f"**Empresa(s):** {', '.join(plan['empresas'])}")
            st.write(f"**Tópicos Identificados:** {', '.join(plan['topicos'])}")
            status.update(label="Plano de análise criado!", state="complete")

        with st.spinner("2️⃣ A executar a busca e a recolher o contexto..."):
            context, sources = execute_rag_analysis(plan, user_query, artifacts)
        
        st.markdown("---")
        st.subheader("📋 Resultado da Análise")

        if not context:
            st.warning("Não foram encontrados contextos relevantes para responder à pergunta.")
            return
            
        with st.spinner("3️⃣ A gerar a resposta final com o Gemini..."):
            prompt = f"""Você é um consultor especialista em planos de incentivo de longo prazo (ILP).
            Sua tarefa é responder à pergunta do utilizador com base no contexto fornecido.
            Se a pergunta for uma comparação, crie um relatório comparativo bem estruturado.
            Se a pergunta for sobre uma única empresa, forneça uma análise detalhada.
            Seja profissional e baseie-se estritamente nos dados. Se a informação não estiver no contexto, afirme isso claramente.

            PERGUNTA ORIGINAL DO UTILIZADOR: "{user_query}"

            CONTEXTO COLETADO DOS DOCUMENTOS:
            {context}

            RELATÓRIO ANALÍTICO FINAL:
            """
            final_answer = get_llm_response(prompt)
            st.markdown(final_answer)

        if sources:
            with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=False):
                for source in sorted(list(sources)):
                    st.write(f"- {os.path.basename(source)}")

if __name__ == "__main__":
    main()
