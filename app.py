# -*- coding: utf-8 -*-
"""
AGENTE DE CONSULTA COM LÓGICA ORIGINAL RESTAURADA (V6)
Aplicação web para análise de planos de incentivo de longo prazo.

Esta versão restaura completamente a robustez e a inteligência de
orquestração do agente original, aplicando-as à nova e eficiente
estrutura de dados V7.
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
BASE_PATH = 'dados'

# Caminhos para os novos artefactos V7
FAISS_INDEX_PATH = os.path.join(BASE_PATH, 'faiss_index_contextual_v7.bin')
CHUNKS_MAP_PATH = os.path.join(BASE_PATH, 'chunks_com_metadata_contextual_v7.json')
CONSOLIDATED_TABLE_PATH = os.path.join(BASE_PATH, 'tabela_consolidada_v7.csv')

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 25

# Configurações da API do Gemini
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash-latest"

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DICIONÁRIOS DE CONHECIMENTO ---
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

# --- CARREGAMENTO DE DADOS ---

@st.cache_resource
def load_all_artifacts():
    artifacts = {"model": None, "index": None, "chunks_dict": None, "consolidated_df": None, "company_catalog": None}
    try:
        logger.info("A carregar artefactos...")
        artifacts["model"] = SentenceTransformer(MODEL_NAME)
        artifacts["index"] = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_MAP_PATH, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        artifacts["chunks_dict"] = {chunk['id']: chunk for chunk in chunks_data}
        artifacts["consolidated_df"] = pd.read_csv(CONSOLIDATED_TABLE_PATH)
        try:
            from catalog_data import company_catalog_rich
            artifacts["company_catalog"] = company_catalog_rich
        except ImportError:
            logger.warning("`catalog_data.py` não encontrado.")
        logger.info("✅ Artefactos carregados com sucesso.")
        return artifacts
    except Exception as e:
        st.error(f"ERRO CRÍTICO AO CARREGAR ARTEFACTOS: {e}")
        return artifacts

# --- LÓGICA DE ORQUESTRAÇÃO E ANÁLISE ---

def get_llm_json_response(prompt):
    if not GEMINI_API_KEY: return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_response)
    except Exception as e:
        logger.error(f"Falha ao obter resposta JSON do LLM: {e}")
        return None

def create_analysis_plan(query, company_catalog, all_known_companies):
    company_prompt = f'Dada a lista de empresas conhecidas: {json.dumps(all_known_companies)}. Analise a pergunta do utilizador: "{query}". Identifique TODAS as empresas da lista que são mencionadas. Retorne APENAS uma lista JSON com os nomes canónicos. Exemplo: ["Empresa A", "Empresa B"]'
    mentioned_companies = get_llm_json_response(company_prompt)

    if not mentioned_companies:
        return None

    topic_prompt = f'Você é um consultor de ILP. Identifique os TÓPICOS CENTRAIS da pergunta: "{query}". Retorne APENAS uma lista JSON com os tópicos mais relevantes da seguinte lista: {json.dumps(AVAILABLE_TOPICS)}. Se for genérica, use ["Estrutura do Plano/Programa", "Vesting", "Elegíveis"].'
    topics = get_llm_json_response(topic_prompt) or ["Estrutura do Plano/Programa", "Vesting", "Elegíveis"]

    return {"empresas": mentioned_companies, "topicos": topics, "tipo_analise": "comparativa" if len(mentioned_companies) > 1 else "unica"}

def execute_rag_analysis(plan, artifacts):
    model, index, chunks_dict = artifacts["model"], artifacts["index"], artifacts["chunks_dict"]
    all_context, all_sources = "", set()

    for company in plan['empresas']:
        specific_query = f"Análise sobre {', '.join(plan['topicos'])} para a empresa {company}"
        logger.info(f"A criar busca específica: '{specific_query}'")
        query_vector = model.encode([specific_query], normalize_embeddings=True).astype('float32')
        distances, ids = index.search(query_vector, TOP_K_SEARCH)
        
        company_context, company_sources = "", set()
        if ids.size > 0:
            for chunk_id in ids[0]:
                if chunk_id != -1 and (chunk_info := chunks_dict.get(chunk_id)) and company.lower() in chunk_info['metadata']['empresa'].lower():
                    metadata = chunk_info['metadata']
                    company_context += f"--- Contexto (Fonte: {metadata['arquivo_origem']}) ---\nSecção: {metadata['section_title']}\nConteúdo: {chunk_info['content']}\n\n"
                    company_sources.add(chunk_info['source'])
        
        if company_context:
            all_context += f"--- ANÁLISE PARA {company.upper()} ---\n\n{company_context}"
            all_sources.update(company_sources)

    return all_context, all_sources

def get_final_answer(prompt):
    if not GEMINI_API_KEY: return "ERRO: Chave de API não configurada."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"ERRO ao gerar resposta final."

def handle_aggregate_query(query, df):
    st.info("Detectada uma pergunta agregada. A analisar a tabela consolidada...")
    query_lower = query.lower()
    try:
        match = re.search(r'(?:com|têm|possuem|oferecem)\s+(.+)', query_lower)
        if not match:
            st.warning("Não consegui entender qual característica procura. Tente 'Quais empresas têm Ações Restritas?'.")
            return
        target_feature = match.group(1).replace('?', '').strip()
        results_df = df[df['nomes_planos'].str.contains(target_feature, case=False, na=False)]
        if results_df.empty:
            st.warning(f"Nenhuma empresa encontrada com planos que mencionam '{target_feature}'.")
        else:
            empresas = sorted(results_df['empresa'].unique())
            st.success(f"✅ **{len(empresas)} empresa(s)** encontrada(s) com planos que mencionam '{target_feature}':")
            st.dataframe(pd.DataFrame(empresas, columns=["Empresa"]), use_container_width=True, hide_index=True)
    except Exception as e:
        logger.error(f"Erro na busca agregada: {e}")
        st.error("Ocorreu um erro ao analisar a sua pergunta.")

# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP (V6)", page_icon="🔍", layout="wide")
    st.title("🤖 Agente de Análise de Planos de Incentivo (V6)")
    st.markdown("---")

    artifacts = load_all_artifacts()
    if not artifacts["model"]: return

    with st.sidebar:
        st.header("📊 Informações do Sistema")
        if artifacts["consolidated_df"] is not None:
            st.metric("Documentos na Base", len(artifacts["consolidated_df"]['caminho_completo'].unique()))
        if artifacts["chunks_dict"] is not None:
            st.metric("Chunks Indexados", len(artifacts["chunks_dict"]))
        st.success("✅ Sistema pronto para análise")

    st.header("💬 Faça a sua pergunta")
    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Como é o plano de vesting da Vale? ou Compare o tratamento de dividendos da Petrobras com a Gerdau.")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return

        # --- ROTEADOR DE INTENÇÃO ---
        is_aggregate = any(keyword in user_query.lower() for keyword in ["quais", "quantas", "liste"])
        
        if is_aggregate:
            handle_aggregate_query(user_query, artifacts["consolidated_df"])
        else:
            with st.status("1️⃣ A criar plano de análise...", expanded=True) as status:
                known_companies = artifacts["consolidated_df"]['empresa'].unique().tolist()
                plan = create_analysis_plan(user_query, artifacts["company_catalog"], known_companies)
                if not plan:
                    st.error("❌ Nenhuma empresa conhecida foi identificada na sua pergunta.")
                    status.update(label="Falha ao criar plano.", state="error")
                    return
                st.write(f"**Tipo de Análise:** {plan['tipo_analise'].title()}")
                st.write(f"**Empresa(s):** {', '.join(plan['empresas'])}")
                st.write(f"**Tópicos Identificados:** {', '.join(plan['topicos'])}")
                status.update(label="Plano de análise criado!", state="complete")

            with st.spinner("2️⃣ A executar a busca e a recolher o contexto..."):
                context, sources = execute_rag_analysis(plan, artifacts)
            
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
                final_answer = get_final_answer(prompt)
                st.markdown(final_answer)

            if sources:
                with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=False):
                    for source in sorted(list(sources)):
                        st.write(f"- {os.path.basename(source)}")

if __name__ == "__main__":
    main()
