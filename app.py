# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP - VERSÃO FINAL E FUNCIONAL
"""

# --- 1. Importações e Configurações ---
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
import pandas as pd
from pathlib import Path

try:
    from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
except ImportError:
    st.error("ERRO CRÍTICO: Crie o arquivo 'knowledge_base.py' e cole o 'DICIONARIO_UNIFICADO_HIERARQUICO' nele.")
    st.stop()

# Configurações Gerais
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash-latest" # Usando o modelo mais recente e capaz
DADOS_PATH = Path("dados")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 2. Carregamento de Dados e Funções Auxiliares ---

@st.cache_resource
def load_all_artifacts():
    DADOS_PATH.mkdir(exist_ok=True)
    ARQUIVOS_REMOTOS = {
        "item_8_4_chunks_map_final.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/item_8_4_chunks_map_final.json",
        "item_8_4_faiss_index_final.bin": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/item_8_4_faiss_index_final.bin",
        "outros_documentos_chunks_map_final.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/outros_documentos_chunks_map_final.json",
        "outros_documentos_faiss_index_final.bin": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/outros_documentos_faiss_index_final.bin",
        "resumo_fatos_e_topicos_final_enriquecido.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/resumo_fatos_e_topicos_final_enriquecido.json"
    }
    for nome_arquivo, url in ARQUIVOS_REMOTOS.items():
        caminho_arquivo = DADOS_PATH / nome_arquivo
        if not caminho_arquivo.exists():
            with st.spinner(f"Baixando artefato: {nome_arquivo}..."):
                try:
                    r = requests.get(url, stream=True)
                    r.raise_for_status()
                    with open(caminho_arquivo, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                except requests.exceptions.RequestException as e:
                    st.error(f"Falha ao baixar {nome_arquivo}: {e}"); st.stop()
    
    _model = SentenceTransformer(MODEL_NAME)
    artifacts = {}
    index_files = glob.glob(str(DADOS_PATH / '*_faiss_index_final.bin'))
    for index_file_path in index_files:
        category = Path(index_file_path).stem.replace('_faiss_index_final', '')
        chunks_file_path = DADOS_PATH / f"{category}_chunks_map_final.json"
        try:
            index = faiss.read_index(index_file_path)
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
        except Exception as e:
            logger.error(f"Erro ao carregar '{category}': {e}"); continue
    
    summary_data = None
    summary_file_path = DADOS_PATH / 'resumo_fatos_e_topicos_final_enriquecido.json'
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo de resumo '{summary_file_path}' não encontrado.")
            
    return _model, artifacts, summary_data

@st.cache_data
def criar_mapa_de_alias():
    alias_to_canonical = {}
    for section, topics in DICIONARIO_UNIFICADO_HIERARQUICO.items():
        for canonical_name, aliases in topics.items():
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical


# --- 3. Funções de Lógica de Negócio (Manipuladores de Query) ---

def handle_analytical_query(query: str, summary_data: dict):
    query_lower = query.lower()
    op_keywords = {'avg': ['medio', 'média', 'típico'], 'min': ['menor', 'mínimo'], 'max': ['maior', 'máximo']}
    fact_keywords = {
        'periodo_vesting': ['vesting', 'período de carência'], 'periodo_lockup': ['lockup', 'lock-up'],
        'desconto_strike_price': ['desconto', 'deságio'], 'prazo_exercicio': ['prazo de exercício']
    }
    operation, target_fact_key = None, None
    for op, keywords in op_keywords.items():
        if any(kw in query_lower for kw in keywords): operation = op; break
    for fact_key, keywords in fact_keywords.items():
        if any(kw in query_lower for kw in keywords): target_fact_key = fact_key; break
    if not operation or not target_fact_key: return False
    
    st.info(f"Analisando: **{operation.upper()}** para o fato **'{target_fact_key}'**...")
    valores, unidade = [], ''
    for data in summary_data.values():
        if target_fact_key in data.get("fatos_extraidos", {}):
            fact_data = data["fatos_extraidos"][target_fact_key]
            valor = fact_data.get('valor_numerico') or fact_data.get('valor')
            if isinstance(valor, (int, float)):
                valores.append(valor)
                if not unidade and 'unidade' in fact_data: unidade = fact_data['unidade']
    if not valores:
        st.warning(f"Não encontrei dados numéricos para '{target_fact_key}' para calcular."); return True

    resultado, label_metrica = 0, ""
    if operation == 'avg': resultado, label_metrica = np.mean(valores), f"Média de {target_fact_key.replace('_', ' ')}"
    elif operation == 'min': resultado, label_metrica = np.min(valores), f"Mínimo de {target_fact_key.replace('_', ' ')}"
    elif operation == 'max': resultado, label_metrica = np.max(valores), f"Máximo de {target_fact_key.replace('_', ' ')}"
    
    valor_formatado = f"{resultado:.1%}" if target_fact_key == 'desconto_strike_price' else f"{resultado:.1f} {unidade}".strip()
    st.metric(label=label_metrica.title(), value=valor_formatado)
    st.caption(f"Cálculo baseado em {len(valores)} empresas com dados para este fato.")
    return True

def handle_aggregate_query(query: str, summary_data: dict, alias_map: dict):
    query_lower = query.lower()
    query_keywords = set()
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    temp_query = query_lower
    for alias in sorted_aliases:
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', temp_query):
            query_keywords.add(alias.lower()); temp_query = temp_query.replace(alias.lower(), "")
    if not query_keywords: st.warning("Não identifiquei um termo técnico na sua pergunta."); return
    st.info(f"Termos para busca: **{', '.join(sorted(list(query_keywords)))}**")
    empresas_encontradas = []
    for empresa, data in summary_data.items():
        company_terms = set()
        for topics in data.get("topicos_encontrados", {}).values():
            for topic, aliases in topics.items():
                company_terms.add(topic.lower()); company_terms.update([a.lower() for a in aliases])
        if query_keywords.issubset(company_terms): empresas_encontradas.append(empresa)
    if not empresas_encontradas: st.warning(f"Nenhuma empresa encontrada com: `{', '.join(query_keywords)}`."); return
    st.success(f"✅ **{len(empresas_encontradas)} empresa(s)** encontrada(s).")
    df = pd.DataFrame(sorted(empresas_encontradas), columns=["Empresa"])
    st.dataframe(df, use_container_width=True, hide_index=True)

def handle_direct_fact_query(query: str, summary_data: dict, alias_map: dict, company_catalog: list):
    query_lower, empresa_encontrada, fato_encontrado_alias = query.lower(), None, None
    for company_data in company_catalog:
        for alias in company_data.get("aliases", []):
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                empresa_encontrada = company_data["canonical_name"].upper(); break
        if empresa_encontrada: break
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
            fato_encontrado_alias = alias; break
    if not empresa_encontrada or not fato_encontrado_alias: return False
    empresa_data = summary_data.get(empresa_encontrada, {})
    st.subheader(f"Fato Direto para: {empresa_encontrada}")
    fato_encontrado = False
    for fact_key, fact_value in empresa_data.get("fatos_extraidos", {}).items():
        if fato_encontrado_alias in fact_key.lower():
            valor, unidade = fact_value.get('valor', ''), fact_value.get('unidade', '')
            st.metric(label=f"Fato: {fact_key.replace('_', ' ').title()}", value=f"{valor} {unidade}".strip())
            fato_encontrado = True; break
    if not fato_encontrado: st.info(f"O tópico '{fato_encontrado_alias}' foi mencionado, mas um fato estruturado não foi extraído.")
    return True

# --- FUNÇÕES DO PIPELINE RAG (VERSÃO FINAL E FUNCIONAL) ---

def create_rag_plan(query: str, company_catalog: list, alias_map: dict):
    """
    (ATUALIZADO) Cria um plano de busca simples para o RAG, identificando empresas e tópicos.
    Não usa mais variáveis globais que foram removidas.
    """
    query_lower = query.lower()
    plan = {"empresas": [], "topicos": []}
    
    # Identifica empresas usando o catálogo
    for company_data in company_catalog:
        for alias in company_data.get("aliases", []):
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                plan["empresas"].append(company_data["canonical_name"])
                break
    
    # Identifica tópicos usando o mapa de alias
    for alias, canonical_name in alias_map.items():
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
            plan["topicos"].append(canonical_name)
    
    plan["topicos"] = list(set(plan["topicos"]))

    if not plan["empresas"]: return None
    if not plan["topicos"]: plan["topicos"] = ["informações gerais do plano de incentivo"]
    
    return plan

def execute_rag_plan(plan: dict, artifacts: dict, model):
    """
    (ATUALIZADO) Executa uma busca semântica robusta.
    Não usa mais funções legadas como search_by_tags ou expand_search_terms.
    """
    full_context, sources, unique_chunks = "", set(), set()

    for empresa in plan["empresas"]:
        full_context += f"--- Contexto Relevante para {empresa.upper()} ---\n\n"
        search_query = f"informações sobre {', '.join(plan['topicos'])} no plano de remuneração da empresa {empresa}"
        query_embedding = model.encode([search_query], normalize_embeddings=True)

        for category, artifact_data in artifacts.items():
            scores, indices = artifact_data['index'].search(query_embedding, TOP_K_SEARCH)
            for i, idx in enumerate(indices[0]):
                if idx == -1 or scores[0][i] < 0.35:
                    continue
                
                mapping = artifact_data["chunks"]["map"][idx]
                
                # A robustez vem daqui: verificamos se o chunk pertence à empresa certa via metadados
                if empresa.upper() == mapping.get("company_name", "").upper():
                    chunk_text = artifact_data["chunks"]["chunks"][idx]
                    if chunk_text not in unique_chunks:
                        source_url = mapping.get("source_url", "Fonte Desconhecida")
                        full_context += f"Fonte: {os.path.basename(source_url)} (Similaridade: {scores[0][i]:.2f})\n{chunk_text}\n\n"
                        unique_chunks.add(chunk_text)
                        sources.add(source_url)
    
    return full_context, sources

def get_final_unified_answer(query: str, context: str):
    """
    (VERSÃO CORRETA) Gera a resposta final usando o contexto recuperado e a API REST do Gemini.
    Esta função é robusta e usa as variáveis globais de configuração.
    """
    if not GEMINI_API_KEY:
        st.error("Chave da API Gemini não configurada. Verifique os segredos do Streamlit.")
        return "Erro: Chave da API não encontrada."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""Você é um consultor especialista em planos de remuneração da CVM. Sua tarefa é responder à pergunta do usuário de forma clara, profissional e em português, baseando-se estritamente no contexto fornecido.

    **Instruções Importantes:**
    1.  Use apenas as informações do 'Contexto Coletado'.
    2.  Se a resposta não estiver no contexto, afirme explicitamente: "A informação não foi encontrada nos documentos analisados.". Não invente dados.
    3.  Use formatação Markdown para melhorar a legibilidade (negrito, listas).

    **Pergunta do Usuário:** "{query}"

    **Contexto Coletado dos Documentos:**
    ---
    {context}
    ---
    
    **Relatório Analítico Detalhado:**
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        # Extrai o texto da resposta da API
        candidate = response.json().get('candidates', [{}])[0]
        content = candidate.get('content', {}).get('parts', [{}])[0]
        return content.get('text', "Não foi possível gerar uma resposta.")
    except requests.exceptions.RequestException as e:
        logger.error(f"ERRO de requisição ao chamar a API Gemini: {e}")
        return f"Erro de comunicação com a API do Gemini. Detalhes: {e}"
    except Exception as e:
        logger.error(f"ERRO inesperado ao processar resposta do Gemini: {e}")
        return f"Ocorreu um erro inesperado ao processar a resposta. Detalhes: {e}"

def handle_rag_query(query: str, artifacts: dict, model, company_catalog: list, alias_map: dict):
    """
    (VERSÃO CORRIGIDA) Orquestra o pipeline RAG, chamando a função de API correta.
    """
    with st.status("Gerando plano de análise RAG...") as status:
        plan = create_rag_plan(query, company_catalog, alias_map)
        if not plan:
            st.error("Não consegui identificar empresas conhecidas na sua pergunta para realizar a análise.")
            return set()
        status.update(label=f"Plano gerado. Analisando para: {', '.join(plan['empresas'])}...")

    with st.spinner("Recuperando e analisando informações..."):
        context, sources = execute_rag_plan(plan, artifacts, model)
        if not context:
            st.warning("Não encontrei informações relevantes nos documentos para esta consulta.")
            return set()
        
        # Chamada para a função de API correta e funcional
        final_answer = get_final_unified_answer(query, context)
        st.markdown(final_answer)
        
    return sources


# --- 4. Aplicação Principal (Interface Streamlit) ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP", page_icon="📄", layout="wide")
    st.title("🤖 Agente de Análise de Planos de Incentivo")

    model, artifacts, summary_data = load_all_artifacts()
    
    if not summary_data:
        st.error(f"Arquivo de resumo não encontrado. Funcionalidades limitadas."); st.stop()
    
    ALIAS_MAP = criar_mapa_de_alias()

    try:
        from catalog_data import company_catalog_rich
        logger.info("Catálogo de empresas 'catalog_data.py' carregado.")
    except ImportError:
        logger.warning("`catalog_data.py` não encontrado. Criando catálogo dinâmico.")
        company_catalog_rich = [{"canonical_name": name, "aliases": [name.split(' ')[0].lower(), name.lower()]} for name in summary_data.keys()]

    st.header("💬 Faça sua pergunta")
    user_query = st.text_input("Sua pergunta:", placeholder="Ex: Qual o desconto médio oferecido?")
    
    if user_query:
        st.markdown("---"); st.subheader("📋 Resultado da Análise")
        
        query_lower = user_query.lower()
        analytical_keywords = ['medio', 'média', 'típico', 'menor', 'mínimo', 'maior', 'máximo']
        aggregate_keywords = ["quais", "quantas", "liste", "mostre"]
        direct_fact_pattern = r'qual\s*(?:é|o|a)\s*.*\s*d[aeo]\s*'
        sources = set()

        # Roteador de Intenção de 4 Níveis
        if any(keyword in query_lower for keyword in analytical_keywords):
            if not handle_analytical_query(query, summary_data):
                 sources = handle_rag_query(query, artifacts, model, company_catalog_rich, ALIAS_MAP)
        elif any(keyword in query_lower for keyword in aggregate_keywords):
            handle_aggregate_query(query, summary_data, ALIAS_MAP)
        elif re.search(direct_fact_pattern, query_lower):
            if not handle_direct_fact_query(query, summary_data, ALIAS_MAP, company_catalog_rich):
                 sources = handle_rag_query(query, artifacts, model, company_catalog_rich, ALIAS_MAP)
        else:
            sources = handle_rag_query(query, artifacts, model, company_catalog_rich, ALIAS_MAP)

        if sources:
             with st.expander(f"📚 Fontes consultadas ({len(sources)})"):
                  st.write(sorted(list(sources)))

if __name__ == "__main__":
    main()
