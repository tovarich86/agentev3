# app.py (versão com Melhoria 1 e 2)

import streamlit as st
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
import re
import unicodedata
import logging
from pathlib import Path
import zipfile
import io
import shutil
from concurrent.futures import ThreadPoolExecutor # <<< MELHORIA 4 ADICIONADA
from tools import (
    find_companies_by_topic,
    get_final_unified_answer,
    suggest_alternative_query,
    analyze_topic_thematically, 
    get_summary_for_topic_at_company,
    rerank_with_cross_encoder,
    _create_alias_to_canonical_map, 
    _get_all_canonical_topics_from_text # <-- CORREÇÃO: Import adicionado
)

# --- Módulos do Projeto (devem estar na mesma pasta) ---
from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
from analytical_engine import AnalyticalEngine

# --- Configurações Gerais ---
st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite"
CVM_SEARCH_URL = "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"

FILES_TO_DOWNLOAD = {
    "item_8_4_chunks_map_final.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/item_8_4_chunks_map_final.json",
    "item_8_4_faiss_index_final.bin": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/item_8_4_faiss_index_final.bin",
    "outros_documentos_chunks_map_final.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/outros_documentos_chunks_map_final.json",
    "outros_documentos_faiss_index_final.bin": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/outros_documentos_faiss_index_final.bin",
    "resumo_fatos_e_topicos_final_enriquecido.json": "https://github.com/tovarich86/agentev2/releases/download/V1.0-data/resumo_fatos_e_topicos_final_enriquecido.json"
}
CACHE_DIR = Path("data_cache")
SUMMARY_FILENAME = "resumo_fatos_e_topicos_final_enriquecido.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CARREGADOR DE DADOS ---
@st.cache_resource(show_spinner="Configurando o ambiente e baixando dados...")
def setup_and_load_data():
    CACHE_DIR.mkdir(exist_ok=True)
    
    for filename, url in FILES_TO_DOWNLOAD.items():
        local_path = CACHE_DIR / filename
        if not local_path.exists():
            logger.info(f"Baixando arquivo '{filename}'...")
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"'{filename}' baixado com sucesso.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao baixar {filename} de {url}: {e}")
                st.stop()
    # --- Carregamento de Modelos ---
    st.write("Carregando modelo de embedding...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    st.write("Carregando modelo de Re-ranking (Cross-Encoder)...")
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    artifacts = {}
    for index_file in CACHE_DIR.glob('*_faiss_index_final.bin'):
        category = index_file.stem.replace('_faiss_index_final', '')
        chunks_file = CACHE_DIR / f"{category}_chunks_map_final.json"
        try:
            artifacts[category] = {
                'index': faiss.read_index(str(index_file)),
                'chunks': json.load(open(chunks_file, 'r', encoding='utf-8'))
            }
        except Exception as e:
            st.error(f"Falha ao carregar artefatos para a categoria '{category}': {e}")
            st.stop()

    summary_file_path = CACHE_DIR / SUMMARY_FILENAME
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Erro crítico: '{SUMMARY_FILENAME}' não foi encontrado.")
        st.stop()
        
    return embedding_model, cross_encoder_model, artifacts, summary_data



# --- FUNÇÕES GLOBAIS E DE RAG ---

def _create_flat_alias_map(kb: dict) -> dict:
    alias_to_canonical = {}
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            canonical_name = topic_name_raw.replace('_', ' ')
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

AVAILABLE_TOPICS = list(set(_create_flat_alias_map(DICIONARIO_UNIFICADO_HIERARQUICO).values()))

def expand_search_terms(base_term: str, kb: dict) -> list[str]:
    base_term_lower = base_term.lower()
    expanded_terms = {base_term_lower}
    for section, topics in kb.items():
        for topic, aliases in topics.items():
            all_terms_in_group = {alias.lower() for alias in aliases} | {topic.lower().replace('_', ' ')}
            if base_term_lower in all_terms_in_group:
                expanded_terms.update(all_terms_in_group)
    return list(expanded_terms)

# Em app.py, substitua esta função
def search_by_tags(artifacts: dict, company_name: str, target_tags: list) -> list:
    results = []
    searchable_company_name = unicodedata.normalize('NFKD', company_name.lower()).encode('ascii', 'ignore').decode('utf-8').split(' ')[0]
    target_tags_lower = {tag.lower() for tag in target_tags}
    
    for index_name, artifact_data in artifacts.items():
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        all_chunks_text = artifact_data.get('chunks', {}).get('chunks', [])
        for i, mapping in enumerate(chunk_map):
            if searchable_company_name in mapping.get("company_name", "").lower():
                chunk_text = all_chunks_text[i]
                found_topics_in_chunk = re.findall(r'\[topico:([^\]]+)\]', chunk_text)
                if found_topics_in_chunk:
                    topics_in_chunk_set = {t.lower() for t in found_topics_in_chunk[0].split(',')}
                    intersection = target_tags_lower.intersection(topics_in_chunk_set)
                    if intersection:
                        # --- CORREÇÃO APLICADA AQUI ---
                        # Padroniza o dicionário de saída para usar as chaves corretas
                        result_item = {
                            "text": all_chunks_text[i],
                            "company_name": mapping.get("company_name"),
                            "source_url": mapping.get("source_url", "N/A"),
                            # Mantém informações úteis para a função que a chama
                            "doc_type": index_name 
                        }
                        results.append(result_item)
    return results

def get_final_unified_answer(query: str, context: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    has_complete_8_4 = "formulário de referência" in query.lower() and "8.4" in query.lower()
    has_tagged_chunks = "--- CONTEÚDO RELEVANTE" in context
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
    if has_complete_8_4:
        structure_instruction = "ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4: Use a estrutura oficial do item 8.4 do Formulário de Referência (a, b, c...)."
    elif has_tagged_chunks:
        structure_instruction = "PRIORIZE as informações dos chunks recuperados e organize a resposta de forma lógica."
    prompt = f"""Você é um consultor especialista em planos de incentivo de longo prazo (ILP).
    PERGUNTA ORIGINAL DO USUÁRIO: "{query}"
    CONTEXTO COLETADO DOS DOCUMENTOS:
    {context}
    {structure_instruction}
    INSTRUÇÕES PARA O RELATÓRIO FINAL:
    1. Responda diretamente à pergunta do usuário com base no contexto fornecido.
    2. Seja detalhado, preciso e profissional na sua linguagem. Use formatação Markdown.
    3. Se uma informação específica pedida não estiver no contexto, declare explicitamente: "Informação não encontrada nas fontes analisadas.". Não invente dados.
    RELATÓRIO ANALÍTICO FINAL:"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"Ocorreu um erro ao contatar o modelo de linguagem. Detalhes: {str(e)}"

# <<< MELHORIA 1 ADICIONADA >>>
def get_query_intent_with_llm(query: str) -> str:
    """
    Usa um LLM para classificar a intenção do usuário em 'quantitativa' ou 'qualitativa'.
    Retorna 'qualitativa' como padrão em caso de erro.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
    Analise a pergunta do usuário e classifique a sua intenção principal. Responda APENAS com uma única palavra em JSON.
    
    As opções de classificação são:
    1. "quantitativa": Se a pergunta busca por números, listas diretas, contagens, médias, estatísticas ou agregações. 
       Exemplos: "Quantas empresas têm TSR Relativo?", "Qual a média de vesting?", "Liste as empresas com desconto no strike.".
    2. "qualitativa": Se a pergunta busca por explicações, detalhes, comparações, descrições ou análises aprofundadas.
       Exemplos: "Como funciona o plano da Vale?", "Compare os planos da Hypera e Movida.", "Detalhe o tratamento de dividendos.".

    Pergunta do Usuário: "{query}"

    Responda apenas com o JSON da classificação. Exemplo de resposta: {{"intent": "qualitativa"}}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 50
        }
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        
        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        intent_json = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group())
        intent = intent_json.get("intent", "qualitativa").lower()
        
        logger.info(f"Intenção detectada pelo LLM: '{intent}' para a pergunta: '{query}'")
        
        if intent in ["quantitativa", "qualitativa"]:
            return intent
        else:
            logger.warning(f"Intenção não reconhecida '{intent}'. Usando 'qualitativa' como padrão.")
            return "qualitativa"

    except Exception as e:
        logger.error(f"ERRO ao determinar intenção com LLM: {e}. Usando 'qualitativa' como padrão.")
        return "qualitativa"

# <<< MELHORIA 2 APLICADA >>>
# Função modificada para lidar com buscas gerais (sem empresa)
# Em app.py, substitua esta função
def execute_dynamic_plan(
    query: str, 
    plan: dict, 
    artifacts: dict, 
    model: SentenceTransformer, 
    cross_encoder_model: CrossEncoder, 
    kb: dict, 
    is_summary_plan: bool = False
) -> tuple[str, list[dict]]:
    """
    Executa o plano de busca de forma inteligente. Coleta uma lista ampla de chunks 
    candidatos (seja para empresas específicas ou em buscas gerais), usa um re-ranker 
    para selecionar os mais relevantes, e constrói o contexto final para o LLM.
    """
    candidate_chunks = {}
    TOP_K_INITIAL_RETRIEVAL = 25 # Aumentamos para ter uma base maior para o re-ranker

    def add_candidate(chunk_text, source_info):
        """Função auxiliar para adicionar um chunk à lista de candidatos, evitando duplicatas."""
        chunk_hash = hash(chunk_text)
        if chunk_hash not in candidate_chunks:
            candidate_chunks[chunk_hash] = {"text": chunk_text, **source_info}

    empresas = plan.get("empresas", [])
    topicos = plan.get("topicos", [])

    # --- LÓGICA DE COLETA DE CANDIDATOS ---

    # Cenário 1: Busca geral (sem empresa especificada, mas com tópicos)
    # Responde a perguntas como "Quais são os modelos típicos de vesting?"
    if not empresas and topicos:
        logger.info(f"Busca geral iniciada para os tópicos: {topicos}")
        for topico in topicos:
            # Cria uma query de busca genérica para o tópico
            search_query = f"explicação detalhada sobre o conceito e funcionamento de {topico}"
            query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
            
            # Busca em todos os índices de documentos
            for doc_type, artifact_data in artifacts.items():
                scores, indices = artifact_data['index'].search(query_embedding, TOP_K_INITIAL_RETRIEVAL)
                for i, idx in enumerate(indices[0]):
                    if idx != -1:
                        chunk_map_item = artifact_data['chunks']['map'][idx]
                        chunk_map_item['doc_type'] = doc_type
                        add_candidate(artifact_data['chunks']['chunks'][idx], chunk_map_item)
    
    # Cenário 2: Busca para empresas específicas
    else:
        for empresa in empresas:
            logger.info(f"Coletando candidatos para: {empresa}")
            
            # Sub-cenário 2a: Plano de resumo otimizado
            if is_summary_plan:
                logger.info("Modo de busca para resumo ativado: busca vetorial direta.")
                for topico in topicos:
                    search_query = f"informações detalhadas sobre {topico} no plano da empresa {empresa}"
                    query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                    for doc_type, artifact_data in artifacts.items():
                        scores, indices = artifact_data['index'].search(query_embedding, TOP_K_INITIAL_RETRIEVAL)
                        for i, idx in enumerate(indices[0]):
                            if idx != -1:
                                chunk_map_item = artifact_data['chunks']['map'][idx]
                                if empresa.lower() in chunk_map_item.get('company_name', '').lower():
                                    chunk_map_item['doc_type'] = doc_type
                                    add_candidate(artifact_data['chunks']['chunks'][idx], chunk_map_item)
            
            # Sub-cenário 2b: Busca híbrida padrão
            else:
                logger.info("Modo de busca padrão ativado: busca híbrida com expansão de termos.")
                target_tags = set()
                for topico in topicos:
                    target_tags.update(expand_search_terms(topico, kb))
                
                tagged_chunks = search_by_tags(artifacts, empresa, list(target_tags))
                for chunk_info in tagged_chunks:
                    add_candidate(chunk_info['text'], chunk_info)

                for topico in topicos:
                    for term in expand_search_terms(topico, kb)[:3]:
                        search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                        query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                        for doc_type, artifact_data in artifacts.items():
                            scores, indices = artifact_data['index'].search(query_embedding, TOP_K_INITIAL_RETRIEVAL)
                            for i, idx in enumerate(indices[0]):
                                if idx != -1:
                                    chunk_map_item = artifact_data['chunks']['map'][idx]
                                    if empresa.lower() in chunk_map_item.get('company_name', '').lower():
                                        chunk_map_item['doc_type'] = doc_type
                                        add_candidate(artifact_data['chunks']['chunks'][idx], chunk_map_item)
    
    # --- ETAPA DE RE-RANKING E SÍNTESE ---
    if not candidate_chunks:
        logger.warning("Nenhum chunk candidato encontrado para re-ranking.")
        return "", []
        
    candidate_list = list(candidate_chunks.values())
    logger.info(f"Total de {len(candidate_list)} chunks candidatos únicos encontrados. Re-ranqueando...")
    
    reranked_chunks = rerank_with_cross_encoder(query, candidate_list, cross_encoder_model, top_n=TOP_K_SEARCH)
    
    full_context, retrieved_sources_structured = "", []
    seen_sources = set()

    for chunk in reranked_chunks:
        company_name = chunk.get('company_name', 'N/A')
        source_url = chunk.get('source_url', 'N/A')
        doc_type = chunk.get('doc_type', 'N/A')

        source_header = f"(Empresa: {company_name}, Documento: {doc_type})"
        source_tuple = (company_name, source_url)
        
        clean_text = re.sub(r'\[(secao|topico):[^\]]+\]', '', chunk.get('text', '')).strip()
        full_context += f"--- CONTEÚDO RELEVANTE {source_header} ---\n{clean_text}\n\n"
        
        if source_tuple not in seen_sources:
            seen_sources.add(source_tuple)
            retrieved_sources_structured.append(chunk)
            
    logger.info(f"Contexto final construído a partir de {len(reranked_chunks)} chunks re-ranqueados.")
    return full_context, retrieved_sources_structured

def create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data):
    query_lower = query.lower().strip()
    
    # --- 1. Identificação de Tópicos e Intenção ---
    alias_map, _ = _create_alias_to_canonical_map(kb)
    topics = _get_all_canonical_topics_from_text(query_lower, alias_map)
    
    listing_keywords = ["quais empresas", "liste as empresas", "quais companhias"]
    thematic_keywords = ["modelos típicos", "padrões comuns", "analise os planos", "formas mais comuns"]
    
    is_listing_request = any(keyword in query_lower for keyword in listing_keywords)
    is_thematic_request = any(keyword in query_lower for keyword in thematic_keywords)

    # --- 2. Identificação de Empresa ---
    mentioned_companies = []
    if company_catalog_rich:
        companies_found_by_alias = {}
        for company_data in company_catalog_rich:
            for alias in company_data.get("aliases", []):
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                    score = len(alias.split())
                    canonical_name = company_data["canonical_name"]
                    if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                        companies_found_by_alias[canonical_name] = score
        if companies_found_by_alias:
            mentioned_companies = [c for c, s in sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)]
    if not mentioned_companies:
        for empresa_nome in summary_data.keys():
            if re.search(r'\b' + re.escape(empresa_nome.lower()) + r'\b', query_lower):
                mentioned_companies.append(empresa_nome)

    # --- 3. Lógica de Validação Aprimorada ---
    # Se a intenção NÃO é de listagem ou temática, E nenhuma empresa foi encontrada, então é um erro.
    if not is_listing_request and not is_thematic_request and not mentioned_companies:
        logger.error("A pergunta parece ser sobre uma empresa específica, mas nenhuma foi identificada.")
        return {"status": "error", "plan": {}}

    # Se nenhum tópico foi identificado, cai no fallback do LLM (exceto para listagem/temática que já deveriam ter um tópico)
    if not topics and not is_listing_request and not is_thematic_request:
        logger.info("Nenhum tópico específico encontrado, consultando LLM para planejamento...")
        # ... (lógica de fallback com LLM, se desejar mantê-la)
        pass # Por enquanto, vamos permitir um plano com tópicos vazios que será tratado depois

    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}
    
def analyze_single_company(
    empresa: str, 
    plan: dict, 
    query: str,  # Novo argumento
    artifacts: dict, 
    model: SentenceTransformer, 
    cross_encoder_model: CrossEncoder, # Novo argumento
    kb: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable
) -> dict:
    """
    Executa o plano de análise para uma única empresa e retorna um dicionário estruturado.
    Esta função é projetada para ser executada em um processo paralelo.
    """
    single_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Adicionado o argumento 'is_summary_plan=False' na chamada.
    context, sources_list = execute_dynamic_plan_func(query, single_plan, artifacts, embedding_model, cross_encoder_model, kb, is_summary_plan=False)
    
    result_data = {
        "empresa": empresa,
        "resumos_por_topico": {topico: "Informação não encontrada" for topico in plan['topicos']},
        "sources": sources_list
    }

    if context:
        summary_prompt = f"""
        Com base no CONTEXTO abaixo sobre a empresa {empresa}, crie um resumo para cada um dos TÓPICOS solicitados.
        Sua resposta deve ser APENAS um objeto JSON válido, sem nenhum texto adicional antes ou depois.
        
        TÓPICOS PARA RESUMIR: {json.dumps(plan['topicos'])}
        
        CONTEXTO:
        {context}
        
        FORMATO OBRIGATÓRIO DA RESPOSTA (APENAS JSON):
        {{
          "resumos_por_topico": {{
            "Tópico 1": "Resumo conciso sobre o Tópico 1...",
            "Tópico 2": "Resumo conciso sobre o Tópico 2...",
            "...": "..."
          }}
        }}
        """
        
        try:
            json_response_str = get_final_unified_answer_func(summary_prompt, context)
            json_match = re.search(r'\{.*\}', json_response_str, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                result_data["resumos_por_topico"] = parsed_json.get("resumos_por_topico", result_data["resumos_por_topico"])
            else:
                logger.warning(f"Não foi possível extrair JSON da resposta para a empresa {empresa}.")

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Erro ao processar o resumo JSON para {empresa}: {e}")
            
    return result_data


def handle_rag_query(
    query: str, 
    artifacts: dict, 
    embedding_model: SentenceTransformer,  # <-- CORREÇÃO APLICADA AQUI
    cross_encoder_model: CrossEncoder,
    kb: dict, 
    company_catalog_rich: list, 
    summary_data: dict
) -> tuple[str, list[dict]]:
    """
    Orquestra o pipeline de RAG para perguntas qualitativas, incluindo a geração do plano,
    a execução da busca (com re-ranking) e a síntese da resposta final.
    """
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data)
        
        if plan_response['status'] != "success":
            status.update(label="⚠️ Falha na identificação", state="error", expanded=True)
            
            st.warning("Não consegui identificar uma empresa conhecida na sua pergunta para realizar uma análise profunda.")
            st.info("Para análises detalhadas, por favor, use o nome de uma das empresas listadas na barra lateral.")
            
            with st.spinner("Estou pensando em uma pergunta alternativa que eu possa responder..."):
                alternative_query = suggest_alternative_query(query)
            
            st.markdown("#### Que tal tentar uma pergunta mais geral?")
            st.markdown("Você pode copiar a sugestão abaixo ou reformular sua pergunta original.")
            st.code(alternative_query, language=None)
            
            # Retornamos uma string vazia para o texto e para as fontes, encerrando a análise de forma limpa.
            return "", []
        # --- FIM DO NOVO BLOCO ---
            
        plan = plan_response['plan']
        
        summary_keywords = ['resumo', 'geral', 'completo', 'visão geral', 'como funciona o plano', 'detalhes do plano']
        is_summary_request = any(keyword in query.lower() for keyword in summary_keywords)
        
        specific_topics_in_query = list({canonical for alias, canonical in _create_flat_alias_map(kb).items() if re.search(r'\b' + re.escape(alias) + r'\b', query.lower())})
        is_summary_plan = is_summary_request and not specific_topics_in_query
        
        if plan['empresas']:
            st.write(f"**🏢 Empresas identificadas:** {', '.join(plan['empresas'])}")
        else:
            st.write("**🏢 Nenhuma empresa específica identificada. Realizando busca geral.**")
            
        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        if is_summary_plan:
            st.info("💡 Modo de resumo geral ativado. A busca será otimizada para os tópicos encontrados.")
            
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    final_answer, all_sources_structured = "", []
    seen_sources_tuples = set()

    # --- Lógica para Múltiplas Empresas (Comparação) ---
    if len(plan.get('empresas', [])) > 1:
        st.info(f"Modo de comparação ativado para {len(plan['empresas'])} empresas. Executando análises em paralelo...")
        
        with st.spinner(f"Analisando {len(plan['empresas'])} empresas..."):
            with ThreadPoolExecutor(max_workers=len(plan['empresas'])) as executor:
                futures = [
                    executor.submit(
                        analyze_single_company, 
                        empresa, plan, query, artifacts, embedding_model, cross_encoder_model, kb,
                        execute_dynamic_plan, get_final_unified_answer
                    ) 
                    for empresa in plan['empresas']
                ]
                results = [future.result() for future in futures]

        for result in results:
            for src_dict in result.get('sources', []):
                company_name = src_dict.get('company_name')
                source_url = src_dict.get('source_url')
                
                if company_name and source_url:
                    src_tuple = (company_name, source_url)
                    if src_tuple not in seen_sources_tuples:
                        seen_sources_tuples.add(src_tuple)
                        all_sources_structured.append(src_dict)

        with st.status("Gerando relatório comparativo final...", expanded=True) as status:
            structured_context = json.dumps(results, indent=2, ensure_ascii=False)
            comparison_prompt = f"""
            Sua tarefa é criar um relatório comparativo detalhado sobre "{query}".
            Use os dados estruturados fornecidos no CONTEXTO JSON abaixo.
            O relatório deve começar com uma breve análise textual e, em seguida, apresentar uma TABELA MARKDOWN clara e bem formatada.

            CONTEXTO (em formato JSON):
            {structured_context}
            """
            final_answer = get_final_unified_answer(comparison_prompt, structured_context)
            status.update(label="✅ Relatório comparativo gerado!", state="complete")
            
    # --- Lógica para Empresa Única ou Busca Geral ---
    else:
        with st.status("2️⃣ Recuperando e re-ranqueando contexto...", expanded=True) as status:
            context, all_sources_structured = execute_dynamic_plan(
                query, plan, artifacts, embedding_model, cross_encoder_model, kb, is_summary_plan=is_summary_plan
            )
            
            if not context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", []
                
            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")
            status.update(label="✅ Contexto relevante selecionado!", state="complete")
        
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, all_sources_structured

def main():
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    st.markdown("---")

    embedding_model, cross_encoder_model, artifacts, summary_data = setup_and_load_data()
        
    if not summary_data or not artifacts:
        st.error("❌ Falha crítica no carregamento dos dados. O app não pode continuar.")
        st.stop()
    
    engine = AnalyticalEngine(summary_data, DICIONARIO_UNIFICADO_HIERARQUICO) 
    
    try:
        from catalog_data import company_catalog_rich 
    except ImportError:
        company_catalog_rich = [] 
    
    st.session_state.company_catalog_rich = company_catalog_rich

    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Categorias de Documentos (RAG)", len(artifacts))
        st.metric("Empresas no Resumo", len(summary_data))
        with st.expander("Empresas com dados no resumo"):
            st.dataframe(pd.DataFrame(sorted(list(summary_data.keys())), columns=["Empresa"]), use_container_width=True, hide_index=True)
        st.success("✅ Sistema pronto para análise")
        st.info(f"Embedding Model: `{MODEL_NAME}`")
        st.info(f"Generative Model: `{GEMINI_MODEL}`")
    
    st.header("💬 Faça sua pergunta")
    
    # Em app.py, localize o bloco `with st.expander(...)` e substitua seu conteúdo por este:

    with st.expander("ℹ️ **Guia do Usuário: Como Extrair o Máximo do Agente**", expanded=False): # `expanded=False` é uma boa prática para não poluir a tela inicial
        st.markdown("""
        Este agente foi projetado para atuar como um consultor especialista em Planos de Incentivo de Longo Prazo (ILP), analisando uma base de dados de documentos públicos da CVM. Para obter os melhores resultados, formule perguntas que explorem suas principais capacidades.
        """)

        st.subheader("1. Perguntas de Listagem (Quem tem?) 🎯")
        st.info("""
        Use estas perguntas para identificar e listar empresas que adotam uma prática específica. Ideal para mapeamento de mercado.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Liste as empresas que pagam dividendos ou JCP durante o período de carência (vesting).
        - Quais companhias possuem cláusulas de Malus ou Clawback?
        - Gere uma lista de empresas que oferecem planos com contrapartida do empregador (Matching/Coinvestimento).
        - Quais organizações mencionam explicitamente o Comitê de Remuneração como órgão aprovador dos planos?""")

        st.subheader("2. Análise Estatística (Qual a média?) 📈")
        st.info("""
        Pergunte por médias, medianas e outros dados estatísticos para entender os números por trás das práticas de mercado e fazer benchmarks.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Qual o período médio de vesting (em anos) entre as empresas analisadas?
        - Qual a diluição máxima média (% do capital social) que os planos costumam aprovar?
        - Apresente as estatísticas do desconto no preço de exercício (mínimo, média, máximo).
        - Qual o prazo de lock-up (restrição de venda) mais comum após o vesting das ações?""")

        st.subheader("3. Padrões de Mercado (Como é o normal?) 🗺️")
        st.info("""
        Faça perguntas abertas para que o agente analise diversos planos e descreva os padrões e as abordagens mais comuns para um determinado tópico.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Analise os modelos típicos de planos de Ações Restritas (RSU), o tipo mais comum no mercado.
        - Além do TSR, quais são as metas de performance (ESG, Financeiras) mais utilizadas pelas empresas?
        - Descreva os padrões de tratamento para condições de saída (Good Leaver vs. Bad Leaver) nos planos.
        - Quais as abordagens mais comuns para o tratamento de dividendos em ações ainda não investidas?""")

        st.subheader("4. Análise Profunda e Comparativa (Me explique em detalhes) 🧠")
        st.info("""
        Use o poder do RAG para pedir análises detalhadas sobre uma ou mais empresas, comparando regras e estruturas específicas.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Como o plano da Petrobras trata a aceleração de vesting em caso de mudança de controle?
        - Compare as cláusulas de Malus/Clawback da Vale com as do Itaú.
        - Descreva em detalhes o plano de Opções de Compra da Localiza, incluindo prazos, condições e forma de liquidação.
        - Quais as diferenças na elegibilidade de participantes entre os planos da Magazine Luiza e da Lojas Renner?""")


        st.subheader("❗ Conhecendo as Limitações")
        st.warning("""
        - **Fonte dos Dados:** Minha análise se baseia em documentos públicos da CVM com data de corte 31/07/2025. Não tenho acesso a informações em tempo real ou privadas.
        - **Identificação de Nomes:** Para análises profundas, preciso que o nome da empresa seja claro e reconhecível. Se o nome for ambíguo ou não estiver na minha base, posso não encontrar os detalhes.
        - **Escopo:** Sou altamente especializado em Incentivos de Longo Prazo. Perguntas fora deste domínio podem não ter respostas adequadas.
        """)

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quais são os modelos típicos de vesting? ou Como funciona o plano da Vale?")
    
    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            st.stop()
        
        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
        
        with st.spinner("Analisando a intenção da sua pergunta..."):
            intent = get_query_intent_with_llm(user_query)

        if intent == "quantitativa":
            query_lower = user_query.lower()
            listing_keywords = ["quais empresas", "liste as empresas", "quais companhias"]
            thematic_keywords = ["modelos típicos", "padrões comuns", "analise os planos", "formas mais comuns"]
            
            alias_map, _ = _create_alias_to_canonical_map(DICIONARIO_UNIFICADO_HIERARQUICO)
            topics_to_search = _get_all_canonical_topics_from_text(query_lower, alias_map)
            topics_to_search = [t for t in topics_to_search if t.lower() not in listing_keywords and t.lower() not in thematic_keywords]

            # Rota 1: Análise Temática
            if any(keyword in query_lower for keyword in thematic_keywords) and topics_to_search:
                primary_topic = topics_to_search[0]
                with st.spinner(f"Iniciando análise temática... Este processo é detalhado e pode levar alguns minutos."):
                    st.write(f"**Tópico identificado para análise temática:** `{topics_to_search}`")
                    final_report = analyze_topic_thematically(
                        topic=topics_to_search, query=user_query, artifacts=artifacts, model=embedding_model, kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                        execute_dynamic_plan_func=execute_dynamic_plan, get_final_unified_answer_func=get_final_unified_answer
                    )
                    st.markdown(final_report)

            # Rota 2: Listagem de Empresas
            elif any(keyword in query_lower for keyword in listing_keywords) and topics_to_search:
                with st.spinner(f"Usando ferramentas para encontrar empresas..."):
                    st.write(f"**Tópicos identificados para busca:** `{', '.join(topics_to_search)}`")
        
                    all_found_companies = set()
        
                    # CORREÇÃO: Itera sobre a lista e chama a ferramenta para CADA tópico.
                    for topic_item in topics_to_search:
                        companies = find_companies_by_topic(
                            topic=topic_item,  # Passa um único tópico (string)
                            artifacts=artifacts, 
                            model=embedding_model, 
                            kb=DICIONARIO_UNIFICADO_HIERARQUICO
                        )
                        all_found_companies.update(companies)

                    if all_found_companies:
                        sorted_companies = sorted(list(all_found_companies))
                        final_answer = f"#### Foram encontradas {len(sorted_companies)} empresas para os tópicos relacionados:\n"
                        final_answer += "\n".join([f"- {company}" for company in sorted_companies])
                    else:
                        final_answer = "Nenhuma empresa encontrada nos documentos para os tópicos identificados."
                
                st.markdown(final_answer)

            # Rota 3: Fallback para o AnalyticalEngine
            else:
                st.info("Intenção quantitativa detectada. Usando o motor de análise rápida...")
                with st.spinner("Executando análise quantitativa rápida..."):
                    report_text, data_result = engine.answer_query(user_query)
                    if report_text: st.markdown(report_text)
                    if data_result is not None:
                        if isinstance(data_result, pd.DataFrame):
                            if not data_result.empty: st.dataframe(data_result, use_container_width=True, hide_index=True)
                        elif isinstance(data_result, dict):
                            for df_name, df_content in data_result.items():
                                if df_content is not None and not df_content.empty:
                                    st.markdown(f"#### {df_name}")
                                    st.dataframe(df_content, use_container_width=True, hide_index=True)
                    else: 
                        st.info("Nenhuma análise tabular foi gerada para a sua pergunta ou dados insuficientes.")
        
        else: # intent == 'qualitativa'
            final_answer, sources = handle_rag_query(
                user_query, 
                artifacts, 
                embedding_model, 
                cross_encoder_model, 
                kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                company_catalog_rich=st.session_state.company_catalog_rich, 
                summary_data=summary_data
            )
            st.markdown(final_answer)
            
            if sources:
                with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=True):
                    st.caption("Nota: Links diretos para a CVM podem falhar. Use a busca no portal com o protocolo como plano B.")
                    
                    # --- BLOCO FINAL CORRIGIDO ---
                    # Usa 'company_name' para ordenar de forma segura e .get() para todas as chaves
                    for src in sorted(sources, key=lambda x: x.get('company_name', '')):
                        company_name = src.get('company_name', 'N/A')
                        doc_type = src.get('doc_type', '').replace('_', ' ')
                        doc_type_raw = src.get('doc_type', '')
                        url = src.get('source_url', '') # Chave correta é 'source_url'

                        if doc_type_raw == 'outros_documentos_':
                            display_doc_type = 'Plano de Remuneração'
                        else:
                            display_doc_type = doc_type_raw.replace('_', ' ')
                    # --- FIM DA LÓGICA APRIMORADA -
                        
                        display_text = f"{company_name} - {doc_type}"
                        
                        if "frmExibirArquivoIPEExterno" in url:
                            protocolo_match = re.search(r'NumeroProtocoloEntrega=(\d+)', url)
                            protocolo = protocolo_match.group(1) if protocolo_match else "N/A"
                            st.markdown(f"**{display_text}** (Protocolo: **{protocolo}**)")
                            st.markdown(f"↳ [Link Direto ]({url}) | [Buscar na CVM]({CVM_SEARCH_URL})", unsafe_allow_html=True)
                        elif "frmExibirArquivoFRE" in url:
                            st.markdown(f"**{display_text}**")
                            st.markdown(f"↳ [Link Direto para Formulário de Referência]({url})", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{display_text}**: [Link]({url})")

if __name__ == "__main__":
    main()
