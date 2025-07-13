# streamlit_app_definitivo_final.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import requests
import re
import unicodedata
import logging
from pathlib import Path
import zipfile
import io
import shutil

# --- Módulos do Projeto ---
from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
from analytical_engine import AnalyticalEngine

# --- Configurações Gerais ---
st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash-latest"

# --- MUDANÇA 1: ATUALIZANDO A URL PARA O ARQUIVO DE CÓDIGO-FONTE ---
GITHUB_SOURCE_URL = "https://github.com/tovarich86/agentev2/archive/refs/tags/V1.0-data.zip"
CACHE_DIR = Path("data_cache")
SUMMARY_FILENAME = "resumo_fatos_e_topicos_final_enriquecido.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CARREGADOR DE DADOS (ATUALIZADO PARA LIDAR COM O NOVO ZIP) ---
@st.cache_resource(show_spinner="Configurando ambiente e baixando dados...")
def setup_and_load_data():
    """
    Baixa o arquivo .zip do código-fonte, extrai e move os arquivos para o
    diretório de cache raiz, garantindo a estrutura de arquivos correta.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    
    expected_files = ['item_8_4_faiss_index_final.bin', 'outros_documentos_faiss_index_final.bin', SUMMARY_FILENAME]
    
    if not all((CACHE_DIR / f).exists() for f in expected_files):
        logger.info(f"Limpando cache antigo e baixando novos artefatos de {GITHUB_SOURCE_URL}...")
        # Limpa o diretório de cache para garantir uma extração limpa
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)

        try:
            response = requests.get(GITHUB_SOURCE_URL, stream=True, timeout=30)
            response.raise_for_status() 
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(CACHE_DIR)
            logger.info("Artefatos baixados e extraídos para o cache.")

            # --- MUDANÇA 2: LÓGICA PARA LIDAR COM A PASTA EXTRA ---
            # O zip cria uma pasta como 'agentev2-V1.0-data'. Precisamos mover seu conteúdo para a raiz do cache.
            # `next(CACHE_DIR.iterdir())` pega o primeiro (e único) item no cache, que é a pasta extraída.
            extracted_folder = next(CACHE_DIR.iterdir())
            if extracted_folder.is_dir():
                logger.info(f"Movendo conteúdo de '{extracted_folder.name}' para a raiz do cache...")
                # Itera sobre todos os arquivos e pastas dentro da pasta extraída
                for item_to_move in extracted_folder.iterdir():
                    # Move cada item para o diretório de cache principal
                    shutil.move(str(item_to_move), str(CACHE_DIR / item_to_move.name))
                # Remove a pasta agora vazia
                extracted_folder.rmdir()
            
        except requests.exceptions.HTTPError as e:
            st.error(f"Erro de HTTP ao tentar baixar os dados: {e.response.status_code} {e.response.reason}")
            st.code(f"URL: {GITHUB_SOURCE_URL}")
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"Erro de Conexão: {e}")
            st.stop()
    else:
        logger.info("Arquivos de dados encontrados no cache local.")

    # O resto do carregamento agora funciona porque os arquivos estão no lugar certo
    model = SentenceTransformer(MODEL_NAME)
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
            logger.error(f"Falha ao carregar artefatos para '{category}': {e}")
            
    summary_data = json.load(open(CACHE_DIR / SUMMARY_FILENAME, 'r', encoding='utf-8'))
    return model, artifacts, summary_data


#
# --- TODAS AS OUTRAS FUNÇÕES (handle_rag_query, etc.) PERMANECEM EXATAMENTE IGUAIS À VERSÃO ANTERIOR ---
# (O código completo e robusto que preservamos)
#

def _create_flat_alias_map(kb: dict) -> dict:
    alias_to_canonical = {}
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            canonical_name = topic_name_raw.replace('_', ' ')
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

AVAILABLE_TOPICS = list(_create_flat_alias_map(DICIONARIO_UNIFICADO_HIERARQUICO).values())

def expand_search_terms(base_term: str, kb: dict) -> list[str]:
    base_term_lower = base_term.lower()
    expanded_terms = {base_term_lower}
    for section, topics in kb.items():
        for topic, aliases in topics.items():
            all_terms_in_group = {alias.lower() for alias in aliases} | {topic.lower().replace('_', ' ')}
            if base_term_lower in all_terms_in_group:
                expanded_terms.update(all_terms_in_group)
    return list(expanded_terms)

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
                        results.append({'text': chunk_text, 'path': mapping.get('source_url', 'N/A'), 'index': i,'source': index_name, 'tag_found': ','.join(intersection)})
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

def execute_dynamic_plan(plan: dict, artifacts: dict, model, kb: dict) -> tuple[str, set]:
    full_context, all_retrieved_docs, unique_chunks_content = "", set(), set()
    current_token_count, chunks_processed = 0, 0
    class Config:
        MAX_CONTEXT_TOKENS, MAX_CHUNKS_PER_TOPIC, SCORE_THRESHOLD_GENERAL = 256000, 10, 0.4
    
    def add_unique_chunk_to_context(chunk_text, source_info):
        nonlocal full_context, current_token_count, unique_chunks_content, all_retrieved_docs
        chunk_hash = hash(re.sub(r'\s+', '', chunk_text.lower())[:200])
        if chunk_hash in unique_chunks_content: return
        estimated_tokens = len(chunk_text) // 4
        if current_token_count + estimated_tokens > Config.MAX_CONTEXT_TOKENS: return
        unique_chunks_content.add(chunk_hash)
        clean_text = re.sub(r'\[(secao|topico):[^\]]+\]', '', chunk_text).strip()
        full_context += f"--- CONTEÚDO RELEVANTE (Fonte: {source_info}) ---\n{clean_text}\n\n"
        current_token_count += estimated_tokens
        try: all_retrieved_docs.add(source_info.split('path: ')[1].split(')')[0])
        except IndexError: all_retrieved_docs.add(source_info)

    for empresa in plan.get("empresas", []):
        logger.info(f"Executando plano para: {empresa}")
        target_tags = set()
        for topico in plan.get("topicos", []):
            target_tags.update(expand_search_terms(topico, kb))
        tagged_chunks = search_by_tags(artifacts, empresa, list(target_tags))
        for chunk_info in tagged_chunks:
            add_unique_chunk_to_context(chunk_info['text'], f"(path: {chunk_info['path']})")
        for topico in plan.get("topicos", []):
            for term in expand_search_terms(topico, kb)[:3]:
                search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                query_embedding = model.encode([search_query], normalize_embeddings=True)
                for artifact_data in artifacts.values():
                    scores, indices = artifact_data['index'].search(query_embedding, TOP_K_SEARCH)
                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and scores[0][i] > Config.SCORE_THRESHOLD_GENERAL:
                            chunk_map = artifact_data['chunks']['map']
                            if empresa.lower() in chunk_map[idx]['company_name'].lower():
                                add_unique_chunk_to_context(artifact_data['chunks']['chunks'][idx], f"(path: {chunk_map[idx]['source_url']}, score: {scores[0][i]:.2f})")
    return full_context, all_retrieved_docs

def create_dynamic_analysis_plan(query, company_catalog_rich, kb):
    query_lower = query.lower().strip()
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
            mentioned_companies = [c for c, s in sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)]
    if not mentioned_companies:
        return {"status": "error", "plan": {}}
    alias_map = _create_flat_alias_map(kb)
    topics = list({canonical for alias, canonical in alias_map.items() if re.search(r'\b' + re.escape(alias) + r'\b', query_lower)})
    if not topics:
        logger.info("Nenhum tópico local encontrado, consultando LLM para planejamento...")
        prompt = f"""Você é um consultor de ILP. Identifique os TÓPICOS CENTRAIS da pergunta: "{query}".
        Retorne APENAS uma lista JSON com os tópicos mais relevantes de: {json.dumps(AVAILABLE_TOPICS)}.
        Formato: ["Tópico 1", "Tópico 2"]"""
        try:
            llm_response = get_final_unified_answer(prompt, "")
            topics = json.loads(re.search(r'\[.*\]', llm_response, re.DOTALL).group())
        except Exception as e:
            logger.warning(f"Falha ao obter tópicos do LLM: {e}. Usando tópicos padrão.")
            topics = ["Estrutura do Plano", "Vesting", "Outorga"]
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}

def handle_rag_query(query, artifacts, model, kb, company_catalog_rich):
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb)
        if plan_response['status'] != "success" or not plan_response['plan']['empresas']:
            st.error("❌ Não consegui identificar empresas na sua pergunta.")
            return "Análise abortada.", set()
        plan = plan_response['plan']
        st.write(f"**🏢 Empresas identificadas:** {', '.join(plan['empresas'])}")
        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")
    final_answer, all_sources = "", set()
    if len(plan['empresas']) > 1:
        st.info(f"Modo de comparação ativado para {len(plan['empresas'])} empresas.")
        summaries = []
        for i, empresa in enumerate(plan['empresas']):
            with st.status(f"Analisando {i+1}/{len(plan['empresas'])}: {empresa}...", expanded=True):
                single_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
                context, sources = execute_dynamic_plan(single_plan, artifacts, model, kb)
                all_sources.update(sources)
                if not context:
                    summaries.append(f"## Análise para {empresa.upper()}\n\nNenhuma informação encontrada.")
                else:
                    summary_prompt = f"Com base no contexto a seguir sobre a empresa {empresa}, resuma os pontos principais sobre os tópicos: {', '.join(plan['topicos'])}.\n\nContexto:\n{context}"
                    summaries.append(f"## Análise para {empresa.upper()}\n\n{get_final_unified_answer(summary_prompt, context)}")
        with st.status("Gerando relatório comparativo final...", expanded=True) as status:
            comparison_prompt = f"Com base nos resumos individuais a seguir, crie um relatório comparativo detalhado e bem estruturado sobre '{query}'.\n\n" + "\n\n---\n\n".join(summaries)
            final_answer = get_final_unified_answer(comparison_prompt, "\n\n".join(summaries))
            status.update(label="✅ Relatório comparativo gerado!", state="complete")
    else:
        with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
            context, sources = execute_dynamic_plan(plan, artifacts, model, kb)
            all_sources.update(sources)
            if not context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", set()
            st.write(f"**📄 Contexto recuperado de:** {len(all_sources)} documento(s)")
            status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, context)
            status.update(label="✅ Análise concluída!", state="complete")
    return final_answer, all_sources

def main():
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    st.markdown("---")
    model, artifacts, summary_data = setup_and_load_data()
    if not summary_data or not artifacts:
        st.error("❌ Falha crítica no carregamento dos dados.")
        st.stop()
    engine = AnalyticalEngine(summary_data, DICIONARIO_UNIFICADO_HIERARQUICO)
    try: from catalog_data import company_catalog_rich
    except ImportError: company_catalog_rich = []
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Categorias de Documentos (RAG)", len(artifacts))
        st.metric("Empresas no Resumo", len(summary_data))
        with st.expander("Empresas com dados no resumo"):
            st.dataframe(sorted(list(summary_data.keys())), use_container_width=True, hide_index=False)
        st.success("✅ Sistema pronto para análise")
        st.info(f"Embedding Model: `{MODEL_NAME}`")
        st.info(f"Generative Model: `{GEMINI_MODEL}`")
    st.header("💬 Faça sua pergunta")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Experimente uma análise quantitativa:**")
        st.code("Qual o desconto médio no preço de exercício?")
        st.code("Quais empresas tem TSR Relativo?")
        st.code("Qual o período médio de vesting?")
    with col2:
        st.info("**Ou uma análise profunda:**")
        st.code("Compare o vesting da Vale com a Magazine Luiza")
        st.code("Como funciona o plano de lockup da Movida?")
    st.caption(f"**Principais Termos-Chave:** {', '.join(list(_create_flat_alias_map(DICIONARIO_UNIFICADO_HIERARQUICO).values())[:10])}, etc.")
    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quantas empresas oferecem desconto no strike?")
    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            st.stop()
        st.markdown("---"); st.subheader("📋 Resultado da Análise")
        query_lower = user_query.lower()
        aggregate_keywords = ["quais", "quantas", "liste", "qual a lista", "qual o desconto", "qual a média", "qual é o"]
        if any(keyword in query_lower.split() for keyword in aggregate_keywords):
            with st.spinner("Analisando dados estruturados..."):
                report, dataframe = engine.answer_query(user_query)
                if report: st.markdown(report)
                if dataframe is not None and not dataframe.empty: st.dataframe(dataframe, use_container_width=True, hide_index=True)
        else:
            final_answer, sources = handle_rag_query(user_query, artifacts, model, DICIONARIO_UNIFICADO_HIERARQUICO, company_catalog_rich)
            st.markdown(final_answer)
            if sources:
                with st.expander(f"📚 Documentos consultados ({len(sources)})"):
                    for source in sorted(list(sources)): st.write(f"- `{source}`")

if __name__ == "__main__":
    main()
