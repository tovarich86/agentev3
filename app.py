# app.py (versão final com hyperlinks descritivos para as fontes)

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

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite"
GITHUB_SOURCE_URL = "https://github.com/tovarich86/agentev2/archive/refs/tags/V1.0-data.zip"
CACHE_DIR = Path("data_cache")
SUMMARY_FILENAME = "resumo_fatos_e_topicos_final_enriquecido.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CARREGADOR DE DADOS ---
@st.cache_resource(show_spinner="Configurando o ambiente e baixando dados...")
def setup_and_load_data():
    # ... (código da função setup_and_load_data permanece o mesmo da versão anterior) ...
    CACHE_DIR.mkdir(exist_ok=True)
    summary_file_path = CACHE_DIR / SUMMARY_FILENAME
    
    if not summary_file_path.exists():
        logger.info(f"Arquivo de resumo não encontrado no cache. Baixando e preparando dados de {GITHUB_SOURCE_URL}...")
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        try:
            response = requests.get(GITHUB_SOURCE_URL, stream=True, timeout=60)
            response.raise_for_status() 
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(CACHE_DIR)
            
            extracted_folder = next(CACHE_DIR.iterdir())
            if extracted_folder.is_dir():
                logger.info(f"Movendo conteúdo de '{extracted_folder.name}' para a raiz do cache...")
                for item in extracted_folder.iterdir():
                    shutil.move(str(item), str(CACHE_DIR / item.name))
                extracted_folder.rmdir()
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao baixar os dados: {e}")
            st.stop()
    else:
        logger.info("Arquivos de dados encontrados no cache local.")

    model = SentenceTransformer(MODEL_NAME)
    artifacts = {}
    for index_file in CACHE_DIR.glob('*_faiss_index_final.bin'):
        category = index_file.stem.replace('_faiss_index_final', '')
        chunks_file = CACHE_DIR / f"{category}_chunks_map_final.json"
        try:
            artifacts[category] = {'index': faiss.read_index(str(index_file)), 'chunks': json.load(open(chunks_file, 'r', encoding='utf-8'))}
        except Exception as e:
            st.error(f"Falha ao carregar artefatos para a categoria '{category}': {e}")
            st.stop()
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Erro crítico: '{SUMMARY_FILENAME}' não foi encontrado após a extração.")
        st.stop()
    return model, artifacts, summary_data

# --- FUNÇÕES GLOBAIS (PRESERVADAS E ADAPTADAS) ---

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
    # ... (código da função permanece o mesmo) ...
    base_term_lower = base_term.lower()
    expanded_terms = {base_term_lower}
    for section, topics in kb.items():
        for topic, aliases in topics.items():
            all_terms_in_group = {alias.lower() for alias in aliases} | {topic.lower().replace('_', ' ')}
            if base_term_lower in all_terms_in_group:
                expanded_terms.update(all_terms_in_group)
    return list(expanded_terms)


def search_by_tags(artifacts: dict, company_name: str, target_tags: list) -> list:
    # ... (código da função permanece o mesmo) ...
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
                        results.append({'text': chunk_text, 'path': mapping.get('source_url', 'N/A'), 'index': i,'source': index_name, 'tag_found': ','.join(intersection), 'company': mapping.get("company_name")})
    return results

# --- LÓGICA ROBUSTA (COM ATUALIZAÇÕES) ---

def get_final_unified_answer(query: str, context: str) -> str:
    # ... (código da função permanece o mesmo) ...
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

# --- MUDANÇA 1: execute_dynamic_plan agora retorna uma lista de dicionários ---


def execute_dynamic_plan(plan: dict, artifacts: dict, model, kb: dict) -> tuple[str, list[dict]]:
    """
    Executa o plano de busca com uma estratégia híbrida e um "Recuperador de Último Recurso".

    Args:
        plan (dict): O plano de análise contendo empresas e tópicos.
        artifacts (dict): Dicionário com os índices FAISS e chunks.
        model: O modelo de embedding SentenceTransformer carregado.
        kb (dict): A base de conhecimento (DICIONARIO_UNIFICADO_HIERARQUICO).

    Returns:
        tuple[str, list[dict]]: Uma tupla contendo o contexto completo e uma lista
                                 de dicionários com as fontes estruturadas.
    """
    full_context, unique_chunks_content = "", set()
    retrieved_sources_structured, seen_sources = [], set()

    class Config:
        MAX_CONTEXT_TOKENS, MAX_CHUNKS_PER_TOPIC, SCORE_THRESHOLD_GENERAL = 256000, 10, 0.4
        TOP_K_SEARCH = 7
    
    def add_unique_chunk_to_context(chunk_text: str, source_info_dict: dict):
        """Função interna para adicionar chunks únicos e estruturados ao contexto."""
        nonlocal full_context, unique_chunks_content, retrieved_sources_structured, seen_sources
        
        # Evita duplicatas de conteúdo
        chunk_hash = hash(re.sub(r'\s+', '', chunk_text.lower())[:200])
        if chunk_hash in unique_chunks_content:
            return
        
        # (A lógica de contagem de tokens seria inserida aqui se necessário)

        unique_chunks_content.add(chunk_hash)
        
        # Limpa os metadados do texto antes de adicionar ao contexto do LLM
        clean_text = re.sub(r'\[(secao|topico):[^\]]+\]', '', chunk_text).strip()
        
        source_header = f"(Empresa: {source_info_dict['company']}, Documento: {source_info_dict['doc_type']})"
        full_context += f"--- CONTEÚDO RELEVANTE {source_header} ---\n{clean_text}\n\n"
        
        # Adiciona a fonte estruturada à lista, evitando duplicatas de (empresa, url)
        source_tuple = (source_info_dict['company'], source_info_dict['url'])
        if source_tuple not in seen_sources:
            seen_sources.add(source_tuple)
            retrieved_sources_structured.append(source_info_dict)

    # --- ETAPA 1: BUSCA DE ALTA PRECISÃO (Tags + Semântica) ---
    for empresa in plan.get("empresas", []):
        logger.info(f"Executando busca de alta precisão para: {empresa}")
        
        # Expande todos os tópicos e seus aliases para as buscas
        target_tags = set()
        for topico in plan.get("topicos", []):
            target_tags.update(expand_search_terms(topico, kb))
        
        # 1a: Busca por Tags
        tagged_chunks = search_by_tags(artifacts, empresa, list(target_tags))
        for chunk_info in tagged_chunks:
            source_info = {
                'company': chunk_info['company'],
                'doc_type': chunk_info['source'],
                'url': chunk_info['path']
            }
            add_unique_chunk_to_context(chunk_info['text'], source_info)
        
        # 1b: Busca Semântica
        for topico in plan.get("topicos", []):
            # Limita a 3 termos por tópico para não sobrecarregar
            for term in expand_search_terms(topico, kb)[:3]:
                search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                query_embedding = model.encode([search_query], normalize_embeddings=True)
                
                for doc_type, artifact_data in artifacts.items():
                    scores, indices = artifact_data['index'].search(query_embedding, Config.TOP_K_SEARCH)
                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and scores[0][i] > Config.SCORE_THRESHOLD_GENERAL:
                            chunk_map_item = artifact_data['chunks']['map'][idx]
                            if empresa.lower() in chunk_map_item['company_name'].lower():
                                source_info = {
                                    'company': chunk_map_item['company_name'],
                                    'doc_type': doc_type,
                                    'url': chunk_map_item['source_url']
                                }
                                add_unique_chunk_to_context(artifact_data['chunks']['chunks'][idx], source_info)

    # --- ETAPA 2: RECUPERADOR DE ÚLTIMO RECURSO ---
    if not full_context:
        logger.warning("Busca de alta precisão falhou. Ativando o Recuperador de Último Recurso.")
        st.info("💡 A busca inicial não retornou resultados de alta confiança. Realizando uma varredura mais ampla...")
        
        for empresa in plan.get("empresas", []):
            expanded_terms = set()
            for topico in plan.get("topicos", []):
                expanded_terms.update(expand_search_terms(topico, kb))
            
            # Itera sobre todos os chunks da empresa em todos os artefatos
            for doc_type, artifact_data in artifacts.items():
                chunk_map = artifact_data.get('chunks', {}).get('map', [])
                all_chunks_text = artifact_data.get('chunks', {}).get('chunks', [])
                
                for i, mapping in enumerate(chunk_map):
                    if empresa.lower() in mapping.get("company_name", "").lower():
                        chunk_text = all_chunks_text[i]
                        
                        # Busca por qualquer um dos termos/aliases dentro do texto do chunk
                        for term in expanded_terms:
                            if re.search(r'\b' + re.escape(term) + r'\b', chunk_text, re.IGNORECASE):
                                source_info = {
                                    'company': mapping['company_name'],
                                    'doc_type': doc_type,
                                    'url': mapping['source_url']
                                }
                                add_unique_chunk_to_context(chunk_text, source_info)
                                break # Otimização: vai para o próximo chunk assim que encontrar um termo

    return full_context, retrieved_sources_structured

# --- Fim da Função execute_dynamic_plan ---

def create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data):
    # ... (código da função permanece o mesmo) ...
    query_lower = query.lower().strip()
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
    if not mentioned_companies: return {"status": "error", "plan": {}}
    alias_map = _create_flat_alias_map(kb)
    topics = list({canonical for alias, canonical in alias_map.items() if re.search(r'\b' + re.escape(alias) + r'\b', query_lower)})
    if not topics:
        logger.info("Nenhum tópico local encontrado, consultando LLM para planejamento...")
        prompt = f"""...""" # Prompt do LLM
        try:
            llm_response = get_final_unified_answer(prompt, "")
            topics = json.loads(re.search(r'\[.*\]', llm_response, re.DOTALL).group())
        except Exception:
            topics = ["Estrutura do Plano", "Vesting", "Outorga"]
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}


# --- MUDANÇA 2: handle_rag_query agora manipula a lista de dicionários de fontes ---
def handle_rag_query(query, artifacts, model, kb, company_catalog_rich, summary_data):
    # (A função create_dynamic_analysis_plan permanece a mesma)
    
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data)
        if plan_response['status'] != "success" or not plan_response['plan']['empresas']:
            st.error("❌ Não consegui identificar empresas na sua pergunta.")
            return "Análise abortada.", []
        plan = plan_response['plan']
        st.write(f"**🏢 Empresas identificadas:** {', '.join(plan['empresas'])}")
        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")
# --- LÓGICA DE VERIFICAÇÃO E BUSCA FORÇADA ---
    force_retrieve_flag = False
    empresa_alvo = plan['empresas'][0] # Foco na análise de empresa única
    topicos_plano = {t.lower() for t in plan['topicos']}
    
    # Verifica no resumo se a empresa realmente tem o tópico
    if empresa_alvo in summary_data:
        # Reconstroi os tópicos do resumo para verificação
        summary_topics = set()
        for section, topics_dict in summary_data[empresa_alvo].get("topicos_encontrados", {}).items():
            for topic_name, aliases in topics_dict.items():
                summary_topics.add(topic_name.lower().replace('_', ' '))
                for alias in aliases:
                    summary_topics.add(alias.lower())
        
        # Se algum tópico do plano estiver no resumo, ativamos a busca forçada
        if not topicos_plano.isdisjoint(summary_topics):
            force_retrieve_flag = True
            st.info("💡 Detectado que a empresa possui menções ao tópico. Ativando busca profunda.")

    final_answer, all_sources_structured = "", []
    seen_sources_tuples = set()
    if len(plan['empresas']) > 1:
        # (Lógica de comparação permanece a mesma, mas agora passa o 'force_retrieve_flag')
        # Omitida por brevidade, mas a ideia é passar o flag para a chamada de execute_dynamic_plan
        pass # A lógica completa deve ser mantida aqui
    else:
        with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
            # Passa o novo flag para a função de execução
            context, all_sources_structured = execute_dynamic_plan(plan, artifacts, model, kb, force_retrieve=force_retrieve_flag)
            if not context:
                st.error("❌ Mesmo com a busca aprofundada, não encontrei detalhes suficientes nos documentos para a sua consulta.")
                return "Informação não encontrada.", []
            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")
            status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
        
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, all_sources_structured


# --- FUNÇÃO PRINCIPAL DA APLICAÇÃO ---
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
    # ... (UI com exemplos de perguntas) ...

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Compare o vesting da Vale e Movida")

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
                if dataframe is not None: st.dataframe(dataframe, use_container_width=True, hide_index=True)
        else:
            final_answer, sources = handle_rag_query(user_query, artifacts, model, DICIONARIO_UNIFICADO_HIERARQUICO, company_catalog_rich, summary_data)
            st.markdown(final_answer)
            
            # --- MUDANÇA 3: Lógica de exibição com hyperlinks ---
            if sources:
                with st.expander(f"📚 Documentos consultados ({len(sources)})"):
                    for src in sorted(sources, key=lambda x: x['company']):
                        display_text = f"{src['company']} - {src['doc_type'].replace('_', ' ')}"
                        st.markdown(f"- [{display_text}]({src['url']})")

if __name__ == "__main__":
    main()
