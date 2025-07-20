# app.py (versão com Melhoria 1 - Roteador de Intenção LLM)

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
            st.error(f"Falha ao carregar artefatos para a categoria '{category}': {e}")
            st.stop()

    summary_file_path = CACHE_DIR / SUMMARY_FILENAME
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Erro crítico: '{SUMMARY_FILENAME}' não foi encontrado.")
        st.stop()
        
    return model, artifacts, summary_data


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
                        results.append({'text': chunk_text, 'path': mapping.get('source_url', 'N/A'), 'index': i,'source': index_name, 'tag_found': ','.join(intersection), 'company': mapping.get("company_name")})
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

def execute_dynamic_plan(plan: dict, artifacts: dict, model, kb: dict) -> tuple[str, list[dict]]:
    full_context, unique_chunks_content = "", set()
    retrieved_sources_structured, seen_sources = [], set()
    class Config:
        MAX_CONTEXT_TOKENS, SCORE_THRESHOLD_GENERAL = 256000, 0.4
    
    def add_unique_chunk_to_context(chunk_text, source_info_dict):
        nonlocal full_context, unique_chunks_content, retrieved_sources_structured, seen_sources
        chunk_hash = hash(re.sub(r'\s+', '', chunk_text.lower())[:200])
        if chunk_hash in unique_chunks_content: return
        
        estimated_tokens = len(full_context + chunk_text) // 4
        if estimated_tokens > Config.MAX_CONTEXT_TOKENS: return

        unique_chunks_content.add(chunk_hash)
        clean_text = re.sub(r'\[(secao|topico):[^\]]+\]', '', chunk_text).strip()
        source_header = f"(Empresa: {source_info_dict['company']}, Documento: {source_info_dict['doc_type']})"
        source_tuple = (source_info_dict['company'], source_info_dict['url'])
        full_context += f"--- CONTEÚDO RELEVANTE {source_header} ---\n{clean_text}\n\n"
        if source_tuple not in seen_sources:
            seen_sources.add(source_tuple)
            retrieved_sources_structured.append(source_info_dict)

    for empresa in plan.get("empresas", []):
        logger.info(f"Executando plano para: {empresa}")
        target_tags = set()
        for topico in plan.get("topicos", []):
            target_tags.update(expand_search_terms(topico, kb))
        
        tagged_chunks = search_by_tags(artifacts, empresa, list(target_tags))
        for chunk_info in tagged_chunks:
            source_info = {'company': chunk_info['company'],'doc_type': chunk_info['source'],'url': chunk_info['path']}
            add_unique_chunk_to_context(chunk_info['text'], source_info)

        for topico in plan.get("topicos", []):
            for term in expand_search_terms(topico, kb)[:3]:
                search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                for doc_type, artifact_data in artifacts.items():
                    scores, indices = artifact_data['index'].search(query_embedding, TOP_K_SEARCH)
                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and scores[0][i] > Config.SCORE_THRESHOLD_GENERAL:
                            chunk_map_item = artifact_data['chunks']['map'][idx]
                            if empresa.lower() in chunk_map_item['company_name'].lower():
                                source_info = {'company': chunk_map_item['company_name'],'doc_type': doc_type,'url': chunk_map_item['source_url']}
                                add_unique_chunk_to_context(artifact_data['chunks']['chunks'][idx], source_info)
    
    return full_context, retrieved_sources_structured

def create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data):
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
            llm_response = get_final_unified_answer("Gere uma lista de tópicos para a pergunta.", prompt) # Contexto é o prompt aqui
            topics = json.loads(re.search(r'\[.*\]', llm_response, re.DOTALL).group())
        except Exception as e:
            logger.warning(f"Falha ao obter tópicos do LLM: {e}. Usando tópicos padrão.")
            topics = ["Estrutura do Plano", "Vesting", "Outorga"]
            
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}

def handle_rag_query(query, artifacts, model, kb, company_catalog_rich, summary_data):
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data)
        if plan_response['status'] != "success" or not plan_response['plan']['empresas']:
            st.error("❌ Não consegui identificar empresas na sua pergunta.")
            return "Análise abortada.", []
        plan = plan_response['plan']
        st.write(f"**🏢 Empresas identificadas:** {', '.join(plan['empresas'])}")
        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    final_answer, all_sources_structured = "", []
    seen_sources_tuples = set()

    if len(plan['empresas']) > 1:
        st.info(f"Modo de comparação ativado para {len(plan['empresas'])} empresas.")
        summaries = []
        for i, empresa in enumerate(plan['empresas']):
            with st.status(f"Analisando {i+1}/{len(plan['empresas'])}: {empresa}...", expanded=True):
                single_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
                context, sources_list = execute_dynamic_plan(single_plan, artifacts, model, kb)
                for src_dict in sources_list:
                    src_tuple = (src_dict['company'], src_dict['url'])
                    if src_tuple not in seen_sources_tuples:
                        seen_sources_tuples.add(src_tuple)
                        all_sources_structured.append(src_dict)
                if not context:
                    summaries.append(f"## Análise para {empresa.upper()}\n\nNenhuma informação encontrada.")
                else:
                    summary_prompt = f"Com base no contexto a seguir sobre a empresa {empresa}, resuma os pontos principais sobre os tópicos: {', '.join(plan['topicos'])}.\n\nContexto:\n{context}"
                    summaries.append(f"## Análise para {empresa.upper()}\n\n{get_final_unified_answer(summary_prompt, context)}")
        
        with st.status("Gerando relatório comparativo final...", expanded=True) as status:
            comparison_prompt = f"Com base nos resumos individuais a seguir, crie um relatório comparativo detalhado e bem estruturado com ajuda de tabela sobre '{query}'.\n\n" + "\n\n---\n\n".join(summaries)
            final_answer = get_final_unified_answer(comparison_prompt, "\n\n".join(summaries))
            status.update(label="✅ Relatório comparativo gerado!", state="complete")
    else:
        with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
            context, all_sources_structured = execute_dynamic_plan(plan, artifacts, model, kb)
            if not context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", []
            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")
            status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
        
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, all_sources_structured

def main():
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    st.markdown("---")

    model, artifacts, summary_data = setup_and_load_data()
    
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
    
    with st.expander("ℹ️ **Sobre este Agente: Capacidades e Limitações**"):
        # ... (conteúdo do expander permanece o mesmo) ...
        st.markdown("""
        Este agente foi projetado para atuar como um consultor especialista em Planos de Incentivo de Longo Prazo (ILP), analisando uma base de dados de documentos públicos da CVM. Ele possui duas capacidades principais de análise:
        """)
        st.subheader("1. Análise Quantitativa Rápida 📊")
        st.info("""
        Para perguntas que buscam **listas, contagens, médias ou estatísticas**, o agente utiliza um motor de análise de fatos pré-extraídos para fornecer respostas quase instantâneas.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Qual o desconto médio no preço de exercício?
- Quais empresas possuem TSR Relativo?
- Liste as empresas que oferecem desconto no strike.
- Quantas empresas mencionam planos de matching?""")
        st.subheader("2. Análise Qualitativa Profunda 🧠")
        st.info("""
        Para perguntas abertas que buscam **detalhes, explicações ou comparações**, o agente utiliza um pipeline de Recuperação Aumentada por Geração (RAG). Ele lê os trechos mais relevantes dos documentos para construir uma resposta detalhada.
        """)
        st.markdown("**Exemplos:**")
        st.code("""- Como funciona o plano de vesting da Vale?
- Detalhe o tratamento de dividendos no plano da Magazine Luiza.
- Compare os planos de ações restritas da Hypera e da Movida.""")
        st.subheader("❗ Limitações Importantes")
        st.warning("""
        * **Conhecimento Estático:** O agente **NÃO** tem acesso à internet.
        * **Não Emite Opinião:** Ele **encontra e apresenta** informações, mas não fornece conselhos.
        * **Dependência da Extração:** As análises quantitativas dependem da extração prévia. Se um fato não foi extraído, ele não aparecerá.
        """)

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Qual o período de vesting médio e a moda dos planos de ações restritas?")
    
    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            st.stop()
        
        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
        
        # <<< MELHORIA 1 APLICADA >>>
        # Roteador de Intenção baseado em LLM para maior precisão
        with st.spinner("Analisando a intenção da sua pergunta..."):
            intent = get_query_intent_with_llm(user_query)

        if intent == "quantitativa":
            with st.spinner("Executando análise quantitativa rápida..."):
                report_text, data_result = engine.answer_query(user_query)
                
                if report_text:
                    st.markdown(report_text)
                
                if data_result is not None:
                    if isinstance(data_result, pd.DataFrame):
                        if not data_result.empty:
                            st.dataframe(data_result, use_container_width=True, hide_index=True)
                    elif isinstance(data_result, dict):
                        for df_name, df_content in data_result.items():
                            if df_content is not None and not df_content.empty:
                                st.markdown(f"#### {df_name}")
                                st.dataframe(df_content, use_container_width=True, hide_index=True)
                else: 
                    st.info("Nenhuma análise tabular foi gerada para a sua pergunta ou dados insuficientes.")
        else: # A intenção é 'qualitativa' ou houve um erro (fallback seguro)
            final_answer, sources = handle_rag_query(
                user_query,
                artifacts,
                model,
                DICIONARIO_UNIFICADO_HIERARQUICO,
                st.session_state.company_catalog_rich,
                summary_data
            )
            st.markdown(final_answer)
            
            if sources:
                with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=True):
                    st.caption("Nota: Links diretos para a CVM podem falhar. Use a busca no portal com o protocolo como plano B.")
                    for src in sorted(sources, key=lambda x: x['company']):
                        display_text = f"{src['company']} - {src['doc_type'].replace('_', ' ')}"
                        url = src['url']
                        if "frmExibirArquivoIPEExterno" in url:
                            protocolo_match = re.search(r'NumeroProtocoloEntrega=(\d+)', url)
                            protocolo = protocolo_match.group(1) if protocolo_match else "N/A"
                            st.markdown(f"**{display_text}** (Protocolo: **{protocolo}**)")
                            st.markdown(f"↳ [Link Direto (Pode falhar)]({url}) | [Buscar na CVM]({CVM_SEARCH_URL})", unsafe_allow_html=True)
                        elif "frmExibirArquivoFRE" in url:
                            st.markdown(f"**{display_text}**")
                            st.markdown(f"↳ [Link Direto para Formulário de Referência]({url})", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{display_text}**: [Link]({url})")

if __name__ == "__main__":
    main()
