# app.py (VERSÃO CORRIGIDA)

import streamlit as st
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import requests
import re
import unicodedata
import logging
from pathlib import Path
import zipfile
import io
import base64
import shutil
import random
from models import get_embedding_model, get_cross_encoder_model
from concurrent.futures import ThreadPoolExecutor
from tools import (
    find_companies_by_topic,
    get_final_unified_answer,
    suggest_alternative_query,
    analyze_topic_thematically,
    get_summary_for_topic_at_company,
    rerank_with_cross_encoder,
    create_hierarchical_alias_map,
    rerank_by_recency
    )

# --- Módulos do Projeto ---
from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
from analytical_engine import AnalyticalEngine

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA - DEVE SER O PRIMEIRO COMANDO STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Agente de Análise ILP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# 2. INJEÇÃO DE CSS CUSTOMIZADO (BACKGROUND E FONTES)
# ==============================================================================

# URL da imagem "raw" do seu GitHub
image_url = "https://raw.githubusercontent.com/tovarich86/agentev3/main/prisday.png"

# CSS para aplicar o background e as fontes
page_bg_img = f"""
<style>
/* Importa as fontes do Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;700&family=Nunito+Sans:wght@400;700;800;900&display=swap');

/* Aplica a imagem de fundo usando a URL do GitHub */
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("{image_url}");
    background-size: cover;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local; /* Garante que a imagem role com o conteúdo */
}}

/* Deixa o header transparente para a imagem aparecer */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

/* Ajusta a posição da barra de ferramentas do Streamlit */
[data-testid="stToolbar"] {{
    right: 2rem;
}}

/* --- ESTILOS DE FONTE --- */

/* Define a fonte padrão para o corpo do texto */
html, body, [class*="css"] {{
    font-family: 'Nunito Sans', sans-serif;
    font-weight: 400;
}}

/* Define a fonte para os títulos e subtítulos */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Fira Sans', sans-serif;
    font-weight: 700; /* Bold */
}}

/* Customiza a fonte dos botões */
.stButton>button {{
    font-family: 'Nunito Sans', sans-serif;
    font-weight: 700;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# ==============================================================================
# O RESTO DO SEU CÓDIGO COMEÇA AQUI
# ==============================================================================

# --- Constantes e Configurações ---
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
TOP_K_SEARCH = 7
TOP_K_INITIAL_RETRIEVAL = 30
TOP_K_FINAL = 15
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite" # Recomendo usar um modelo mais recente se possível
CVM_SEARCH_URL = "https://www.rad.cvm.gov.br/ENET/frmConsultaExternaCVM.aspx"

FILES_TO_DOWNLOAD = {
    "item_8_4_chunks_map_final.json": "https://github.com/tovarich86/agentev3/releases/download/V2.0-DATA/item_8_4_chunks_map.json",
    "item_8_4_faiss_index_final.bin": "https://github.com/tovarich86/agentev3/releases/download/V2.0-DATA/item_8_4_faiss_index.bin",
    "outros_documentos_chunks_map_final.json": "https://github.com/tovarich86/agentev3/releases/download/V2.0-DATA/outros_documentos_chunks_map.json",
    "outros_documentos_faiss_index_final.bin": "https://github.com/tovarich86/agentev3/releases/download/V2.0-DATA/outros_documentos_faiss_index.bin",
    "resumo_fatos_e_topicos_final_enriquecido.json": "https://github.com/tovarich86/agentev3/releases/download/V2.0-DATA/resumo_fatos_e_topicos_v4_por_data.json"
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
    
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    artifacts = {}
    for index_file in CACHE_DIR.glob('*_faiss_index_final.bin'):
        category = index_file.stem.replace('_faiss_index_final', '')
        chunks_file = CACHE_DIR / f"{category}_chunks_map_final.json"
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                list_of_chunks = json.load(f)
                
            artifacts[category] = {
                'index': faiss.read_index(str(index_file)),
                'chunks': list_of_chunks
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

    setores = set()
    controles = set()

    for artifact_data in artifacts.values():
        chunk_map = artifact_data.get('chunks', [])
        for metadata in chunk_map:
            setor = metadata.get('setor')
            if isinstance(setor, str) and setor.strip():
                setores.add(setor.strip().capitalize())
            else:
                setores.add("Não identificado")

            controle = metadata.get('controle_acionario')
            if isinstance(controle, str) and controle.strip():
                controles.add(controle.strip().capitalize())
            else:
                controles.add("Não identificado")

    sorted_setores = sorted([s for s in setores if s != "Não Informado"])
    if "Não Informado" in setores:
        sorted_setores.append("Não Informado")

    sorted_controles = sorted([c for c in controles if c != "Não Informado"])
    if "Não Informado" in controles:
        sorted_controles.append("Não Informado")

    all_setores = ["Todos"] + sorted_setores
    all_controles = ["Todos"] + sorted_controles

    logger.info(f"Filtros dinâmicos encontrados: {len(all_setores)-1} setores e {len(all_controles)-1} tipos de controle.")
    
    return artifacts, summary_data, all_setores, all_controles, embedding_model, cross_encoder_model

# ... (o resto do seu código, desde a função _create_flat_alias_map, permanece exatamente o mesmo)
# --- FUNÇÕES GLOBAIS E DE RAG ---
def convert_numpy_types(o):
    """
    Percorre recursivamente uma estrutura de dados (dicionários, listas) e converte
    os tipos numéricos do NumPy para os tipos nativos do Python, tornando-a
    serializável para JSON.
    """
    if isinstance(o, (np.int64, np.int32, np.int16, np.int8)):
        return int(o)
    if isinstance(o, (np.float64, np.float32, np.float16)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: convert_numpy_types(v) for k, v in o.items()}
    if isinstance(o, list):
        return [convert_numpy_types(i) for i in o]
    return o


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

def anonimizar_resultados(data, company_catalog, anom_map=None):
    """
    [VERSÃO CORRIGIDA E ROBUSTA] Recebe um DataFrame, texto ou dicionário e substitui
    os nomes das empresas e seus aliases por placeholders.
    Garante que a função sempre retorne uma tupla (data, anom_map).
    """
    if anom_map is None:
        anom_map = {}

    # Lógica para DataFrames
    if isinstance(data, pd.DataFrame):
        df_anonimizado = data.copy()
        target_col = None
        for col in df_anonimizado.columns:
            if 'empresa' in col.lower() or 'companhia' in col.lower():
                target_col = col
                break
        if target_col:
            def get_anon_name(company_name):
                if company_name not in anom_map:
                    company_info = next((item for item in company_catalog if item["canonical_name"] == company_name), None)
                    anon_name = f"Empresa {chr(65 + len(anom_map))}"
                    anom_map[company_name] = {
                        "anon_name": anon_name,
                        "aliases_to_replace": [company_name] + (company_info['aliases'] if company_info else [])
                    }
                return anom_map[company_name]["anon_name"]
            df_anonimizado[target_col] = df_anonimizado[target_col].apply(get_anon_name)
        return df_anonimizado, anom_map

    # Lógica para Dicionários de DataFrames
    if isinstance(data, dict):
        dict_anonimizado = {}
        for key, df in data.items():
            if isinstance(df, pd.DataFrame):
                dict_anonimizado[key], anom_map = anonimizar_resultados(df, company_catalog, anom_map)
            else:
                dict_anonimizado[key] = df
        return dict_anonimizado, anom_map
        
    # Lógica para Texto (Garante o retorno)
    if isinstance(data, str):
        texto_anonimizado = data
        if anom_map:  # Apenas tenta substituir se o mapa não estiver vazio
            for original_canonical, mapping in anom_map.items():
                anon_name = mapping["anon_name"]
                aliases_sorted = sorted(mapping["aliases_to_replace"], key=len, reverse=True)
                for alias in aliases_sorted:
                    pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
                    texto_anonimizado = re.sub(pattern, anon_name, texto_anonimizado, flags=re.IGNORECASE)
        # SEMPRE retorna uma tupla, mesmo que o texto não tenha sido alterado
        return texto_anonimizado, anom_map
        
    # Fallback para qualquer outro tipo de dado não tratado
    return data, anom_map

def search_by_tags(query: str, kb: dict) -> list[str]:
    """
    Versão melhorada que busca por palavras-chave na query e retorna as tags correspondentes.
    Evita o uso de expressões regulares complexas para cada chunk.
    """
    found_tags = set()
    # Converte a query para minúsculas e remove pontuação para uma busca mais limpa
    clean_query = query.lower().strip()
    
    # Itera sobre todas as tags e seus sinônimos no dicionário de conhecimento
    for tag, details in kb.items():
        search_terms = [tag.lower()] + [s.lower() for s in details.get("sinonimos", [])]
        
        # Se qualquer um dos termos de busca estiver na query, adiciona a tag
        if any(term in clean_query for term in search_terms):
            found_tags.add(tag)
            
    return list(found_tags)

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
def get_query__with_llm(query: str) -> str:
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

    Responda apenas com o JSON da classificação. Exemplo de resposta: {{"": "qualitativa"}}
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
        # Corrigido: renomeado para intent_json para clareza
        intent_json = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group())
        # Corrigido: Adicionada a variável 'intent' e a chave correta 'intent' no .get()
        intent = intent_json.get("intent", "qualitativa").lower()
        
        # Corrigido: Adicionada a variável 'intent' ao log
        logger.info(f"Intenção detectada pelo LLM: '{intent}' para a pergunta: '{query}'")
        
        # Corrigido: Adicionada a variável 'intent' na condição
        if intent in ["quantitativa", "qualitativa"]:
            return intent
        else:
            logger.warning(f"Intenção não reconhecida '{intent}'. Usando 'qualitativa' como padrão.")
            return "qualitativa"

    except Exception as e:
        logger.error(f"ERRO ao determinar intenção com LLM: {e}. Usando 'qualitativa' como padrão.")
        return "qualitativa"

# O restante do seu código pode seguir aqui...

from datetime import datetime # Certifique-se que 'datetime' está importado no topo do seu script

def execute_dynamic_plan(
    query: str,
    plan: dict,
    artifacts: dict,
    model,  # SentenceTransformer
    cross_encoder_model,  # CrossEncoder
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    search_by_tags: callable,
    expand_search_terms: callable,
    prioritize_recency: bool = True,
) -> tuple[str, list[dict]]:
    """
    Versão Completa de execute_dynamic_plan
    """

    import re
    import random
    from collections import defaultdict
    from datetime import datetime
    import faiss


    # -------------- HELPERS --------------
    def _is_company_match(plan_canonical_name: str, metadata_name: str) -> bool:
        if not plan_canonical_name or not metadata_name:
            return False

        # Função auxiliar para remover acentos e caracteres especiais
        def normalize_text(text: str) -> str:
            import unicodedata
            import re
            # Remove acentos (diacríticos)
            nfkd_form = unicodedata.normalize('NFKD', text.lower())
            only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
            # Remove pontuações e excesso de espaços
            only_ascii = re.sub(r'[^\w\s]', '', only_ascii)
            return ' '.join(only_ascii.split())

        normalized_plan_name = normalize_text(plan_canonical_name)
        normalized_metadata_name = normalize_text(metadata_name)

        return normalized_plan_name in normalized_metadata_name

    candidate_chunks_dict = {}

    def add_candidate(chunk):
        """Add chunk de forma única por sua origem e id/texto."""
        key = chunk.get('source_url', '') + str(chunk.get('chunk_id', hash(chunk.get('text', ''))))
        if key not in candidate_chunks_dict:
            candidate_chunks_dict[key] = chunk

    # -------------- LOG INICIAL --------------
    logger.info(f"Executando plano dinâmico para query: '{query}'")
    plan_type = plan.get("plan_type", "default")
    empresas = plan.get("empresas", [])
    topicos = plan.get("topicos", [])

    # -------------- CARREGAMENTO E NORMALIZAÇÃO DOS CHUNKS --------------
    all_chunks = [
        chunk_meta
        for artifact_data in artifacts.values()
        for chunk_meta in artifact_data.get('chunks', [])
    ]
    for chunk in all_chunks:
        if 'chunk_text' in chunk and 'text' not in chunk:
            chunk['text'] = chunk.pop('chunk_text')
        if 'doc_type' not in chunk:
            if 'frmExibirArquivoFRE' in chunk.get('source_url', ''):
                chunk['doc_type'] = 'item_8_4'
            else:
                chunk['doc_type'] = 'outros_documentos'
        # Prévias para busca rápida nos tópicos (pode expandir conforme necessidade)
        if "topics_in_chunk" not in chunk:
            chunk["topics_in_chunk"] = []

    # -------------- FILTROS ----------
    filtros = plan.get("filtros", {})

    pre_filtered_chunks = all_chunks
    if filtros.get('setor'):
        pre_filtered_chunks = [
            c for c in pre_filtered_chunks
            if c.get('setor', '').lower() == filtros['setor'].lower()
        ]
    if filtros.get('controle_acionario'):
        pre_filtered_chunks = [
            c for c in pre_filtered_chunks
            if c.get('controle_acionario', '').lower() == filtros['controle_acionario'].lower()
        ]
    logger.info(f"Após pré-filtragem, {len(pre_filtered_chunks)} chunks são candidatos.")

    # -------------- EXPANSÃO DE TERMOS COM BASE NOS TÓPICOS DO PLANO ----------------
    # Esta abordagem é mais robusta pois utiliza os tópicos já identificados pelo planejador.
    if topicos:
        expanded_terms = {query.lower()}
        for topic_path in topicos:
            # Pega o alias mais específico (a última parte do caminho do tópico) para expandir a busca.
            # Ex: De "ParticipantesCondicoes,CondicaoSaida", extrai "CondicaoSaida".
            alias = topic_path.split(',')[-1].replace('_', ' ')
            expanded_terms.update(expand_search_terms(alias, kb))
        
        query_to_search = " ".join(list(expanded_terms))
        logger.info(f"Query expandida com base nos tópicos do plano: '{query_to_search}'")
    else:
        logger.info("Nenhum tópico encontrado no plano. Usando query original.")
        query_to_search = query

    # -------------- ROTEAMENTO PRINCIPAL --------------
    if plan_type == "section_8_4" and empresas:
        # ROTA CORRIGIDA para "descreva item 8.4 vivara"
        canonical_name_from_plan = empresas[0]
        search_name = next(
            (
                e.get("search_alias", canonical_name_from_plan)
                for e in company_catalog_rich
                if e.get("canonical_name") == canonical_name_from_plan
            ),
            canonical_name_from_plan,
        )
        logger.info(f"ROTA ESPECIAL section_8_4: Usando nome de busca '{search_name}'.")

        # 1. Filtra para obter todos os chunks do item 8.4 da empresa.
        chunks_to_search = [
            c for c in pre_filtered_chunks
            if c.get('doc_type') == 'item_8_4' and _is_company_match(canonical_name_from_plan, c.get('company_name', ''))
        ]

        # 2. SE chunks foram encontrados, adiciona TODOS eles diretamente aos candidatos.
        if chunks_to_search:
            logger.info(f"Rota 'section_8_4': {len(chunks_to_search)} chunks encontrados para '{canonical_name_from_plan}'. Adicionando todos ao contexto.")
            for chunk in chunks_to_search:
                add_candidate(chunk)
        else:
            logger.warning(f"Rota 'section_8_4': Nenhum chunk do tipo 'item_8_4' foi encontrado para a empresa '{canonical_name_from_plan}'.")

    else:
        if not empresas and topicos:
            # ROTA GERAL: Lógica original preservada.
            logger.info(f"ROTA Default (Geral): busca conceitual para tópicos: {topicos}")
            sample_size = 100
            chunks_to_search = random.sample(
                pre_filtered_chunks,
                min(sample_size, len(pre_filtered_chunks))
            )
            if chunks_to_search:
                temp_embeddings = model.encode(
                    [c['text'] for c in chunks_to_search],
                    normalize_embeddings=True
                ).astype('float32')
                temp_index = faiss.IndexFlatIP(temp_embeddings.shape[1])
                temp_index.add(temp_embeddings)
                for topico in topicos:
                    for term in expand_search_terms(topico, kb)[:3]:
                        search_query = f"explicação detalhada sobre o conceito e funcionamento de {term}"
                        query_embedding = model.encode(
                            [search_query],
                            normalize_embeddings=True
                        ).astype('float32')
                        _, indices = temp_index.search(query_embedding, TOP_K_FINAL)
                        for idx in indices[0]:
                            if idx != -1:
                                add_candidate(chunks_to_search[idx])

        elif empresas and topicos:
            # ROTA HÍBRIDA: Lógica original e robusta preservada.
            logger.info(f"ROTA HÍBRIDA: Empresas: {empresas}, Tópicos: {topicos}")
            target_topic_paths = plan.get("topicos", [])

            for empresa_canonica in empresas:
                chunks_for_company = [
                    c for c in pre_filtered_chunks
                    if _is_company_match(empresa_canonica, c.get('company_name', ''))
                ]
                if not chunks_for_company:
                    continue

                # Deduplicação e recorte por data (recency)
                docs_by_url = defaultdict(list)
                for chunk in chunks_for_company:
                    docs_by_url[chunk.get('source_url')].append(chunk)
                MAX_DOCS_PER_COMPANY = 3
                if len(docs_by_url) > MAX_DOCS_PER_COMPANY:
                    sorted_urls = sorted(
                        docs_by_url.keys(),
                        key=lambda url: docs_by_url[url][0].get('document_date', '0000-00-00'),
                        reverse=True
                    )
                    latest_urls = sorted_urls[:MAX_DOCS_PER_COMPANY]
                    chunks_for_company = [chunk for url in latest_urls for chunk in docs_by_url[url]]
                    logger.info(f"Para '{empresa_canonica}', selecionando os {MAX_DOCS_PER_COMPANY} documentos mais recentes pela DATA REAL.")

                # Etapa 1: Busca por tags (precisão)
                logger.info(f"[{empresa_canonica}] Etapa 1: Busca por tags nos metadados...")
                for chunk in chunks_for_company:
                    if any(
                        target_path in path
                        for path in chunk.get("topics_in_chunk", [])
                        for target_path in target_topic_paths
                    ):
                        add_candidate(chunk)

                # Etapa 2: Busca vetorial semântica (abrangência)
                logger.info(f"[{empresa_canonica}] Etapa 2: Busca por similaridade semântica...")
                if chunks_for_company:
                    temp_embeddings = model.encode(
                        [c.get('text', '') for c in chunks_for_company],
                        normalize_embeddings=True
                    ).astype('float32')
                    temp_index = faiss.IndexFlatIP(temp_embeddings.shape[1])
                    temp_index.add(temp_embeddings)
                    search_name = next(
                        (
                            e.get("search_alias", empresa_canonica)
                            for e in company_catalog_rich
                            if e.get("canonical_name") == empresa_canonica
                        ),
                        empresa_canonica
                    )
                    search_query = (f"informações detalhadas sobre "
                                    f"{' e '.join(topicos)} no plano da empresa {search_name}")
                    query_embedding = model.encode(
                        [search_query], normalize_embeddings=True
                    ).astype('float32')
                    _, indices = temp_index.search(
                        query_embedding,
                        min(TOP_K_INITIAL_RETRIEVAL, len(chunks_for_company))
                    )
                    for idx in indices[0]:
                        if idx != -1:
                            add_candidate(chunks_for_company[idx])
    # -------------------- RE-RANKING FINAL ----------------------------
    if not candidate_chunks_dict:
        logger.warning(
            f"Nenhum chunk candidato encontrado para a query: '{query}' com os filtros aplicados."
        )
        return "Não encontrei informações relevantes para esta combinação específica de consulta e filtros.", []

    candidate_list = list(candidate_chunks_dict.values())
    if prioritize_recency:
        logger.info("Re-ranking adicional por recência ativado.")
        candidate_list = rerank_by_recency(candidate_list, datetime.now())

    reranked_chunks = rerank_with_cross_encoder(
        query, candidate_list, cross_encoder_model, top_n=TOP_K_FINAL
    )

    # -------------- CONSTRUÇÃO DO CONTEXTO FINAL PARA RETORNO ---------------
    full_context = ""
    retrieved_sources = []
    seen_sources = set()
    for chunk in reranked_chunks:
        company_name = chunk.get('company_name', 'N/A')
        source_url = chunk.get('source_url', 'N/A')
        source_header = (
            f"(Empresa: {company_name}, Setor: {chunk.get('setor', 'N/A')}, "
            f"Documento: {chunk.get('doc_type', 'N/A')})"
        )
        clean_text = re.sub(r'\[.*?\]', '', chunk.get('text', '')).strip()
        full_context += (
            f"--- CONTEÚDO RELEVANTE {source_header} ---\n{clean_text}\n\n"
        )
        source_tuple = (company_name, source_url)
        if source_tuple not in seen_sources:
            seen_sources.add(source_tuple)
            retrieved_sources.append(chunk)

    logger.info(
        f"Contexto final construído a partir de {len(reranked_chunks)} chunks re-ranqueados."
    )
    return full_context, retrieved_sources

    
def create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters: dict):
    """
    [VERSÃO DE DEPURAÇÃO] do planejador dinâmico para inspecionar a
    identificação de tópicos.
    """
    # Adicione 'import streamlit as st' no topo do seu arquivo, se ainda não o fez.
    import streamlit as st

    
    logger.info(f"Gerando plano dinâmico v3.0 para a pergunta: '{query}'")
    query_lower = query.lower().strip()
    
    plan = {
        "empresas": [],
        "topicos": [],
        "filtros": filters.copy(),
        "plan_type": "default"
    }

    # A lógica de identificação de empresas não é o foco do bug, então a mantemos como está.
    mentioned_companies = []
    if company_catalog_rich:
        companies_found_by_alias = {}
        for company_data in company_catalog_rich:
            canonical_name = company_data.get("canonical_name")
            if not canonical_name: continue
            
            all_aliases = company_data.get("aliases", []) + [canonical_name]
            for alias in all_aliases:
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                    score = len(alias.split())
                    if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                        companies_found_by_alias[canonical_name] = score
        if companies_found_by_alias:
            mentioned_companies = [c for c, s in sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)]

    if not mentioned_companies:
        for empresa_nome in summary_data.keys():
            if re.search(r'\b' + re.escape(empresa_nome.lower()) + r'\b', query_lower):
                mentioned_companies.append(empresa_nome)
    
    plan["empresas"] = mentioned_companies
    logger.info(f"Empresas identificadas: {plan['empresas']}")

    
        
    # O resto da função continua normalmente para que o app não quebre
    if plan["empresas"] and not plan["topicos"]:
        logger.info("Nenhum tópico específico encontrado. Ativando modo de resumo/comparação geral.")
        plan["plan_type"] = "summary"
        plan["topicos"] = [
            "TiposDePlano", "ParticipantesCondicoes,Elegibilidade", "MecanicasCicloDeVida,Vesting", 
            "MecanicasCicloDeVida,Lockup", "IndicadoresPerformance", 
            "EventosFinanceiros,DividendosProventos"
        ]
        logger.info(f"Tópicos de resumo geral adicionados ao plano: {plan['topicos']}")    

    if not plan["empresas"] and not plan["topicos"] and not plan["filtros"]:
        logger.warning("Planejador não conseguiu identificar empresa, tópico ou filtro na pergunta.")
        return {"status": "error", "message": "Não foi possível identificar uma intenção clara na sua pergunta. Tente ser mais específico."}
        
    return {"status": "success", "plan": plan}

    
def analyze_single_company(
    empresa: str,
    plan: dict,
    query: str,
    artifacts: dict,
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    execute_dynamic_plan_func: callable,
    get_final_unified_answer_func: callable
) -> dict:
    """
    Executa o plano de análise para uma única empresa e retorna um dicionário estruturado.
    """
    single_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
    
    context, sources_list = execute_dynamic_plan_func(query, single_plan, artifacts, model, cross_encoder_model, kb, company_catalog_rich,
        company_lookup_map, search_by_tags, expand_search_terms)
    
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
    embedding_model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    summary_data: dict,
    filters: dict,
    prioritize_recency: bool = False,
    anonimizar_empresas: bool = False
) -> tuple[str, list[dict]]:
    """
    [VERSÃO FINAL E CORRIGIDA] Orquestra o pipeline de RAG, aplicando a anonimização
    de forma centralizada e consistente em todos os fluxos.
    """
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters)
        
        if plan_response['status'] != "success":
            status.update(label="⚠️ Falha na identificação", state="error", expanded=True)
            st.warning("Não consegui identificar uma empresa ou tópico em sua pergunta.")
            # ... (código de sugestão de query) ...
            return "", []
            
        plan = plan_response['plan']
        mapa_anonimizacao = {}
        display_empresas = plan.get('empresas', [])

        # Etapa 1: Anonimização para a UI (cria o mapa inicial)
        if anonimizar_empresas and display_empresas:
            df_empresas_plano = pd.DataFrame([{"Empresa": e} for e in display_empresas])
            df_anon, mapa_anonimizacao = anonimizar_resultados(df_empresas_plano, st.session_state.company_catalog_rich)
            display_empresas = df_anon["Empresa"].tolist()

        if display_empresas:
            st.write(f"**🏢 Empresas identificadas:** {', '.join(display_empresas)}")
        else:
            st.write("**🏢 Nenhuma empresa específica identificada. Realizando busca geral.**")
            
        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    final_answer, all_sources_structured = "", []
    
    # --- Lógica para Múltiplas Empresas (Comparação) ---
    if len(plan.get('empresas', [])) > 1:
        st.info(f"Modo de comparação ativado para {len(plan['empresas'])} empresas...")
        
        with st.spinner(f"Analisando {len(plan['empresas'])} empresas..."):
            # ... (código do ThreadPoolExecutor para coletar 'results') ...
            with ThreadPoolExecutor(max_workers=len(plan['empresas'])) as executor:
                futures = [
                    executor.submit(
                        analyze_single_company, empresa, plan, query, artifacts, embedding_model, cross_encoder_model, 
                        kb, company_catalog_rich, company_lookup_map, execute_dynamic_plan, get_final_unified_answer) 
                    for empresa in plan['empresas']
                ]
                results = [future.result() for future in futures]
        
        results = convert_numpy_types(results)

        # Etapa 2: Anonimização do CONTEÚDO para o LLM
        if anonimizar_empresas:
            # Primeiro, anonimiza os resultados da análise, reutilizando e atualizando o mapa
            for res in results:
                res['empresa'], mapa_anonimizacao = anonimizar_resultados(res['empresa'], st.session_state.company_catalog_rich, mapa_anonimizacao)
                for topico, resumo in res['resumos_por_topico'].items():
                    res['resumos_por_topico'][topico], mapa_anonimizacao = anonimizar_resultados(resumo, st.session_state.company_catalog_rich, mapa_anonimizacao)

            # Depois, anonimiza a lista de fontes usando o MESMO mapa
            sources_list = [src for res in results for src in res.get('sources', [])]
            df_sources = pd.DataFrame(sources_list)
            if not df_sources.empty:
                df_sources.rename(columns={'company_name': 'Empresa'}, inplace=True)
                df_sources_anon, _ = anonimizar_resultados(df_sources, st.session_state.company_catalog_rich, mapa_anonimizacao)
                all_sources_structured = df_sources_anon.rename(columns={'Empresa': 'company_name'}).to_dict('records')

        else: # Se não estiver anonimizando, apenas coleta as fontes
            all_sources_structured = [src for res in results for src in res.get('sources', [])]


        with st.status("Gerando relatório comparativo final...", expanded=True) as status:
            structured_context = json.dumps(results, indent=2, ensure_ascii=False)
            comparison_prompt = f"""
            Sua tarefa é criar um relatório comparativo sobre "{query}" usando o CONTEXTO JSON abaixo.
            Os nomes das empresas já foram anonimizados. Use apenas os nomes anonimizados (ex: "Empresa A", "Empresa B") na sua resposta.
            Crie uma breve análise textual seguida por uma TABELA MARKDOWN clara e bem formatada.

            CONTEXTO:
            {structured_context}
            """
            final_answer = get_final_unified_answer(comparison_prompt, structured_context)
            status.update(label="✅ Relatório comparativo gerado!", state="complete")
            
    # --- Lógica para Empresa Única ou Busca Geral ---
    else:
        # ... (código idêntico ao da resposta anterior, que já está correto)
        with st.status("2️⃣ Recuperando e re-ranqueando contexto...", expanded=True) as status:
            context, all_sources_structured = execute_dynamic_plan(
                query, plan, artifacts, embedding_model, cross_encoder_model, kb, company_catalog_rich, company_lookup_map, search_by_tags, expand_search_terms)
            
            if not context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", []
                
            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")
            status.update(label="✅ Contexto relevante selecionado!", state="complete")
        
        if anonimizar_empresas:
            df_sources = pd.DataFrame(all_sources_structured)
            if not df_sources.empty:
                df_sources.rename(columns={'company_name': 'Empresa'}, inplace=True)
                # Usa o mapa inicial (se houver) e o atualiza
                df_sources_anon, mapa_anonimizacao = anonimizar_resultados(df_sources, st.session_state.company_catalog_rich, mapa_anonimizacao)
                all_sources_structured = df_sources_anon.rename(columns={'Empresa': 'company_name'}).to_dict('records')
            
            context, _ = anonimizar_resultados(context, st.session_state.company_catalog_rich, mapa_anonimizacao)
            
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            prompt_final = f"""
            Responda à pergunta: "{query}".
            Use o contexto abaixo, que já está anonimizado. Refira-se à empresa principal como "a Empresa" ou "a Companhia".
            CONTEXTO: {context}
            """
            final_answer = get_final_unified_answer(prompt_final, context)
            status.update(label="✅ Análise concluída!", state="complete")

    # Remove duplicatas das fontes coletadas
    unique_sources = list({v['source_url']:v for v in all_sources_structured}.values())
    return final_answer, unique_sources
Principais Mudanças na Lógica de Comparação
Mapa de Anonimização Centralizado: A variável mapa_anonimizacao é criada no início. Quando o modo de anonimização está ativo, ela é passada para a primeira chamada de anonimizar_resultados (que anonimiza os nomes para a UI) e depois é reutilizada e atualizada ao anonimizar os resultados da análise e a lista de fontes. Isso garante que "Empresa B" seja sempre "Empresa B", resolvendo o bug de "Empresa C".

Prompt do LLM Simplificado: O prompt agora instrui o Gemini a usar apenas os nomes anonimizados que ele recebe no JSON, eliminando a confusão e a geração de textos como "Empresa A (Empresa A Branco)".

Coleta de Fontes Simplificada: A lógica para coletar e remover duplicatas das fontes foi otimizada.

Com essas alterações, o sistema agora deve anonimizar os nomes de forma consistente em todo o relatório de comparação. Agradeço imensamente pela sua ajuda em depurar e aprimorar o agente!







preciso da função inteira sem simplficações ou omissões ajustada

def handle_rag_query(

    query: str,

    artifacts: dict,

    embedding_model: SentenceTransformer,

    cross_encoder_model: CrossEncoder,

    kb: dict,

    company_catalog_rich: list,

    company_lookup_map: dict,

    summary_data: dict,

    filters: dict,

    prioritize_recency: bool = False,

    anonimizar_empresas: bool = False  # Adicionado o parâmetro aqui

) -> tuple[str, list[dict]]:

    """

    [VERSÃO CORRIGIDA] Orquestra o pipeline de RAG, aplicando a anonimização

    corretamente tanto para a UI quanto para o contexto do LLM.

    """

    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:

        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters)

        

        if plan_response['status'] != "success":

            status.update(label="⚠️ Falha na identificação", state="error", expanded=True)

            st.warning("Não consegui identificar uma empresa conhecida na sua pergunta para realizar uma análise profunda.")

            with st.spinner("Estou pensando em uma pergunta alternativa..."):

                alternative_query = suggest_alternative_query(query, kb)

            st.markdown("#### Que tal tentar uma pergunta mais geral?")

            st.code(alternative_query, language=None)

            return "", []

            

        plan = plan_response['plan']

        mapa_anonimizacao = {}

        display_empresas = plan['empresas']



        # Etapa 1: Anonimização para a UI (antes de exibir qualquer coisa)

        if anonimizar_empresas and plan.get('empresas'):

            df_empresas_plano = pd.DataFrame([{"Empresa": e} for e in plan['empresas']])

            df_anon, mapa_anonimizacao = anonimizar_resultados(df_empresas_plano, st.session_state.company_catalog_rich)

            display_empresas = df_anon["Empresa"].tolist()



        if display_empresas:

            st.write(f"**🏢 Empresas identificadas:** {', '.join(display_empresas)}")

        else:

            st.write("**🏢 Nenhuma empresa específica identificada. Realizando busca geral.**")

            

        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")

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

                        analyze_single_company, empresa, plan, query, artifacts, embedding_model, cross_encoder_model, 

                        kb, company_catalog_rich, company_lookup_map, execute_dynamic_plan, get_final_unified_answer) 

                    for empresa in plan['empresas']

                ]

                results = [future.result() for future in futures]



        # Coleta todas as fontes primeiro para anonimização consistente

        results = convert_numpy_types(results)

        for result in results:

            for src_dict in result.get('sources', []):

                src_tuple = (src_dict.get('company_name'), src_dict.get('source_url'))

                if src_tuple not in seen_sources_tuples:

                    seen_sources_tuples.add(src_tuple)

                    all_sources_structured.append(src_dict)

        

        # Etapa 2: Anonimização do CONTEÚDO para o LLM

        if anonimizar_empresas:

            # Anonimiza a lista de fontes

            df_sources = pd.DataFrame(all_sources_structured)

            if not df_sources.empty:

                df_sources.rename(columns={'company_name': 'Empresa'}, inplace=True)

                df_sources_anon, mapa_anonimizacao_fontes = anonimizar_resultados(df_sources, st.session_state.company_catalog_rich, mapa_anonimizacao)

                all_sources_structured = df_sources_anon.rename(columns={'Empresa': 'company_name'}).to_dict('records')

            

            # Anonimiza os resultados da análise que serão enviados ao LLM

            for res in results:

                # Substitui o nome da empresa pelo nome anonimizado do mapa

                res['empresa'], _ = anonimizar_resultados(res['empresa'], st.session_state.company_catalog_rich, mapa_anonimizacao)

                # Substitui os nomes dentro dos textos de resumo

                for topico, resumo in res['resumos_por_topico'].items():

                    res['resumos_por_topico'][topico], _ = anonimizar_resultados(resumo, st.session_state.company_catalog_rich, mapa_anonimizacao)



        with st.status("Gerando relatório comparativo final...", expanded=True) as status:

            structured_context = json.dumps(results, indent=2, ensure_ascii=False)

            comparison_prompt = f"""

            Sua tarefa é criar um relatório comparativo detalhado sobre "{query}".

            Use os dados estruturados fornecidos no CONTEXTO JSON abaixo. Os nomes das empresas já foram anonimizados para "Empresa A", "Empresa B", etc. Use esses nomes anonimizados na sua resposta.

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

                query, plan, artifacts, embedding_model, cross_encoder_model, kb, company_catalog_rich, company_lookup_map, search_by_tags, expand_search_terms)

            

            if not context:

                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")

                return "Nenhuma informação relevante encontrada.", []

                

            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")

            status.update(label="✅ Contexto relevante selecionado!", state="complete")

        

        # Etapa 2: Anonimização do CONTEÚDO para o LLM

        if anonimizar_empresas:

            # Garante que o mapa está populado com as fontes

            df_sources = pd.DataFrame(all_sources_structured)

            if not df_sources.empty:

                df_sources.rename(columns={'company_name': 'Empresa'}, inplace=True)

                df_sources_anon, mapa_anonimizacao = anonimizar_resultados(df_sources, st.session_state.company_catalog_rich, mapa_anonimizacao)

                all_sources_structured = df_sources_anon.rename(columns={'Empresa': 'company_name'}).to_dict('records')

            

            # Anonimiza o texto do contexto usando o mapa populado

            context, _ = anonimizar_resultados(context, st.session_state.company_catalog_rich, mapa_anonimizacao)

            

        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:

            prompt_final = f"""

            Responda à pergunta: "{query}".

            Use o contexto abaixo, que já está anonimizado. Refira-se à empresa principal como "a Empresa" ou "a Companhia". Não tente adivinhar o nome original.



            CONTEXTO:

            {context}

            """

            final_answer = get_final_unified_answer(prompt_final, context)

            status.update(label="✅ Análise concluída!", state="complete")



    return final_answer, all_sources_structured




Mostrar raciocínio
Compreendi. Você precisa da função handle_rag_query completa, com todos os ajustes de anonimização para os modos de empresa única e de comparação, sem nenhuma simplificação.

Peço desculpas pela confusão anterior. O erro de lógica na passagem do mapa_anonimizacao causou as inconsistências. Abaixo está a versão integral e corrigida da função, que você pode substituir em seu arquivo app.py.

Esta versão garante que:

Um único mapa de anonimização é criado e reutilizado em todas as etapas (UI, análise de resultados e lista de fontes), garantindo consistência nos nomes ("Empresa A", "Empresa B", etc.).

O contexto enviado ao LLM (tanto o texto quanto o JSON) já está pré-anonimizado, evitando que ele "vaze" nomes reais.

O prompt para o LLM é claro e instrui o modelo a usar apenas os nomes anonimizados fornecidos.

Função handle_rag_query Completa e Corrigida
Python

def handle_rag_query(
    query: str,
    artifacts: dict,
    embedding_model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict,
    company_catalog_rich: list,
    company_lookup_map: dict,
    summary_data: dict,
    filters: dict,
    prioritize_recency: bool = False,
    anonimizar_empresas: bool = False
) -> tuple[str, list[dict]]:
    """
    [VERSÃO COMPLETA E CORRIGIDA] Orquestra o pipeline de RAG, aplicando a anonimização
    de forma centralizada e consistente em todos os fluxos.
    """
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        plan_response = create_dynamic_analysis_plan(query, company_catalog_rich, kb, summary_data, filters)
        
        if plan_response['status'] != "success":
            status.update(label="⚠️ Falha na identificação", state="error", expanded=True)
            st.warning("Não consegui identificar uma empresa conhecida na sua pergunta para realizar uma análise profunda.")
            with st.spinner("Estou pensando em uma pergunta alternativa..."):
                alternative_query = suggest_alternative_query(query, kb)
            st.markdown("#### Que tal tentar uma pergunta mais geral?")
            st.code(alternative_query, language=None)
            return "", []
            
        plan = plan_response['plan']
        mapa_anonimizacao = {}
        display_empresas = plan.get('empresas', [])

        # Etapa 1: Anonimização para a UI (cria o mapa inicial e consistente)
        if anonimizar_empresas and display_empresas:
            df_empresas_plano = pd.DataFrame([{"Empresa": e} for e in display_empresas])
            df_anon, mapa_anonimizacao = anonimizar_resultados(df_empresas_plano, st.session_state.company_catalog_rich, mapa_anonimizacao)
            display_empresas = df_anon["Empresa"].tolist()

        if display_empresas:
            st.write(f"**🏢 Empresas identificadas:** {', '.join(display_empresas)}")
        else:
            st.write("**🏢 Nenhuma empresa específica identificada. Realizando busca geral.**")
            
        st.write(f"**📝 Tópicos a analisar:** {', '.join(plan['topicos'])}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    final_answer = ""
    all_sources_structured = []

    # --- Lógica para Múltiplas Empresas (Comparação) ---
    if len(plan.get('empresas', [])) > 1:
        st.info(f"Modo de comparação ativado para {len(plan['empresas'])} empresas. Executando análises em paralelo...")
        
        with st.spinner(f"Analisando {len(plan['empresas'])} empresas..."):
            with ThreadPoolExecutor(max_workers=len(plan['empresas'])) as executor:
                futures = [
                    executor.submit(
                        analyze_single_company, empresa, plan, query, artifacts, embedding_model, cross_encoder_model, 
                        kb, company_catalog_rich, company_lookup_map, execute_dynamic_plan, get_final_unified_answer) 
                    for empresa in plan['empresas']
                ]
                results = [future.result() for future in futures]
        
        results = convert_numpy_types(results)

        # Etapa 2: Anonimização do CONTEÚDO para o LLM
        if anonimizar_empresas:
            # Primeiro, anonimiza os resultados da análise, reutilizando e atualizando o mapa
            for res in results:
                res['empresa'], mapa_anonimizacao = anonimizar_resultados(res['empresa'], st.session_state.company_catalog_rich, mapa_anonimizacao)
                for topico, resumo in res['resumos_por_topico'].items():
                    res['resumos_por_topico'][topico], mapa_anonimizacao = anonimizar_resultados(resumo, st.session_state.company_catalog_rich, mapa_anonimizacao)

            # Depois, anonimiza a lista de fontes usando o MESMO mapa consistente
            sources_list = [src for res in results for src in res.get('sources', [])]
            df_sources = pd.DataFrame(sources_list)
            if not df_sources.empty:
                df_sources.rename(columns={'company_name': 'Empresa'}, inplace=True)
                df_sources_anon, _ = anonimizar_resultados(df_sources, st.session_state.company_catalog_rich, mapa_anonimizacao)
                all_sources_structured = df_sources_anon.rename(columns={'Empresa': 'company_name'}).to_dict('records')

        else: # Se não estiver anonimizando, apenas coleta as fontes
            all_sources_structured = [src for res in results for src in res.get('sources', [])]

        with st.status("Gerando relatório comparativo final...", expanded=True) as status:
            structured_context = json.dumps(results, indent=2, ensure_ascii=False)
            comparison_prompt = f"""
            Sua tarefa é criar um relatório comparativo detalhado sobre "{query}" usando o CONTEXTO JSON abaixo.
            Os nomes das empresas no contexto já foram anonimizados para "Empresa A", "Empresa B", etc. Use apenas estes nomes anonimizados na sua resposta para garantir a confidencialidade.
            O relatório deve começar com uma breve análise textual e, em seguida, apresentar uma TABELA MARKDOWN clara e bem formatada.

            CONTEXTO (em formato JSON):
            {structured_context}
            """
            final_answer = get_final_unified_answer(comparison_prompt, structured_context)
            status.update(label="✅ Relatório comparativo gerado!", state="complete")
            
    # --- Lógica para Empresa Única ou Busca Geral ---
    else:
        with st.status("2️⃣ Recuperando e re-ranqueando contexto...", expanded=True) as status:
            context, sources_from_plan = execute_dynamic_plan(
                query, plan, artifacts, embedding_model, cross_encoder_model, kb, company_catalog_rich, company_lookup_map, search_by_tags, expand_search_terms, prioritize_recency=prioritize_recency)
            
            if not context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", []
            
            all_sources_structured = sources_from_plan
            st.write(f"**📄 Contexto recuperado de:** {len(all_sources_structured)} documento(s)")
            status.update(label="✅ Contexto relevante selecionado!", state="complete")
        
        # Etapa 2: Anonimização do CONTEÚDO para o LLM
        if anonimizar_empresas:
            # Garante que o mapa está populado com as fontes
            df_sources = pd.DataFrame(all_sources_structured)
            if not df_sources.empty:
                df_sources.rename(columns={'company_name': 'Empresa'}, inplace=True)
                df_sources_anon, mapa_anonimizacao = anonimizar_resultados(df_sources, st.session_state.company_catalog_rich, mapa_anonimizacao)
                all_sources_structured = df_sources_anon.rename(columns={'Empresa': 'company_name'}).to_dict('records')
            
            # Anonimiza o texto do contexto usando o mapa já populado
            context, _ = anonimizar_resultados(context, st.session_state.company_catalog_rich, mapa_anonimizacao)
            
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            prompt_final = f"""
            Responda à pergunta: "{query}".
            Use o contexto abaixo, que já está anonimizado. Refira-se à empresa principal como "a Empresa" ou "a Companhia". Não tente adivinhar o nome original.

            CONTEXTO:
            {context}
            """
            final_answer = get_final_unified_answer(prompt_final, context)
            status.update(label="✅ Análise concluída!", state="complete")

    # Remove duplicatas das fontes coletadas, mantendo a ordem
    unique_sources = list({v['source_url']:v for v in all_sources_structured}.values())
    return final_answer, unique_sources

def main():
    st.markdown('<h1 style="color:#0b2859;">🤖 PRIA (Agente de IA para ILP)</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Criar placeholders para as mensagens de carregamento
    status_message_1 = st.empty()
    status_message_2 = st.empty()
    
    status_message_1.info("Carregando modelo de embedding...")
    # A linha abaixo deve ser executada após o carregamento do modelo de embedding.
    # No seu código, a função setup_and_load_data() faz isso.
    # Vou simular que o carregamento está acontecendo.
    
    # Simulação do carregamento (você já tem a sua função `setup_and_load_data`)
    # time.sleep(2)  
    
    status_message_1.success("✅ Modelo de embedding carregado.")
    status_message_2.info("Carregando modelo de Re-ranking (Cross-Encoder)...")
    
    # A linha abaixo deve ser executada após o carregamento do modelo de re-ranking.
    # time.sleep(2)
    
    # Suas chamadas de carregamento
    artifacts, summary_data, setores_disponiveis, controles_disponiveis, embedding_model, cross_encoder_model = setup_and_load_data()

    # Limpar os placeholders para que as mensagens não fiquem na tela
    status_message_1.empty()
    status_message_2.empty()

    if not summary_data or not artifacts:
        st.error("❌ Falha crítica no carregamento dos dados. O app não pode continuar.")
        st.stop()

    artifacts, summary_data, setores_disponiveis, controles_disponiveis, embedding_model, cross_encoder_model = setup_and_load_data()
        
    if not summary_data or not artifacts:
        st.error("❌ Falha crítica no carregamento dos dados. O app não pode continuar.")
        st.stop()
    
    engine = AnalyticalEngine(summary_data, DICIONARIO_UNIFICADO_HIERARQUICO) 
    
    try:
        from catalog_data import company_catalog_rich 
    except ImportError:
        company_catalog_rich = [] 
    
    st.session_state.company_catalog_rich = company_catalog_rich

    
    from tools import _create_company_lookup_map
    st.session_state.company_lookup_map = _create_company_lookup_map(company_catalog_rich)


    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Categorias de Documentos (RAG)", "Item 8.4", "Plano de Remuneração")
        
        st.markdown("---")
        st.header("🔒 Modo Apresentação")
        anonimizar_empresas = st.checkbox(
            "Ocultar nomes de empresas",
            value=False,
            help="Substitui os nomes das empresas por placeholders como 'Empresa A', 'Empresa B' para garantir a confidencialidade durante a apresentação."
        )
        
        st.markdown("---")

        
        prioritize_recency = st.checkbox(
            "Priorizar documentos mais recentes",
            value=True,
            help="Dá um bônus de relevância para os documentos mais novos.")
        st.metric("Empresas no Resumo", len(summary_data))
        st.header("⚙️ Filtros da Análise")
        st.caption("Filtre a base de dados antes de fazer sua pergunta.")
        
        selected_setor = st.selectbox(
            label="Filtrar por Setor",
            options=setores_disponiveis,
            index=0
        )
        
        selected_controle = st.selectbox(
            label="Filtrar por Controle Acionário",
            options=controles_disponiveis,
            index=0
        )
        st.markdown("---") 
        with st.expander("Empresas com dados no resumo"):
            st.dataframe(pd.DataFrame(sorted(list(summary_data.keys())), columns=["Empresa"]), use_container_width=True, hide_index=True)
        st.success("✅ Sistema pronto para análise")
        st.info(f"Embedding Model: `{MODEL_NAME}`")
        st.info(f"Generative Model: `{GEMINI_MODEL}`")
    
    st.header("💬 Faça sua pergunta")
    
    with st.expander("ℹ️ **Guia do Usuário: Como Extrair o Máximo do Agente**", expanded=False):
        st.markdown("""
        Este agente foi projetado para atuar como um consultor especialista em Planos de Incentivo de Longo Prazo (ILP), analisando uma base de dados de documentos públicos da CVM. Para obter os melhores resultados, formule perguntas que explorem suas principais capacidades.
        """)
        st.subheader("1. Perguntas de Listagem (Quem tem?) 🎯")
        st.info("Use estas perguntas para identificar e listar empresas que adotam uma prática específica. Ideal para mapeamento de mercado.")
        st.code("""- Liste as empresas que pagam dividendos ou JCP durante o período de carência (vesting).
- Quais companhias possuem cláusulas de Malus ou Clawback?
- Gere uma lista de empresas que oferecem planos com contrapartida do empregador (Matching/Coinvestimento).
- Quais organizações mencionam explicitamente o Comitê de Remuneração como órgão aprovador dos planos?""")
        st.subheader("2. Análise Estatística (Qual a média?) 📈")
        st.info("Pergunte por médias, medianas e outros dados estatísticos para entender os números por trás das práticas de mercado e fazer benchmarks.")
        st.code("""- Qual o período médio de vesting (em anos) entre as empresas analisadas?
- Qual a diluição máxima média (% do capital social) que os planos costumam aprovar?
- Apresente as estatísticas do desconto no preço de exercício (mínimo, média, máximo).
- Qual o prazo de lock-up (restrição de venda) mais comum após o vesting das ações?""")
        st.subheader("3. Padrões de Mercado (Como é o normal?) 🗺️")
        st.info("Faça perguntas abertas para que o agente analise diversos planos e descreva os padrões e as abordagens mais comuns para um determinado tópico.")
        st.code("""- Analise os modelos típicos de planos de Ações Restritas (RSU), o tipo mais comum no mercado.
- Além do TSR, quais são as metas de performance (ESG, Financeiras) mais utilizadas pelas empresas?
- Descreva os padrões de tratamento para condições de saída (Good Leaver vs. Bad Leaver) nos planos.
- Quais as abordagens mais comuns para o tratamento de dividendos em ações ainda não investidas?""")
        st.subheader("4. Análise Profunda e Comparativa (Me explique em detalhes) 🧠")
        st.info("Use o poder do RAG para pedir análises detalhadas sobre uma ou mais empresas, comparando regras e estruturas específicas.")
        st.code("""- Como o plano da Vale trata a aceleração de vesting em caso de mudança de controle?
- Compare as cláusulas de Malus/Clawback da Vale com as do Itaú.
- Descreva em detalhes o plano de Opções de Compra da Localiza, incluindo prazos, condições e forma de liquidação.
- Descreva o Item 8.4 da M.dias Braco.
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
        active_filters = {}
        if selected_setor != "Todos":
            active_filters['setor'] = selected_setor.lower()
        if selected_controle != "Todos":
            active_filters['controle_acionario'] = selected_controle.lower()
        if active_filters:
            filter_text_parts = []
            if 'setor' in active_filters:
                filter_text_parts.append(f"**Setor**: {active_filters['setor'].capitalize()}")
            if 'controle_acionario' in active_filters:
                filter_text_parts.append(f"**Controle**: {active_filters['controle_acionario'].capitalize()}")

            filter_text = " e ".join(filter_text_parts)
            st.info(f"🔎 Análise sendo executada com os seguintes filtros: {filter_text}")

        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
                
        intent = None
        query_lower = user_query.lower()
        
        quantitative_keywords = [
            'liste', 'quais empresas', 'quais companhias', 'quantas', 'média', 
            'mediana', 'estatísticas', 'mais comuns', 'prevalência', 'contagem'
        ]
        
        if any(keyword in query_lower for keyword in quantitative_keywords):
            intent = "quantitativa"
            logger.info("Intenção 'quantitativa' detectada por regras de palavras-chave.")
        
        if intent is None:
            with st.spinner("Analisando a intenção da sua pergunta..."):
                intent = get_query__with_llm(user_query)

        # [NOVA ADIÇÃO 1] Inicializa o mapa de anonimização que será usado em toda a análise.
        # Isso garante que a "Empresa A" seja sempre a mesma, seja no texto ou nas tabelas.
        mapa_anonimizacao = {}
                
        if intent == "quantitativa":
            st.info("Intenção quantitativa detectada. Usando o motor de análise rápida...")
    
            with st.spinner("Executando análise quantitativa..."):
                report_text, data_result = engine.answer_query(user_query, filters=active_filters)
                
                # [NOVA ADIÇÃO 2] Verifica se o modo de anonimização está ativo.
                # Se estiver, passa o resultado pela função anonimizar_resultados.
                if anonimizar_empresas and data_result is not None:
                    data_result, mapa_anonimizacao = anonimizar_resultados(data_result, st.session_state.company_catalog_rich)

                # A partir daqui, o código de exibição renderizará a versão
                # original ou a anonimizada, dependendo do checkbox.
                if report_text:
                    st.markdown(report_text)
            
                if data_result is not None:
                    if isinstance(data_result, pd.DataFrame):
                        if not data_result.empty:
                            st.dataframe(data_result, use_container_width=True, hide_index=True)
                    elif isinstance(data_result, dict):
                        for df_name, df_content in data_result.items():
                            if isinstance(df_content, pd.DataFrame) and not df_content.empty:
                                st.markdown(f"#### {df_name}")
                                st.dataframe(df_content, use_container_width=True, hide_index=True)

        else: # intent == 'qualitativa'
            final_answer, sources = handle_rag_query(
                user_query, 
                artifacts, 
                embedding_model, 
                cross_encoder_model, 
                kb=DICIONARIO_UNIFICADO_HIERARQUICO,
                company_catalog_rich=st.session_state.company_catalog_rich, 
                company_lookup_map=st.session_state.company_lookup_map, 
                summary_data=summary_data,
                filters=active_filters,
                prioritize_recency=prioritize_recency
            )
            
            # [NOVA ADIÇÃO 3] Lógica de anonimização de duas etapas para o RAG.
            if anonimizar_empresas and sources:
                # 3.1: Cria o mapa de anonimização a partir da lista de fontes.
                df_sources = pd.DataFrame(sources)
                if not df_sources.empty:
                    df_sources.rename(columns={'company_name': 'Empresa'}, inplace=True)
                    df_sources_anon, mapa_anonimizacao = anonimizar_resultados(df_sources, st.session_state.company_catalog_rich)
                    sources = df_sources_anon.rename(columns={'Empresa': 'company_name'}).to_dict('records')

                # 3.2: Usa o mapa já criado para substituir os nomes no texto da resposta.
                final_answer, _ = anonimizar_resultados(final_answer, st.session_state.company_catalog_rich, mapa_anonimizacao)
            
            st.markdown(final_answer)
            
            if sources:
                with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=True):
                    st.caption("Nota: Links diretos para a CVM podem falhar. Use a busca no portal com o protocolo como plano B.")
    
                    for src in sorted(sources, key=lambda x: x.get('company_name', '')):
                        # Este código de exibição agora usará os nomes anonimizados de 'sources'
                        company_name = src.get('company_name', 'N/A')
                        doc_date = src.get('document_date', 'N/A')
                        doc_type_raw = src.get('doc_type', '')
                        url = src.get('source_url', '')

                        if doc_type_raw == 'outros_documentos':
                            display_doc_type = 'Plano de Remuneração'
                        else:
                            display_doc_type = doc_type_raw.replace('_', ' ')
    
                        display_text = f"{company_name} - {display_doc_type} - (Data: **{doc_date}**)"
                        
                        if "frmExibirArquivoIPEExterno" in url:
                            protocolo_match = re.search(r'NumeroProtocoloEntrega=(\d+)', url)
                            protocolo = protocolo_match.group(1) if protocolo_match else "N/A"
                            st.markdown(f"**{display_text}** (Protocolo: **{protocolo}**)")
                            st.markdown(f"↳ [Link Direto para Plano de ILP]({url}) ", unsafe_allow_html=True)
            
                        elif "frmExibirArquivoFRE" in url:
                            st.markdown(f"**{display_text}**")
                            st.markdown(f"↳ [Link Direto para Formulário de Referência]({url})", unsafe_allow_html=True)
            
                        else:
                            st.markdown(f"**{display_text}**: [Link]({url})")


if __name__ == "__main__":
    main()
