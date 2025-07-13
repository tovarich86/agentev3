# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP v5.0 - "DEFINITIVO"

Este agente representa a fusão das melhores características dos modelos anteriores:
- A arquitetura robusta e pragmática com roteador de múltiplos níveis.
- As técnicas de IA avançadas para análise profunda e comparativa.
"""

# --- 1. Importações e Configurações ---
import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
import glob
import os
import re
import logging
import pandas as pd
from pathlib import Path
import unicodedata  

# Carregamento de dependências locais com tratamento de erro
try:
    from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
    from catalog_data import company_catalog_rich
except ImportError:
    st.error("ERRO CRÍTICO: Verifique se os arquivos 'knowledge_base.py' e 'catalog_data.py' existem e estão corretos.")
    st.stop()

# --- Configurações Gerais ---
MODEL_NAME_BI_ENCODER = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
MODEL_NAME_CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
TOP_K_INITIAL_SEARCH = 20
TOP_K_RERANKED = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite"
DADOS_PATH = Path("dados")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Funções de Carregamento de Recursos em Cache ---

@st.cache_resource
def load_models():
    """Carrega os modelos de embedding (Bi-Encoder) e de re-ranking (Cross-Encoder)."""
    bi_encoder = SentenceTransformer(MODEL_NAME_BI_ENCODER)
    cross_encoder = CrossEncoder(MODEL_NAME_CROSS_ENCODER)
    return bi_encoder, cross_encoder

@st.cache_resource
def load_artifacts():
    """Carrega os índices FAISS e os arquivos de resumo/mapa, baixando se necessário."""
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
                    r = requests.get(url, stream=True, timeout=300)
                    r.raise_for_status()
                    with open(caminho_arquivo, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                except requests.exceptions.RequestException as e:
                    st.error(f"Falha ao baixar {nome_arquivo}: {e}"); st.stop()

    artifacts = {}
    index_files = glob.glob(str(DADOS_PATH / '*_faiss_index_final.bin'))
    for index_file_path in index_files:
        category = Path(index_file_path).stem.replace('_faiss_index_final', '')
        chunks_file_path = DADOS_PATH / f"{category}_chunks_map_final.json"
        try:
            index = faiss.read_index(index_file_path)
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks_data': chunk_data}
        except Exception as e:
            logger.error(f"Erro ao carregar artefato '{category}': {e}"); continue
    
    summary_data = None
    summary_file_path = DADOS_PATH / 'resumo_fatos_e_topicos_final_enriquecido.json'
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo de resumo '{summary_file_path}' não encontrado.")
    return artifacts, summary_data

st.cache_data
def criar_mapa_de_alias(knowledge_base: dict):
    """
    Cria um mapa de alias robusto a partir do dicionário hierárquico.
    Mapeia tanto os nomes dos tópicos quanto das SEÇÕES principais.
    """
    alias_to_canonical = {}
    for section, topics in knowledge_base.items():
        # Adiciona a própria seção como um tópico pesquisável
        # Ex: "indicadoresperformance" -> "IndicadoresPerformance"
        alias_to_canonical[section.lower()] = section
        
        for canonical_name, aliases in topics.items():
            # Mapeia o nome canônico do tópico para si mesmo
            alias_to_canonical[canonical_name.lower()] = canonical_name
            # Mapeia todos os sinônimos para o nome canônico
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical


# --- 3. Handlers de Análise Rápida (para o Roteador) ---

def handle_analytical_query(query: str, summary_data: dict):
    """Lida com perguntas analíticas (média, mínimo, máximo). Retorna True se teve sucesso."""
    if not summary_data: return False
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
    
    valores, unidade = [], ''
    for data in summary_data.values():
        if target_fact_key in data.get("fatos_extraidos", {}):
            fact_data = data["fatos_extraidos"][target_fact_key]
            valor = fact_data.get('valor_numerico') or fact_data.get('valor')
            if isinstance(valor, (int, float)):
                valores.append(valor)
                if not unidade and 'unidade' in fact_data: unidade = fact_data['unidade']
    if not valores: return False

    resultado, label_metrica = 0, ""
    if operation == 'avg': resultado, label_metrica = np.mean(valores), f"Média de {target_fact_key.replace('_', ' ')}"
    elif operation == 'min': resultado, label_metrica = np.min(valores), f"Mínimo de {target_fact_key.replace('_', ' ')}"
    elif operation == 'max': resultado, label_metrica = np.max(valores), f"Máximo de {target_fact_key.replace('_', ' ')}"
    
    valor_formatado = f"{resultado:.1%}" if 'desconto' in target_fact_key else f"{resultado:.1f} {unidade}".strip()
    st.metric(label=label_metrica.title(), value=valor_formatado)
    st.caption(f"Cálculo baseado em {len(valores)} empresas com dados para este fato.")
    return True

def handle_aggregate_query(query: str, summary_data: dict, alias_map: dict):
    """Lida com perguntas agregadas (quais, quantas). Retorna True se encontrou resultados."""
    if not summary_data: return False
    query_lower = query.lower()
    query_keywords = set()
    for alias, canonical in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            query_keywords.add(canonical)
    if not query_keywords: return False

    empresas_encontradas = []
    for empresa, data in summary_data.items():
        company_topics = set(data.get("topicos_identificados", []))
        if query_keywords.issubset(company_topics):
            empresas_encontradas.append(empresa)
            
    if not empresas_encontradas: return False

    st.success(f"✅ **{len(empresas_encontradas)} empresa(s)** encontrada(s) com os termos: `{', '.join(query_keywords)}`")
    df = pd.DataFrame(sorted(empresas_encontradas), columns=["Empresa"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    return True

def handle_direct_fact_query(query: str, summary_data: dict, alias_map: dict, company_catalog: list):
    """Lida com perguntas de fato direto. Retorna True se teve sucesso."""
    if not summary_data: return False
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
    if not empresa_data: return False
    
    fato_encontrado = False
    canonical_topic = alias_map.get(fato_encontrado_alias)
    for fact_key, fact_value in empresa_data.get("fatos_extraidos", {}).items():
        if canonical_topic and canonical_topic.lower().replace(' ', '_') in fact_key.lower():
            valor, unidade = fact_value.get('valor', ''), fact_value.get('unidade', '')
            st.metric(label=f"Fato para {empresa_encontrada}: {fact_key.replace('_', ' ').title()}", value=f"{valor} {unidade}".strip())
            fato_encontrado = True; break
    if not fato_encontrado: return False
    return True


# --- 4. Motor RAG v5.0: Funções do Pipeline Avançado ---

def detect_query_intent(query: str) -> str:
    """
    Analisa a pergunta do usuário para detectar uma intenção específica.
    Retorna a categoria do artefato a ser priorizado.
    """
    query_lower = query.lower()
    # Se a pergunta menciona explicitamente o item 8.4 ou o formulário de referência
    if '8.4' in query_lower or 'formulário de referência' in query_lower:
        return 'item_8_4'
    
    # Adicione outras intenções aqui no futuro, se necessário
    # Ex: if 'proposta da administração' in query_lower: return 'proposta_remuneracao'
    
    # Se nenhuma intenção específica for encontrada, retorna 'geral'
    return 'general'

def normalize_company_name(name: str) -> str:
    """
    Normaliza nomes de empresas para uma comparação robusta, removendo acentos,
    pontuação, sufixos comuns e convertendo para minúsculas.
    """
    if not isinstance(name, str): return ""
    # Converte para minúsculas e remove acentos
    name = ''.join(c for c in unicodedata.normalize('NFD', name.lower()) if unicodedata.category(c) != 'Mn')
    # Remove pontuações e caracteres especiais
    name = re.sub(r'[.,-]', '', name)
    # Remove sufixos comuns de empresas
    suffixes = [r'\s+s\s*a\s*$', r'\s+ltda\s*$', r'\s+holding\s*$', r'\s+participacoes\s*$', r'\s+cia\s*$', r'\s+ind\s*$', r'\s+com\s*$']
    for suffix in suffixes:
        name = re.sub(suffix, '', name)
    # Remove espaços extras no início e no fim
    return name.strip()

def call_gemini_api(prompt, max_tokens=8192):
    """Função auxiliar centralizada para chamadas à API Gemini."""
    if not GEMINI_API_KEY:
        logger.error("Chave da API Gemini não configurada.")
        return "ERRO: Chave da API Gemini não configurada."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens}
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        logger.error(f"Erro na chamada da API Gemini: {e}")
        return f"ERRO na comunicação com a API: {e}"

def decompose_query_with_llm(user_query: str) -> list[str]:
    # ... (implementação completa da Otimização 1) ...
    pass

def retrieve_hybrid_and_reranked_context(query: str, company: str | None, intent: str, artifacts: dict, bi_encoder, cross_encoder, alias_map):
    """(VERSÃO CORRIGIDA) Usa a intenção para focar a busca e compara nomes normalizados."""
    candidate_chunks_with_source = []
    seen_texts = set()

    # Lógica para selecionar os índices a serem pesquisados com base na intenção
    target_categories = []
    if intent == 'item_8_4' and 'item_8_4' in artifacts:
        st.info("Focando a busca no índice 'Item 8.4' para maior precisão.")
        target_categories = ['item_8_4']
    else:
        target_categories = list(artifacts.keys())

    query_embedding = bi_encoder.encode(query, normalize_embeddings=True)
    
    for category in target_categories:
        data = artifacts.get(category)
        if not data or 'index' not in data or 'chunks_data' not in data: continue
        
        scores, ids = data['index'].search(np.array([query_embedding]).astype('float32'), TOP_K_INITIAL_SEARCH)
        
        for i, doc_id in enumerate(ids[0]):
            if doc_id != -1 and scores[0][i] > 0.3:
                chunk_map_item = data['chunks_data']['map'][doc_id]
                chunk_company_name = chunk_map_item.get("company_name", "")

                # Filtro de empresa NORMALIZADO E ROBUSTO
                if company and normalize_company_name(company) != normalize_company_name(chunk_company_name):
                    continue
                
                chunk_text = data['chunks_data']['chunks'][doc_id]
                if chunk_text not in seen_texts:
                    candidate_chunks_with_source.append({
                        'text': chunk_text, 
                        'source': chunk_map_item.get('source_url', 'Fonte desconhecida')
                    })
                    seen_texts.add(chunk_text)

    if not candidate_chunks_with_source:
        return "", set()

    # Lógica de Re-ranking
    pure_texts = [re.sub(r'^\[.*?\]\s*', '', c['text']) for c in candidate_chunks_with_source]
    sentence_pairs = [[query, text] for text in pure_texts]
    rerank_scores = cross_encoder.predict(sentence_pairs, show_progress_bar=False)
    reranked_results = sorted(zip(rerank_scores, candidate_chunks_with_source), key=lambda x: x[0], reverse=True)

    final_context, final_sources = "", set()
    for score, chunk_data in reranked_results[:TOP_K_RERANKED]:
        final_context += f"Fonte: {os.path.basename(chunk_data['source'])} (Relevância: {score:.2f})\n{chunk_data['text']}\n\n"
        final_sources.add(chunk_data['source'])
        
    return final_context, final_sources

def generate_answer_extract_synthesize(original_query, context):
    """(SÍNTESE SEGURA) Usa a técnica "Extrair-Depois-Sintetizar"."""
    # ETAPA 1: Extrair
    extract_prompt = f"""Analise o contexto e extraia fatos, números e citações relevantes para a pergunta.
    Pergunta: "{original_query}"
    Contexto: --- {context} ---
    Fatos Brutos Extraídos:"""
    extracted_facts = call_gemini_api(extract_prompt)
    if "ERRO:" in extracted_facts: return extracted_facts

    # ETAPA 2: Sintetizar
    synthesize_prompt = f"""Usando APENAS os fatos brutos extraídos, escreva uma resposta completa em português para a pergunta original. Se os fatos não forem suficientes, diga que a informação não foi encontrada.
    Pergunta Original: "{original_query}"
    Fatos Extraídos: --- {extracted_facts} ---
    Resposta Final:"""
    final_answer = call_gemini_api(synthesize_prompt)
    return final_answer


# --- 5. Orquestrador Principal do Agente v5.0 ---

def handle_specialist_rag_query(user_query, artifacts, bi_encoder, cross_encoder, alias_map, company_catalog, knowledge_base):
    """Orquestra o pipeline RAG, aplicando a lógica especialista quando aplicável."""
    st.info("Iniciando análise com motor RAG especialista v5.1...")

    # --- ETAPA DE PLANEJAMENTO INTELIGENTE ---
    plan = {'empresas': [], 'topicos_principais': [], 'topicos_expandidos': [], 'is_specialist_query': False}

    # Identificar empresas
    for company_data in company_catalog:
        for alias in company_data.get("aliases", []):
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', user_query.lower()):
                plan['empresas'].append(company_data["canonical_name"]); break
    plan['empresas'] = sorted(list(set(plan['empresas'])))

    # Identificar se a query aciona um "modo especialista"
    for term in sorted(alias_map.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(term) + r'\b', user_query.lower()):
            canonical_name = alias_map[term]
            # Uma query é "especialista" se ela menciona uma SEÇÃO principal do dicionário
            if canonical_name in knowledge_base:
                plan['is_specialist_query'] = True
                plan['topicos_principais'].append(canonical_name)
                # Expande o plano para incluir todos os sub-tópicos daquela seção
                plan['topicos_expandidos'].extend(list(knowledge_base[canonical_name].keys()))

    plan['topicos_principais'] = sorted(list(set(plan['topicos_principais'])))
    plan['topicos_expandidos'] = sorted(list(set(plan['topicos_expandidos'])))
    
    final_answer, full_sources = "", set()

    # --- ROTA 1: ANÁLISE ESPECIALISTA (ESTRUTURADA) ---
    if plan['is_specialist_query']:
        empresa = plan['empresas'][0] if plan['empresas'] else None
        if not empresa: 
            st.warning("Consulta especialista identificada, mas nenhuma empresa foi mencionada."); return "", set()
        
        st.success(f"Modo especialista ativado para a(s) seção(ões): **{', '.join(plan['topicos_principais'])}**")
        structured_context = {}
        
        for i, sub_topic_key in enumerate(plan['topicos_expandidos']):
            # Pega o primeiro alias como nome de exibição (ex: "Ações Restritas")
            display_name = knowledge_base[plan['topicos_principais'][0]].get(sub_topic_key, [sub_topic_key])[0]
            
            with st.spinner(f"Buscando sub-tópico {i+1}/{len(plan['topicos_expandidos'])}: '{display_name}'..."):
                query_focada = f"informações sobre {display_name} no plano de remuneração da empresa {empresa}"
                context_part, sources_part = retrieve_hybrid_and_reranked_context(query_focada, empresa, 'general', artifacts, bi_encoder, cross_encoder, alias_map)
                structured_context[display_name] = context_part
                full_sources.update(sources_part)
        
        final_context_str = "\n\n".join([f"--- Contexto para '{topic}' ---\n{text or 'Nenhuma informação específica encontrada.'}" for topic, text in structured_context.items()])
        final_answer = generate_answer_extract_synthesize(user_query, final_context_str) # A síntese já lida bem com isso
        return final_answer, full_sources

    # --- ROTA 2: ANÁLISE PADRÃO (SEM EXPANSÃO OU MODO COMPARATIVO) ---
    else:
        st.info("Iniciando análise RAG padrão.")
        # (Aqui entra a lógica do modo comparativo do seu v5.0, se desejar)
        # Por simplicidade, faremos a busca direta que já existia:
        empresa_unica = plan['empresas'][0] if plan['empresas'] else None
        context, sources = retrieve_hybrid_and_reranked_context(user_query, empresa_unica, 'general', artifacts, bi_encoder, cross_encoder, alias_map)
        if not context: st.warning("Não encontrei informações relevantes."); return "", set()
        final_answer = generate_answer_extract_synthesize(user_query, context)
        return final_answer, sources

def generate_final_answer(original_query: str, context: str):
    """Gera a resposta final, com um prompt adaptado se o contexto for estruturado."""
    # (Esta função pode ser uma versão simplificada da sua 'generate_answer_extract_synthesize')
    # O importante é que ela receba o contexto e a query e gere a resposta.
    return call_gemini_api(f"Responda a pergunta '{original_query}' usando o contexto: {context}")


# --- 6. Aplicação Principal (Interface Streamlit) ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP v5.0", page_icon="🏆", layout="wide", initial_sidebar_state="expanded")
    if "history" not in st.session_state: st.session_state.history = []
    
    st.title("🏆 Agente de Análise LTIP v5.0 (Definitivo)")

    with st.spinner("Inicializando sistemas e carregando modelos..."):
        bi_encoder, cross_encoder = load_models()
        artifacts, summary_data = load_artifacts()
        ALIAS_MAP = criar_mapa_de_alias(DICIONARIO_UNIFICADO_HIERARQUICO)

    with st.sidebar:
        st.header("📊 Informações do Sistema")
        total_chunks = sum(len(data['chunks_data']['chunks']) for cat, data in artifacts.items()) if artifacts else 0
        st.metric("Documentos Indexados (Chunks)", f"{total_chunks:,}")
        st.metric("Empresas no Resumo Rápido", len(summary_data) if summary_data else "0")
        st.success("✅ Sistema pronto")

        with st.expander("📜 Histórico da Sessão", expanded=False):
            if not st.session_state.history:
                st.caption("Nenhuma pergunta feita nesta sessão.")
            else:
                for i, entry in enumerate(reversed(st.session_state.history)):
                    st.info(f"**P{len(st.session_state.history) - i}:** {entry['query']}")
    
    st.header("💬 Faça sua pergunta")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Experimente análises rápidas:**")
        st.code("Quais empresas possuem planos com matching?")
        st.code("Qual o desconto médio oferecido?")
    with col2:
        st.info("**Ou uma análise profunda:**")
        st.code("Compare as políticas de vesting e clawback da Vale e Itaú")
        st.code("Resumo do item 8.4 da Ambev")

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Compare o TSR da Vale com a Petrobras ou pergunte sobre o vesting da Ambev...")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip(): st.warning("⚠️ Por favor, digite uma pergunta."); st.stop()
        st.markdown("---"); st.subheader("📋 Resultado da Análise")

        # --- ROTEADOR DE MÚLTIPLOS NÍVEIS ---
        analytical_keywords = ['medio', 'média', 'típico', 'menor', 'mínimo', 'maior', 'máximo']
        aggregate_keywords = ["quais", "quantas", "liste", "mostre"]
        
        final_answer, sources = "", set()

        if any(kw in user_query.lower() for kw in analytical_keywords):
            st.info("Detectada pergunta **analítica**. Usando o resumo rápido...")
            if not handle_analytical_query(user_query, summary_data):
                st.warning("Análise rápida sem dados. Acionando motor RAG para uma resposta completa...")
                final_answer, sources = handle_definitive_rag_query(user_query, artifacts, bi_encoder, cross_encoder, ALIAS_MAP, company_catalog_rich)
        
        elif any(kw in user_query.lower() for kw in aggregate_keywords):
            st.info("Detectada pergunta **agregada**. Usando o resumo rápido...")
            if not handle_aggregate_query(user_query, summary_data, ALIAS_MAP):
                st.warning("Análise rápida sem dados. Acionando motor RAG para uma resposta completa...")
                final_answer, sources = handle_definitive_rag_query(user_query, artifacts, bi_encoder, cross_encoder, ALIAS_MAP, company_catalog_rich)

        # (Você pode adicionar o `handle_direct_fact_query` aqui se desejar)

        else:
            final_answer, sources = handle_definitive_rag_query(user_query, artifacts, bi_encoder, cross_encoder, ALIAS_MAP, company_catalog_rich)
            
        if final_answer and "ERRO:" not in final_answer:
            st.markdown(final_answer)
            st.session_state.history.append({'query': user_query, 'answer': final_answer})
        
        if sources:
            with st.expander(f"📚 Fontes Consultadas ({len(sources)})"):
                st.dataframe(pd.DataFrame(sorted(list(sources)), columns=["Documento"]), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
