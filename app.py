# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP - VERSÃO STREAMLIT
Aplicação web para análise de planos de incentivo de longo prazo
"""

import streamlit as st
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import glob
import os
import re
import unicodedata
# CORREÇÃO: Removido caractere invisível no final do import
from catalog_data import company_catalog_rich

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- DICIONÁRIOS E LISTAS DE TÓPICOS (sem alterações) ---
TERMOS_TECNICOS_LTIP = {
    # ... (seu dicionário de termos permanece igual)
}
AVAILABLE_TOPICS = [
    # ... (sua lista de tópicos permanece igual)
]

# --- FUNÇÕES AUXILIARES (sem alterações) ---
def expand_search_terms(base_term):
    # ... (sua função permanece igual)
    expanded_terms = [base_term.lower()]
    for category, terms in TERMOS_TECNICOS_LTIP.items():
        if any(term.lower() in base_term.lower() for term in terms):
            expanded_terms.extend([term.lower() for term in terms])
    return list(set(expanded_terms))


def search_by_tags(artifacts, company_name, target_tags):
    # ... (sua função permanece igual)
    results = []
    for index_name, artifact_data in artifacts.items():
        chunk_data = artifact_data['chunks']
        for i, mapping in enumerate(chunk_data.get('map', [])):
            document_path = mapping['document_path']
            if re.search(re.escape(company_name.split(' ')[0]), document_path, re.IGNORECASE):
                chunk_text = chunk_data["chunks"][i]
                for tag in target_tags:
                    if f"Tópicos:" in chunk_text and tag in chunk_text:
                        results.append({
                            'text': chunk_text, 'path': document_path, 'index': i,
                            'source': index_name, 'tag_found': tag
                        })
                        break
    return results


def normalize_name(name):
    """Normaliza nomes para comparação, removendo acentos, pontuação, sufixos e espaços."""
    try:
        nfkd_form = unicodedata.normalize('NFKD', name.lower())
        name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        name = re.sub(r'[.,-]', '', name)
        suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)
        return re.sub(r'\s+', '', name).strip()
    except Exception as e:
        return name.lower()


# --- CACHE PARA CARREGAR ARTEFATOS ---
@st.cache_resource
def load_all_artifacts():
    """Carrega todos os artefatos (índices FAISS, chunks e modelo de embedding)."""
    artifacts = {}
    model = SentenceTransformer(MODEL_NAME)
    dados_path = "dados"
    index_files = glob.glob(os.path.join(dados_path, '*_faiss_index.bin'))

    if not index_files:
        return None, None

    for index_file in index_files:
        category = os.path.basename(index_file).replace('_faiss_index.bin', '')
        chunks_file = os.path.join(dados_path, f"{category}_chunks_map.json")
        try:
            index = faiss.read_index(index_file)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
        except FileNotFoundError:
            continue

    if not artifacts:
        return None, None

    # LIMPEZA: Não precisamos mais criar o catálogo antigo aqui.
    return artifacts, model

# --- FUNÇÕES PRINCIPAIS ---

# NOVO: Esta é a nova função de análise com lógica de scoring.
def create_dynamic_analysis_plan_v2(query, company_catalog_rich, available_indices):
    """
    Gera um plano de ação dinâmico com identificação de empresas baseada em scoring
    para maior precisão, lidando com nomes compostos, apelidos e variações.
    """
    api_key = GEMINI_API_KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

    # --- LÓGICA DE IDENTIFICAÇÃO DE EMPRESAS POR SCORING ---
    query_lower = query.lower().strip()
    normalized_query = normalize_name(query_lower)
    company_scores = {}

    for company_data in company_catalog_rich:
        canonical_name = company_data["canonical_name"]
        score = 0
        
        # Pass 1: Correspondência de apelidos (maior pontuação)
        for alias in company_data.get("aliases", []):
            if alias in query_lower:
                score += 10

        # Pass 2: Correspondência de palavras-chave individuais (menor pontuação)
        name_for_parts = re.sub(r'[.,()]', '', canonical_name)
        suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
        for suffix in suffixes:
            name_for_parts = re.sub(suffix, '', name_for_parts, flags=re.IGNORECASE)
        
        parts = name_for_parts.split()
        for part in parts:
            key = normalize_name(part)
            if len(key) > 2 and key in normalized_query:
                score += 1

        if score > 0:
            company_scores[canonical_name] = score

    mentioned_companies = []
    if company_scores:
        sorted_companies = sorted(company_scores.items(), key=lambda item: item[1], reverse=True)
        max_score = sorted_companies[0][1]
        # Pega empresas com pontuação alta (pelo menos 80% do máximo) para capturar ambiguidades relevantes
        mentioned_companies = [company for company, score in sorted_companies if score >= max_score * 0.8 and max_score > 1]
        if not mentioned_companies:
            mentioned_companies = [sorted_companies[0][0]] # Se todos tiverem score baixo, pega o melhor

    # --- FIM DA LÓGICA DE IDENTIFICAÇÃO ---
    prompt = f"""
    Você é um planejador de análise. Sua tarefa é analisar a "Pergunta do Usuário" e identificar os tópicos de interesse.
    **Instruções:**
    1. **Identifique os Tópicos:** Analise a pergunta para identificar os tópicos de interesse. Se a pergunta for genérica (ex: "resumo dos planos", "análise da empresa"), inclua todos os "Tópicos de Análise Disponíveis". Se for específica (ex: "fale sobre o vesting e dividendos"), inclua apenas os tópicos relevantes.
    2. **Formate a Saída:** Retorne APENAS uma lista JSON de strings contendo os tópicos identificados.
    **Tópicos de Análise Disponíveis:** {json.dumps(AVAILABLE_TOPICS, indent=2)}
    **Pergunta do Usuário:** "{query}"
    **Tópicos de Interesse (responda APENAS com a lista JSON de strings):**
    """
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
            topics = AVAILABLE_TOPICS
    except Exception:
        topics = AVAILABLE_TOPICS
        
    plan = {"empresas": list(mentioned_companies), "topicos": topics}
    return {"status": "success", "plan": plan}


def execute_dynamic_plan(plan, query_intent, artifacts, model):
    # ... (sua função permanece igual)
    # Nenhuma alteração necessária aqui
    full_context = ""
    all_retrieved_docs = set()
    return full_context, [str(doc) for doc in all_retrieved_docs]


def get_final_unified_answer(query, context):
    # ... (sua função permanece igual)
    # Nenhuma alteração necessária aqui
    return "Resposta final gerada aqui."


# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(
        page_title="Agente de Análise LTIP",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 Agente de Análise de Planos de Incentivo Longo Prazo ILP")
    st.markdown("---")
    
    with st.spinner("Inicializando sistema..."):
        # CORREÇÃO: A função agora retorna 2 valores, o catálogo antigo foi removido
        loaded_artifacts, embedding_model = load_all_artifacts()
    
    if not loaded_artifacts:
        st.error("❌ Erro no carregamento dos artefatos. Verifique os arquivos na pasta 'dados'.")
        return
    
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Fontes disponíveis", len(loaded_artifacts))
        st.metric("Empresas identificadas", len(company_catalog_rich))
        
        with st.expander("📋 Ver empresas disponíveis"):
            sorted_companies = sorted([company['canonical_name'] for company in company_catalog_rich])
            for company_name in sorted_companies:
                st.write(f"• {company_name}")
        
        with st.expander("📁 Ver fontes de dados"):
            for source in loaded_artifacts.keys():
                st.write(f"• {source}")
        
        st.markdown("---")
        st.success("✅ Sistema carregado")
        st.info(f"Modelo: {MODEL_NAME}")

    st.header("💬 Faça sua pergunta")
    
    with st.expander("💡 Exemplos de perguntas"):
        # ... (sem alterações aqui)
        pass

    user_query = st.text_area(
        "Digite sua pergunta sobre planos de incentivo:",
        height=100,
        placeholder="Ex: Fale sobre o vesting da Magalu ou planos da Vibra Energia",
        help="Seja específico sobre a empresa e o tópico de interesse"
    )

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return
        
        with st.container():
            st.markdown("---")
            st.subheader("📋 Processo de Análise")
            
            with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
                # CORREÇÃO: A chamada agora usa a nova função 'create_dynamic_analysis_plan_v2' 
                # e o catálogo 'company_catalog_rich'.
                plan_response = create_dynamic_analysis_plan_v2(
                    user_query,
                    company_catalog_rich,
                    list(loaded_artifacts.keys())
                )
                
                if plan_response['status'] != 'success':
                    st.error("❌ Erro ao gerar plano de análise")
                    return
                
                plan = plan_response['plan']
                
                if plan.get('empresas'):
                    st.write(f"**🏢 Empresas identificadas:** {', '.join(plan.get('empresas', []))}")
                else:
                    st.write("**🏢 Empresas identificadas:** Nenhuma")
                
                st.write(f"**📝 Tópicos a analisar:** {len(plan.get('topicos', []))}")
                status.update(label="✅ Plano gerado com sucesso!", state="complete")

            if not plan.get("empresas"):
                st.error("❌ Não consegui identificar empresas na sua pergunta. Tente usar nomes, apelidos ou marcas conhecidas (ex: Magalu, Vivo, Itaú).")
                return
            
            # O resto do código para executar o plano e mostrar a resposta permanece o mesmo.
            # ... (código para Etapa 2 e Etapa 3)

if __name__ == "__main__":
    main()
