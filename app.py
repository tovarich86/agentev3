# app.py
# -*- coding: utf-8 -*-
"""
AGENTE COM PLANEJAMENTO DINÂMICO - VERSÃO FINAL (STREAMLIT, MAP-REDUCE, SEGURA E ESCALÁVEL)

Este script implementa o agente de análise de LTIP como uma aplicação web interativa
usando Streamlit. A arquitetura usa "Map-Reduce com Agrupamento de Chunks" para analisar
grandes volumes de documentos de forma rápida e escalável, evitando erros de limite de
requisições (429) e protegendo chaves de API contra exposição em erros.
"""

# --- IMPORTAÇÕES ---
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
import time

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(page_title="Agente de Análise ILP", layout="wide")
st.title("🤖 Agente de Análise de Planos de Incentivo de Longo Prazo")
st.caption("Uma aplicação de IA para analisar documentos de ILP de forma escalável.")

# --- CONFIGURAÇÕES E ESTRUTURAS DE CONHECIMENTO ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
GOOGLE_DRIVE_PATH = './dados'

TERMOS_TECNICOS_LTIP = {
    "Ações Restritas": ["Restricted Shares", "Plano de Ações Restritas", "Outorga de Ações", "ações restritas"],
    "Opções de Compra de Ações": ["Stock Options", "ESOP", "Plano de Opção de Compra", "Outorga de Opções", "opções", "Plano de Opção", "Plano de Opções"],
    "Ações Fantasmas": ["Phantom Shares", "Ações Virtuais"],
    "Opções Fantasmas (SAR)": ["Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
    "Bônus Diferido": ["Staying Bonus", "Retention Bonus", "Bônus de Permanência", "Bônus de Retenção", "bônus"],
    "Planos com Condição de Performance": ["Performance Shares", "Performance Stock Options", "Plano de Desempenho", "Metas de Performance", "performance", "desempenho"],
    "Vesting": ["Período de Carência", "Condições de Carência", "Aquisição de Direitos", "carência"],
    "Antecipação de Vesting": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação de carência", "antecipação do vesting", "antecipação"],
    "Tranche / Lote": ["Tranche", "Lote", "Parcela do Vesting"],
    "Cliff": ["Cliff Period", "Período de Cliff", "Carência Inicial"],
    "Matching": ["Contrapartida", "Co-investimento", "Plano de Matching", "matching", "investimento"],
    "Lockup": ["Período de Lockup", "Restrição de Venda", "lockup"],
    "Estrutura do Plano/Programa": ["Plano", "Planos", "Programa", "Programas"],
    "Ciclo de Vida do Exercício": ["pagamento", "liquidação", "vencimento", "expiração"],
    "Eventos Corporativos": ["IPO", "grupamento", "desdobramento", "bonificações", "bonificação"],
    "Encargos": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"],
}
AVAILABLE_TOPICS = [
    "termos e condições gerais", "data de aprovação e órgão responsável", "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas", "condições de aquisição de ações", "critérios para fixação do preço de aquisição ou exercício", "preço de exercício", "strike price", "critérios para fixação do prazo de aquisição ou exercício", "forma de liquidação", "liquidação", "pagamento", "restrições à transferência das ações", "critérios e eventos de suspensão/extinção", "efeitos da saída do administrador", "Tipos de Planos", "Condições de Carência", "Vesting", "período de carência", "cronograma de vesting", "Matching", "contrapartida", "co-investimento", "Lockup", "período de lockup", "restrição de venda", "Tratamento de Dividendos", "equivalente em dividendos", "proventos", "Stock Options", "opções de ações", "SOP", "Ações Restritas", "RSU", "restricted shares", "Eventos Corporativos", "IPO", "grupamento", "desdobramento"
]

# --- FUNÇÃO SEGURA PARA CHAMADAS DE API ---
def safe_api_call(url, payload, headers, timeout=90):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        reason = e.response.reason
        return None, f"Erro de API com código {status_code}: {reason}. Por favor, tente novamente mais tarde."
    except requests.exceptions.RequestException:
        return None, "Erro de conexão ao tentar contatar a API. Verifique sua conexão com a internet."

# --- FUNÇÕES DE LÓGICA ---
def expand_search_terms(base_term):
    expanded_terms = [base_term.lower()]
    for category, terms in TERMOS_TECNICOS_LTIP.items():
        if any(term.lower() in base_term.lower() for term in terms):
            expanded_terms.extend([term.lower() for term in terms])
    return list(set(expanded_terms))

def search_by_tags(artifacts, company_name, target_tags):
    results = []
    for index_name, artifact_data in artifacts.items():
        chunk_data = artifact_data['chunks']
        for i, mapping in enumerate(chunk_data.get('map', [])):
            document_path = mapping['document_path']
            if document_path.lower().startswith(company_name.lower()):
                chunk_text = chunk_data["chunks"][i]
                
                for tag in target_tags:
                    if f"Tópicos:" in chunk_text and tag in chunk_text:
                        results.append({'text': chunk_text, 'path': document_path, 'source': index_name, 'tag_found': tag})
                        break
    return results

@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    canonical_company_names = set()
    with st.spinner("Carregando modelo de embedding (isso só acontece na primeira vez)..."):
        model = SentenceTransformer(MODEL_NAME)
    
    index_files = glob.glob(os.path.join(GOOGLE_DRIVE_PATH, '*_faiss_index.bin'))
    if not index_files:
        st.error(f"ERRO CRÍTICO: Nenhum arquivo de índice (*.bin) encontrado na pasta '{GOOGLE_DRIVE_PATH}'. Verifique se os arquivos de dados estão na pasta 'dados' no GitHub.")
        return None, None, None

    for index_file in index_files:
        category = os.path.basename(index_file).replace('_faiss_index.bin', '')
        chunks_file = os.path.join(GOOGLE_DRIVE_PATH, f"{category}_chunks_map.json")
        try:
            index = faiss.read_index(index_file)
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
            for mapping in chunk_data.get('map', []):
                parts = mapping['document_path'].split('/')
                if parts:
                    canonical_company_names.add(parts[0])
        except FileNotFoundError:
            st.warning(f"AVISO: Arquivo de chunks '{chunks_file}' não encontrado.")
            continue
    
    if not artifacts:
        st.error("ERRO CRÍTICO: Nenhum artefato foi carregado com sucesso.")
        return None, None, None
    return artifacts, model, list(canonical_company_names)


def create_dynamic_analysis_plan(_query, company_catalog, available_indices):
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    def normalize_name(name):
        try:
            nfkd_form = unicodedata.normalize('NFKD', name.lower()); name = "".join([c for c in nfkd_form if not unicodedata.combining(c)]); name = re.sub(r'[.,-]', '', name); suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b'];
            for suffix in suffixes: name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            return re.sub(r'\s+', '', name).strip()
        except Exception: return name.lower()
    query_words = set(re.findall(r'\b\w{3,}\b', _query.lower())) # Pega palavras com 3+ letras da pergunta
    mentioned_companies = []
    for canonical_name in company_catalog:
        # Verifica se alguma palavra da pergunta está no nome da empresa
        # Isso é mais direto e menos propenso a erros
        company_name_lower = canonical_name.lower()
        for word in query_words:
            if word in company_name_lower:
                if canonical_name not in mentioned_companies:
                    mentioned_companies.append(canonical_name)
                break # Evita adicionar a mesma empresa várias vezes
    if not mentioned_companies and len(query_clean) <= 6:
        for canonical_name in company_catalog:
            if query_clean.upper() in canonical_name.upper():
                if canonical_name not in mentioned_companies: mentioned_companies.append(canonical_name)
    
    prompt = f'Você é um planejador de análise. Sua tarefa é analisar a "Pergunta do Usuário" e identificar os tópicos de interesse. Se a pergunta for genérica (ex: "resumo dos planos"), inclua todos os "Tópicos de Análise Disponíveis". Se for específica, inclua apenas os tópicos relevantes. Retorne APENAS uma lista JSON de strings. Tópicos de Análise Disponíveis: {json.dumps(AVAILABLE_TOPICS)}. Pergunta do Usuário: "{_query}". Tópicos de Interesse (JSON):'
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    response_data, error_message = safe_api_call(url, payload, headers)
    if error_message:
        st.warning(f"Erro no planejamento: {error_message}")
        plan = {"empresas": list(set(mentioned_companies)), "topicos": AVAILABLE_TOPICS}
        return {"status": "success", "plan": plan}
    text_response = response_data['candidates'][0]['content']['parts'][0]['text']
    json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
    if json_match:
        topics = json.loads(json_match.group(0))
        plan = {"empresas": list(set(mentioned_companies)), "topicos": topics}
        return {"status": "success", "plan": plan}
    plan = {"empresas": list(set(mentioned_companies)), "topicos": AVAILABLE_TOPICS}
    return {"status": "success", "plan": plan}

def execute_dynamic_plan(plan, query_intent, artifacts, model):
    full_context = ""
    all_retrieved_docs = set()
    for empresa in plan.get("empresas", []):
        full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
        if query_intent == 'item_8_4_query' and 'item_8_4' in artifacts:
            artifact_data = artifacts['item_8_4']
            chunk_data = artifact_data['chunks']
            for i, mapping in enumerate(chunk_data.get('map', [])):
                if re.search(re.escape(empresa.split(' ')[0]), mapping['document_path'], re.IGNORECASE):
                    all_retrieved_docs.add(mapping['document_path'])
                    full_context += f"--- Chunk Item 8.4 (Doc: {mapping['document_path']}) ---\n{chunk_data['chunks'][i]}\n\n"
        target_tags = []
        for topico in plan.get("topicos", []): target_tags.extend(expand_search_terms(topico))
        target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
        tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
        for chunk_info in tagged_chunks:
            full_context += f"--- Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']}) ---\n{chunk_info['text']}\n\n"
            all_retrieved_docs.add(chunk_info['path'])
    return full_context, [str(doc) for doc in all_retrieved_docs]

# --- FUNÇÕES DE SÍNTESE (ESTRATÉGIA MAP-REDUCE) ---
def summarize_chunk(chunk_text, query):
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    prompt = f'Com base na pergunta do usuário e no chunk de texto abaixo, extraia e resuma APENAS as informações relevantes. Se o chunk não contiver informações relevantes, responda com "N/A".\n\nPergunta do Usuário: "{query}"\n\nChunk de Texto:\n---\n{chunk_text}\n---\n\nResumo Conciso das Informações Relevantes:'
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024}}
    headers = {'Content-Type': 'application/json'}
    response_data, error_message = safe_api_call(url, payload, headers)
    if error_message:
        print(f"Erro ao resumir chunk: {error_message}")
        return None
    summary = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    if summary.upper() != "N/A" and len(summary) > 20:
        return summary
    return None

def get_final_unified_answer(query, context, plan):
    chunks = re.split(r'--- Chunk|--- INÍCIO DA ANÁLISE PARA:', context)
    relevant_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
    if not relevant_chunks:
        return "Não encontrei conteúdo relevante nos documentos para analisar."

    TOKEN_LIMIT_PER_BATCH = 6000
    chunk_batches = []; current_batch = []; current_batch_tokens = 0
    st.info(f"Encontrados {len(relevant_chunks)} chunks. Agrupando em lotes otimizados...")
    for chunk in relevant_chunks:
        chunk_tokens = len(chunk) / 3
        if current_batch_tokens + chunk_tokens > TOKEN_LIMIT_PER_BATCH and current_batch:
            chunk_batches.append("\n\n---\n\n".join(current_batch))
            current_batch = [chunk]; current_batch_tokens = chunk_tokens
        else:
            current_batch.append(chunk); current_batch_tokens += chunk_tokens
    if current_batch:
        chunk_batches.append("\n\n---\n\n".join(current_batch))
    st.info(f"Chunks agrupados em {len(chunk_batches)} lotes para uma análise mais rápida.")
    
    summaries = []
    progress_bar = st.progress(0, text="Mapeando e resumindo lotes de chunks...")
    for i, batch_text in enumerate(chunk_batches):
        summary = summarize_chunk(batch_text, query)
        if summary: summaries.append(summary)
        time.sleep(1.1)
        progress_bar.progress((i + 1) / len(chunk_batches), text=f"Mapeando e resumindo lotes... ({i+1}/{len(chunk_batches)})")
    progress_bar.empty()

    if not summaries:
        return "Não foi possível extrair informações relevantes dos documentos para montar um relatório."

    st.info("Todos os lotes foram analisados. Sintetizando o relatório final...")
    final_context = "\n\n---\n\n".join(summaries)
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    final_prompt = f'Você é um analista financeiro sênior. Sua tarefa é criar um relatório coeso e bem estruturado respondendo à pergunta do usuário. Use os resumos de contexto fornecidos abaixo, que foram extraídos de vários documentos. Sintetize as informações em uma única resposta final. Não liste os resumos, use-os para construir seu texto.\n\nPergunta Original do Usuário: "{query}"\n\nResumos de Contexto para usar como base:\n---\n{final_context}\n---\n\nRelatório Analítico Final (responda de forma completa e profissional):'
    payload = {"contents": [{"parts": [{"text": final_prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    response_data, error_message = safe_api_call(url, payload, headers, timeout=180)
    if error_message:
        return f"ERRO ao gerar a síntese final do relatório: {error_message}"
    return response_data['candidates'][0]['content']['parts'][0]['text'].strip()

# --- LÓGICA PRINCIPAL DA APLICAÇÃO STREAMLIT ---
try:
    loaded_artifacts, embedding_model, company_catalog = load_all_artifacts()
except Exception as e:
    st.error(f"Ocorreu um erro fatal durante o carregamento inicial dos recursos: {e}")
    st.stop()

if loaded_artifacts:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Qual empresa ou plano de incentivo você gostaria de analisar?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Digite sua pergunta sobre CCR, Vale, Vibra, etc."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Iniciando análise..."):
                plan_response = create_dynamic_analysis_plan(prompt, company_catalog, list(loaded_artifacts.keys()))
                if plan_response['status'] != 'success' or not plan_response['plan'].get("empresas"):
                    response_text = "Não consegui identificar uma empresa em sua pergunta ou houve um erro no planejamento. Por favor, seja mais específico."
                else:
                    plan = plan_response['plan']
                    st.info(f"Plano de análise criado. Foco em: {plan['empresas'][0] if plan['empresas'] else 'N/A'}.")
                    query_intent = 'item_8_4_query' if any(term in prompt.lower() for term in ['8.4', '8-4', 'item 8.4', 'formulário']) else 'general_query'
                    retrieved_context, sources = execute_dynamic_plan(plan, query_intent, loaded_artifacts, embedding_model)
                    if not retrieved_context.strip():
                        response_text = "Não encontrei informações relevantes nos documentos para a sua solicitação."
                    else:
                        final_answer = get_final_unified_answer(prompt, retrieved_context, plan)
                        response_text = final_answer
                        if sources:
                            unique_sources = sorted(list(set(sources)))
                            with st.expander(f"Fontes Consultadas ({len(unique_sources)})"):
                                for source in unique_sources:
                                    st.write(f"- {source}")
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

