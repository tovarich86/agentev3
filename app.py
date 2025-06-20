# app.py
# -*- coding: utf-8 -*-
"""
AGENTE COM PLANEJAMENTO DINÂMICO (PLANNER-EXECUTOR) - VERSÃO STREAMLIT

Este script adapta o agente de análise de LTIP para uma aplicação web interativa
usando Streamlit. Todas as funções de lógica foram integradas em um único
arquivo para facilitar o deploy via GitHub Web.
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

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(page_title="Agente de Análise LTIP", layout="wide")
st.title("🤖 Agente de Análise de Planos de Incentivo")
st.caption("Uma aplicação que usa IA para analisar Formulários de Referência da CVM.")

# --- CONFIGURAÇÕES E ESTRUTURAS DE CONHECIMENTO ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
AMBIGUITY_THRESHOLD = 3 
# O caminho agora aponta para a pasta 'dados' no repositório do GitHub
GOOGLE_DRIVE_PATH = './dados'

# Dicionário especializado para termos técnicos de LTIP
TERMOS_TECNICOS_LTIP = {
    "tratamento de dividendos": [
        "tratamento de dividendos", "equivalente em dividendos", "dividendos", 
        "juros sobre capital próprio", "proventos", "dividend equivalent",
        "dividendos pagos em ações", "ajustes por dividendos"
    ],
    "preço de exercício": [
        "preço de exercício", "strike price", "preço de compra", "preço fixo", 
        "valor de exercício", "preço pré-estabelecido", "preço de aquisição"
    ],
    "forma de liquidação": [
        "forma de liquidação", "liquidação", "pagamento", "entrega física", 
        "pagamento em dinheiro", "transferência de ações", "settlement"
    ],
    "vesting": [
        "vesting", "período de carência", "carência", "aquisição de direitos", 
        "cronograma de vesting", "vesting schedule", "período de cliff"
    ],
    "eventos corporativos": [
        "eventos corporativos", "desdobramento", "grupamento", "dividendos pagos em ações",
        "bonificação", "split", "ajustes", "reorganização societária"
    ],
    "stock options": [
        "stock options", "opções de ações", "opções de compra", "SOP", 
        "plano de opções", "ESOP", "opção de compra de ações"
    ],
    "ações restritas": [
        "ações restritas", "restricted shares", "RSU", "restricted stock units", 
        "ações com restrição", "plano de ações restritas"
    ]
}

# Tópicos expandidos de análise que o planejador pode usar
AVAILABLE_TOPICS = [
    "termos e condições gerais", "data de aprovação e órgão responsável",
    "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas",
    "condições de aquisição de ações", "critérios para fixação do preço de aquisição ou exercício",
    "preço de exercício", "strike price",
    "critérios para fixação do prazo de aquisição ou exercício", 
    "forma de liquidação", "liquidação", "pagamento",
    "restrições à transferência das ações", "critérios e eventos de suspensão/extinção",
    "efeitos da saída do administrador", "Tipos de Planos", "Condições de Carência", 
    "Vesting", "período de carência", "cronograma de vesting",
    "Matching", "contrapartida", "co-investimento",
    "Lockup", "período de lockup", "restrição de venda",
    "Tratamento de Dividendos", "equivalente em dividendos", "proventos",
    "Stock Options", "opções de ações", "SOP",
    "Ações Restritas", "RSU", "restricted shares",
    "Eventos Corporativos", "IPO", "grupamento", "desdobramento"
]

# --- FUNÇÕES DE LÓGICA ---

def expand_search_terms(base_term):
    """Expande um termo base com sinônimos e variações técnicas."""
    expanded_terms = [base_term.lower()]
    for category, terms in TERMOS_TECNICOS_LTIP.items():
        if any(term.lower() in base_term.lower() for term in terms):
            expanded_terms.extend([term.lower() for term in terms])
    return list(set(expanded_terms))

def search_by_tags(artifacts, company_name, target_tags):
    """Busca chunks que contenham tags específicas para uma empresa."""
    results = []
    for index_name, artifact_data in artifacts.items():
        chunk_data = artifact_data['chunks']
        for i, mapping in enumerate(chunk_data.get('map', [])):
            document_path = mapping['document_path']
            # Busca pelo primeiro nome da empresa para maior flexibilidade
            if re.search(re.escape(company_name.split(' ')[0]), document_path, re.IGNORECASE):
                chunk_text = chunk_data["chunks"][i]
                for tag in target_tags:
                    if f"Tópicos:" in chunk_text and tag in chunk_text:
                        results.append({
                            'text': chunk_text, 'path': document_path, 'index': i,
                            'source': index_name, 'tag_found': tag
                        })
                        break # Vai para o próximo chunk assim que achar uma tag
    return results

# --- FUNÇÕES DE CARREGAMENTO E CACHE (ADAPTADAS PARA STREAMLIT) ---
@st.cache_resource
def load_all_artifacts():
    """Carrega todos os artefatos e constrói um catálogo. Usa cache do Streamlit."""
    artifacts = {}
    canonical_company_names = set()
    
    with st.spinner("Carregando modelo de embedding (isso só acontece na primeira vez)..."):
        model = SentenceTransformer(MODEL_NAME)

    index_files = glob.glob(os.path.join(GOOGLE_DRIVE_PATH, '*_faiss_index.bin'))
    if not index_files:
        st.error(f"ERRO CRÍTICO: Nenhum arquivo de índice (*_faiss_index.bin) encontrado na pasta '{GOOGLE_DRIVE_PATH}'. Verifique se os arquivos de dados foram enviados para o GitHub.")
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
                # Extrai o nome da empresa do caminho do documento
                parts = mapping['document_path'].split('/')
                if parts:
                    canonical_company_names.add(parts[0])
        except FileNotFoundError:
            st.warning(f"AVISO: Arquivo de chunks '{chunks_file}' não encontrado para o índice '{index_file}'. Pulando.")
            continue
    
    if not artifacts:
        st.error("ERRO CRÍTICO: Nenhum artefato foi carregado com sucesso.")
        return None, None, None

    return artifacts, model, list(canonical_company_names)

# --- FUNÇÕES DE INTERAÇÃO COM A IA (ADAPTADAS PARA STREAMLIT) ---

def create_dynamic_analysis_plan(query, company_catalog, available_indices):
    """CHAMADA AO LLM PLANEJADOR: Gera um plano de ação dinâmico em JSON."""
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    def normalize_name(name):
        try:
            nfkd_form = unicodedata.normalize('NFKD', name.lower())
            name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
            name = re.sub(r'[.,-]', '', name)
            suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
            for suffix in suffixes:
                name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            return re.sub(r'\s+', '', name).strip()
        except Exception:
            return name.lower()

    mentioned_companies = []
    query_clean = query.lower().strip()
    
    for canonical_name in company_catalog:
        # Busca por correspondência exata, substring, partes do nome, e normalizada
        if (canonical_name.lower() in query_clean or 
            any(len(part) > 2 and re.search(r'\b' + re.escape(part.lower()) + r'\b', query_clean) for part in canonical_name.split(' ')) or
            (len(query_clean) > 2 and normalize_name(query_clean) in normalize_name(canonical_name))):
            if canonical_name not in mentioned_companies:
                mentioned_companies.append(canonical_name)

    if not mentioned_companies and len(query_clean) <= 6:
        for canonical_name in company_catalog:
            if query_clean.upper() in canonical_name.upper():
                if canonical_name not in mentioned_companies:
                    mentioned_companies.append(canonical_name)
    
    prompt = f"""
    Você é um planejador de análise. Sua tarefa é analisar a "Pergunta do Usuário" e identificar os tópicos de interesse.
    Instruções:
    1. Identifique os Tópicos: Analise a pergunta para identificar os tópicos de interesse. Se a pergunta for genérica (ex: "resumo dos planos"), inclua todos os "Tópicos de Análise Disponíveis". Se for específica (ex: "fale sobre o vesting e dividendos"), inclua apenas os tópicos relevantes.
    2. Formate a Saída: Retorne APENAS uma lista JSON de strings contendo os tópicos identificados.
    Tópicos de Análise Disponíveis: {json.dumps(AVAILABLE_TOPICS, indent=2)}
    Pergunta do Usuário: "{query}"
    Tópicos de Interesse (responda APENAS com a lista JSON de strings):
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
            plan = {"empresas": list(set(mentioned_companies)), "topicos": topics}
            return {"status": "success", "plan": plan}
    except Exception as e:
        st.warning(f"Erro no planejamento, usando fallback. Detalhe: {e}")
        # Fallback em caso de erro
    plan = {"empresas": list(set(mentioned_companies)), "topicos": AVAILABLE_TOPICS}
    return {"status": "success", "plan": plan}


def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """EXECUTOR APRIMORADO: Busca exaustiva no item 8.4 + busca por tags + expansão de termos."""
    full_context = ""
    all_retrieved_docs = set()
    
    for empresa in plan.get("empresas", []):
        full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
        
        if query_intent == 'item_8_4_query' and 'item_8_4' in artifacts:
            # Estratégia de busca exaustiva para o item 8.4
            artifact_data = artifacts['item_8_4']
            chunk_data = artifact_data['chunks']
            empresa_chunks_8_4 = []
            for i, mapping in enumerate(chunk_data.get('map', [])):
                if re.search(re.escape(empresa.split(' ')[0]), mapping['document_path'], re.IGNORECASE):
                    chunk_text = chunk_data["chunks"][i]
                    all_retrieved_docs.add(mapping['document_path'])
                    empresa_chunks_8_4.append({'text': chunk_text, 'path': mapping['document_path']})
            if empresa_chunks_8_4:
                full_context += f"=== SEÇÃO COMPLETA DO ITEM 8.4 - {empresa.upper()} ===\n\n"
                for chunk_info in empresa_chunks_8_4:
                    full_context += f"--- Chunk Item 8.4 (Doc: {chunk_info['path']}) ---\n{chunk_info['text']}\n\n"
                full_context += f"=== FIM DA SEÇÃO ITEM 8.4 - {empresa.upper()} ===\n\n"

        # Busca por tags e semântica em todos os casos (geral ou como complemento ao 8.4)
        target_tags = []
        for topico in plan.get("topicos", []):
            target_tags.extend(expand_search_terms(topico))
        target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
        
        tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
        if tagged_chunks:
            full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
            for chunk_info in tagged_chunks:
                full_context += f"--- Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']}) ---\n{chunk_info['text']}\n\n"
                all_retrieved_docs.add(chunk_info['path'])
            full_context += f"=== FIM DOS CHUNKS COM TAGS - {empresa.upper()} ===\n\n"
            
    return full_context, [str(doc) for doc in all_retrieved_docs]

def get_final_unified_answer(query, context):
    """SÍNTESE APRIMORADA: Gera a resposta final para o usuário."""
    api_key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    has_complete_8_4 = "=== SEÇÃO COMPLETA DO ITEM 8.4" in context
    structure_instruction = ""
    if has_complete_8_4:
        structure_instruction = "ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4: Use a estrutura oficial do item 8.4 do Formulário de Referência (a, b, c, ...). Para cada subitem, extraia e organize as informações encontradas."
    else:
        structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown, priorizando informações de chunks com tags específicas."
        
    prompt = f"""
    Você é um analista financeiro sênior especializado em Formulários de Referência da CVM.
    PERGUNTA ORIGINAL DO USUÁRIO: "{query}"
    CONTEXTO COLETADO DOS DOCUMENTOS:
    {context}
    {structure_instruction}
    INSTRUÇÕES PARA O RELATÓRIO FINAL:
    1. Responda diretamente à pergunta do usuário com base no contexto.
    2. Seja detalhado, preciso e profissional.
    3. Transcreva dados importantes como valores, datas e percentuais.
    4. Se alguma informação não estiver disponível, indique: "Informação não encontrada nas fontes analisadas".
    RELATÓRIO ANALÍTICO FINAL:
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"ERRO ao gerar resposta final: {e}"

# --- LÓGICA PRINCIPAL DA APLICAÇÃO STREAMLIT ---

# Tenta carregar os recursos. Se falhar, a app para aqui com uma mensagem de erro.
try:
    loaded_artifacts, embedding_model, company_catalog = load_all_artifacts()
except Exception as e:
    st.error(f"Ocorreu um erro fatal durante o carregamento dos recursos: {e}")
    st.stop() # Interrompe a execução do script

# A aplicação só continua se os recursos foram carregados com sucesso
if loaded_artifacts:
    # Inicializa o histórico do chat no estado da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Qual empresa ou plano de incentivo você gostaria de analisar?"}]

    # Exibe as mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura o input do usuário
    if prompt := st.chat_input("Digite sua pergunta sobre CCR, Vibra, etc."):
        # Adiciona e exibe a mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processa a query e exibe a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Analisando documentos e gerando resposta..."):
                # 1. Gerar plano de análise
                plan_response = create_dynamic_analysis_plan(prompt, company_catalog, list(loaded_artifacts.keys()))
                
                if plan_response['status'] != 'success' or not plan_response['plan'].get("empresas"):
                    response_text = "Não consegui identificar uma empresa em sua pergunta ou houve um erro no planejamento. Por favor, seja mais específico."
                else:
                    plan = plan_response['plan']
                    
                    # 2. Executar plano para recuperar contexto
                    query_intent = 'item_8_4_query' if any(term in prompt.lower() for term in ['8.4', '8-4', 'item 8.4', 'formulário']) else 'general_query'
                    retrieved_context, sources = execute_dynamic_plan(plan, query_intent, loaded_artifacts, embedding_model)

                    if not retrieved_context.strip():
                        response_text = "Não encontrei informações relevantes nos documentos para a sua solicitação."
                    else:
                        # 3. Gerar resposta final com base no contexto
                        final_answer = get_final_unified_answer(prompt, retrieved_context)
                        response_text = final_answer

                        # Adiciona as fontes em um expansor se houver alguma
                        if sources:
                            unique_sources = sorted(list(set(sources)))
                            with st.expander(f"Fontes Consultadas ({len(unique_sources)})"):
                                for source in unique_sources:
                                    st.write(f"- {source}")
                
            st.markdown(response_text)
        
        # Adiciona a resposta completa do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": response_text})
