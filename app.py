# app.py
# -*- coding: utf-8 -*-
"""
AGENTE COM PLANEJAMENTO DINÂMICO - VERSÃO STREAMLIT COMPLETA
Mantém a lógica robusta do código original com interface Web
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

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="🚀 Agente de Análise LTIP",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURAÇÕES E ESTRUTURAS DE CONHECIMENTO (DO CÓDIGO ORIGINAL) ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
AMBIGUITY_THRESHOLD = 3

# Dicionário especializado para termos técnicos de LTIP (MANTIDO ORIGINAL)
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

# Tópicos expandidos de análise (MANTIDO ORIGINAL)
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

# --- FUNÇÃO SEGURA PARA CHAMADAS DE API (CONSERVANDO PROTEÇÃO) ---
def safe_api_call(url, payload, headers, timeout=90):
    """Função segura para chamadas de API sem expor a chave em erros."""
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

# --- FUNÇÕES PRINCIPAIS (MANTIDAS DO CÓDIGO ORIGINAL) ---

def expand_search_terms(base_term):
    """Expande um termo base com sinônimos e variações técnicas."""
    expanded_terms = [base_term.lower()]
    
    for category, terms in TERMOS_TECNICOS_LTIP.items():
        if any(term.lower() in base_term.lower() for term in terms):
            expanded_terms.extend([term.lower() for term in terms])
    
    return list(set(expanded_terms))

def search_by_tags(artifacts, company_name, target_tags):
    """Busca chunks que contenham tags específicas para uma empresa. (MANTIDO ORIGINAL)"""
    results = []
    
    for index_name, artifact_data in artifacts.items():
        chunk_data = artifact_data['chunks']
        
        for i, mapping in enumerate(chunk_data.get('map', [])):
            document_path = mapping['document_path']
            if re.search(re.escape(company_name.split(' ')[0]), document_path, re.IGNORECASE):
                chunk_text = chunk_data["chunks"][i]
                
                # Verifica se o chunk contém as tags procuradas
                for tag in target_tags:
                    if f"Tópicos:" in chunk_text and tag in chunk_text:
                        results.append({
                            'text': chunk_text,
                            'path': document_path,
                            'index': i,
                            'source': index_name,
                            'tag_found': tag
                        })
                        st.write(f"     -> Encontrado chunk com tag '{tag}' em {document_path}")
                        break
    
    return results

@st.cache_resource
def load_all_artifacts():
    """Carrega todos os artefatos e constrói um catálogo de nomes de empresas canônicos. (CONSERVANDO LÓGICA ORIGINAL)"""
    artifacts = {}
    canonical_company_names = set()
    
    # CONSERVA O ACESSO AOS ARQUIVOS FAISS
    google_drive_path = st.session_state.get('google_drive_path', './data')
    
    if not os.path.exists(google_drive_path):
        st.error(f"ERRO CRÍTICO: Pasta não encontrada: {google_drive_path}")
        st.info("Por favor, configure o caminho correto dos arquivos FAISS na barra lateral.")
        return None, None, None
    
    st.info("--- Carregando múltiplos artefatos ---")
    
    with st.spinner(f"Carregando modelo de embedding '{MODEL_NAME}'..."):
        model = SentenceTransformer(MODEL_NAME)

    index_files = glob.glob(os.path.join(google_drive_path, '*_faiss_index.bin'))
    if not index_files:
        st.error(f"ERRO CRÍTICO: Nenhum arquivo de índice (*_faiss_index.bin) encontrado em '{google_drive_path}'.")
        st.info("Certifique-se de que os arquivos FAISS estão na pasta correta.")
        return None, None, None

    progress_bar = st.progress(0)
    total_files = len(index_files)
    
    for idx, index_file in enumerate(index_files):
        category = os.path.basename(index_file).replace('_faiss_index.bin', '')
        chunks_file = os.path.join(google_drive_path, f"{category}_chunks_map.json")
        
        try:
            st.info(f"Carregando categoria de documento '{category}'...")
            index = faiss.read_index(index_file)
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            artifacts[category] = {'index': index, 'chunks': chunk_data}
            
            # Extrai nomes das empresas
            for mapping in chunk_data.get('map', []):
                company_name = mapping['document_path'].split('/')[0]
                canonical_company_names.add(company_name)
            
            progress_bar.progress((idx + 1) / total_files)
            
        except FileNotFoundError:
            st.warning(f"AVISO: Arquivo de chunks '{chunks_file}' não encontrado para o índice '{index_file}'. Pulando.")
            continue
        except Exception as e:
            st.error(f"Erro ao carregar '{category}': {e}")
            continue

    if not artifacts:
        st.error("ERRO CRÍTICO: Nenhum artefato foi carregado com sucesso.")
        return None, None, None

    st.success(f"--- {len(artifacts)} categorias de documentos carregadas com sucesso! ---")
    st.success(f"--- {len(canonical_company_names)} empresas únicas identificadas. ---")
    
    return artifacts, model, list(canonical_company_names)

def create_dynamic_analysis_plan(query, company_catalog, available_indices):
    """CHAMADA AO LLM PLANEJADOR: Gera um plano de ação dinâmico em JSON - CONSERVANDO LÓGICA ORIGINAL."""
    
    # CONSERVA O ACESSO SEGURO À API KEY
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = st.session_state.get('gemini_api_key')
        if not api_key:
            st.error("ERRO: API Key do Gemini não configurada!")
            return {"status": "error", "message": "API Key não encontrada"}
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Função de normalização melhorada (MANTIDA ORIGINAL)
    def normalize_name(name):
        try:
            nfkd_form = unicodedata.normalize('NFKD', name.lower())
            name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
            name = re.sub(r'[.,-]', '', name)
            suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
            for suffix in suffixes:
                name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            return re.sub(r'\s+', '', name).strip()
        except Exception as e:
            st.error(f"Erro na normalização: {e}")
            return name.lower()

    # IDENTIFICAÇÃO ROBUSTA DE EMPRESAS COM BUSCA HIERÁRQUICA (MANTIDA ORIGINAL)
    mentioned_companies = []
    query_clean = query.lower().strip()
    
    st.write(f"   -> Buscando empresas na query: '{query_clean}'")
    
    for canonical_name in company_catalog:
        found = False
        
        # 1. BUSCA EXATA
        if canonical_name.lower() == query_clean:
            mentioned_companies.append(canonical_name)
            st.write(f"   -> Correspondência EXATA: {canonical_name}")
            found = True
            continue
        
        # 2. BUSCA POR SUBSTRING (nome completo na query)
        if canonical_name.lower() in query_clean:
            mentioned_companies.append(canonical_name)
            st.write(f"   -> Correspondência por SUBSTRING: {canonical_name}")
            found = True
            continue
        
        # 3. BUSCA POR PARTES DO NOME
        company_parts = canonical_name.split(' ')
        for part in company_parts:
            if len(part) > 2 and re.search(r'\b' + re.escape(part.lower()) + r'\b', query_clean):
                if canonical_name not in mentioned_companies:
                    mentioned_companies.append(canonical_name)
                    st.write(f"   -> Correspondência por PARTE: {canonical_name} (parte: {part})")
                    found = True
                break
        
        if found:
            continue
        
        # 4. BUSCA SIMPLIFICADA (último recurso)
        normalized_canonical = normalize_name(canonical_name)
        normalized_query = normalize_name(query_clean)
        
        if normalized_query and len(normalized_query) > 2:
            if normalized_query in normalized_canonical:
                mentioned_companies.append(canonical_name)
                st.write(f"   -> Correspondência SIMPLIFICADA: {canonical_name}")
    
    st.write(f"   -> Empresas identificadas: {mentioned_companies}")
    
    # Se não encontrou empresas, tenta busca mais agressiva
    if not mentioned_companies:
        st.write("   -> Tentando busca mais agressiva...")
        for canonical_name in company_catalog:
            # Busca por siglas (ex: CCR, Vibra)
            if len(query_clean) <= 6:  # Provável sigla ou nome curto
                if query_clean.upper() in canonical_name.upper():
                    mentioned_companies.append(canonical_name)
                    st.write(f"   -> Correspondência por SIGLA: {canonical_name}")

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
    
    # USA A FUNÇÃO SEGURA PARA CHAMADAS DE API
    response_data, error_message = safe_api_call(url, payload, headers, timeout=90)
    
    if error_message:
        st.warning(f"Erro no planejamento: {error_message}")
        # Fallback em caso de erro
        plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
        return {"status": "success", "plan": plan}
    
    try:
        text_response = response_data['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            topics = json.loads(json_match.group(0))
            plan = {"empresas": mentioned_companies, "topicos": topics}
            return {"status": "success", "plan": plan}
        else:
            # Fallback: usa todos os tópicos se não conseguir extrair JSON
            plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
            return {"status": "success", "plan": plan}
    except Exception as e:
        st.error(f"Erro ao processar resposta da IA: {e}")
        # Fallback em caso de erro
        plan = {"empresas": mentioned_companies, "topicos": AVAILABLE_TOPICS}
        return {"status": "success", "plan": plan}

def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """EXECUTOR APRIMORADO: Busca exaustiva no item 8.4 + busca por tags + expansão de termos. (MANTIDO ORIGINAL)"""
    full_context = ""
    all_retrieved_docs = set()
    
    if query_intent == 'item_8_4_query':
        st.write("-> Estratégia: BUSCA EXAUSTIVA ITEM 8.4 + COMPLEMENTO")
        
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            # FASE 1: BUSCA EXAUSTIVA NO ITEM 8.4
            if 'item_8_4' in artifacts:
                st.write(f"   -> FASE 1: Recuperação EXAUSTIVA do Item 8.4 para {empresa}")
                
                artifact_data = artifacts['item_8_4']
                chunk_data = artifact_data['chunks']
                
                # Recupera TODOS os chunks da empresa no item 8.4
                empresa_chunks_8_4 = []
                for i, mapping in enumerate(chunk_data.get('map', [])):
                    document_path = mapping['document_path']
                    if re.search(re.escape(empresa.split(' ')[0]), document_path, re.IGNORECASE):
                        chunk_text = chunk_data["chunks"][i]
                        all_retrieved_docs.add(str(document_path))
                        empresa_chunks_8_4.append({
                            'text': chunk_text,
                            'path': document_path,
                            'index': i
                        })
                
                st.write(f"     -> Encontrados {len(empresa_chunks_8_4)} chunks do Item 8.4")
                
                # Adiciona TODOS os chunks do item 8.4 ao contexto
                full_context += f"=== SEÇÃO COMPLETA DO ITEM 8.4 - {empresa.upper()} ===\n\n"
                for chunk_info in empresa_chunks_8_4:
                    full_context += f"--- Chunk Item 8.4 (Doc: {chunk_info['path']}) ---\n"
                    full_context += f"{chunk_info['text']}\n\n"
                
                full_context += f"=== FIM DA SEÇÃO ITEM 8.4 - {empresa.upper()} ===\n\n"
            
            # FASE 2: BUSCA COMPLEMENTAR COM EXPANSÃO DE TERMOS (limitada)
            st.write(f"   -> FASE 2: Busca complementar com expansão de termos para {empresa}")
            
            complementary_indices = [idx for idx in artifacts.keys() if idx != 'item_8_4']
            
            for topico in plan.get("topicos", [])[:10]:  # Limita a 10 tópicos
                expanded_terms = expand_search_terms(topico)
                
                st.write(f"     -> Buscando complemento para '{topico}'...")
                
                for term in expanded_terms[:5]:  # Limita a 5 termos
                    search_query = f"item 8.4 {term} empresa {empresa}"
                    
                    for index_name in complementary_indices:
                        if index_name in artifacts:
                            artifact_data = artifacts[index_name]
                            index = artifact_data['index']
                            chunk_data = artifact_data['chunks']
                            
                            query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                            scores, indices = index.search(query_embedding, 3)
                            
                            chunks_found = 0
                            for i, idx in enumerate(indices[0]):
                                if idx != -1 and idx < len(chunk_data.get("chunks", [])) and scores[0][i] > 0.5:
                                    document_path = chunk_data["map"][idx]['document_path']
                                    if re.search(re.escape(empresa.split(' ')[0]), document_path, re.IGNORECASE):
                                        chunk_text = chunk_data["chunks"][idx]
                                        
                                        chunk_hash = hash(chunk_text[:100])
                                        if chunk_hash not in all_retrieved_docs:
                                            all_retrieved_docs.add(chunk_hash)
                                            score = scores[0][i]
                                            full_context += f"--- Contexto COMPLEMENTAR para '{topico}' via '{term}' (Fonte: {index_name}, Score: {score:.3f}) ---\n{chunk_text}\n\n"
                                            chunks_found += 1
                            
                            if chunks_found > 0:
                                st.write(f"         -> {chunks_found} chunks complementares de '{index_name}' para '{term}'")
                                break
                    
                    if chunks_found > 0:
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    
    else:
        # BUSCA GERAL COM TAGS E EXPANSÃO DE TERMOS
        st.write("-> Estratégia: BUSCA GERAL COM TAGS E EXPANSÃO DE TERMOS")
        
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            # FASE 1: BUSCA POR TAGS ESPECÍFICAS
            st.write(f"   -> FASE 1: Busca por tags específicas para {empresa}")
            
            target_tags = []
            for topico in plan.get("topicos", []):
                expanded_terms = expand_search_terms(topico)
                target_tags.extend(expanded_terms)
            
            # Remove duplicatas e mantém apenas termos relevantes
            target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
            
            st.write(f"     -> Tags procuradas: {target_tags[:5]}...")
            
            tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
            
            if tagged_chunks:
                full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
                for chunk_info in tagged_chunks:
                    full_context += f"--- Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']}) ---\n"
                    full_context += f"{chunk_info['text']}\n\n"
                    all_retrieved_docs.add(str(chunk_info['path']))
                full_context += f"=== FIM DOS CHUNKS COM TAGS - {empresa.upper()} ===\n\n"
            
            # FASE 2: BUSCA SEMÂNTICA COMPLEMENTAR
            st.write(f"   -> FASE 2: Busca semântica complementar para {empresa}")
            
            indices_to_search = list(artifacts.keys())
            
            for topico in plan.get("topicos", []):
                expanded_terms = expand_search_terms(topico)
                
                st.write(f"     -> Buscando '{topico}'...")
                
                for term in expanded_terms[:3]:  # Top 3 termos
                    search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                    
                    chunks_found = 0
                    for index_name in indices_to_search:
                        if index_name in artifacts:
                            artifact_data = artifacts[index_name]
                            index = artifact_data['index']
                            chunk_data = artifact_data['chunks']
                            
                            query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                            scores, indices = index.search(query_embedding, TOP_K_SEARCH)
                            
                            for i, idx in enumerate(indices[0]):
                                if idx != -1 and scores[0][i] > 0.4:
                                    document_path = chunk_data["map"][idx]['document_path']
                                    if re.search(re.escape(empresa.split(' ')[0]), document_path, re.IGNORECASE):
                                        chunk_text = chunk_data["chunks"][idx]
                                        
                                        chunk_hash = hash(chunk_text[:100])
                                        if chunk_hash not in all_retrieved_docs:
                                            all_retrieved_docs.add(chunk_hash)
                                            score = scores[0][i]
                                            full_context += f"--- Contexto para '{topico}' via '{term}' (Fonte: {index_name}, Score: {score:.3f}) ---\n{chunk_text}\n\n"
                                            chunks_found += 1
                    
                    if chunks_found > 0:
                        st.write(f"       -> {chunks_found} chunks encontrados para '{term}'")
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    
    # CORREÇÃO: Converte todos os elementos para string
    return full_context, [str(doc) for doc in all_retrieved_docs]

def get_final_unified_answer(query, context):
    """SÍNTESE APRIMORADA: Processa contexto exaustivo do item 8.4. (CONSERVANDO LÓGICA ORIGINAL)"""
    
    # CONSERVA O ACESSO SEGURO À API KEY
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = st.session_state.get('gemini_api_key')
        if not api_key:
            return "ERRO: API Key do Gemini não configurada para gerar resposta final!"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Detecta se há seção completa do item 8.4
    has_complete_8_4 = "=== SEÇÃO COMPLETA DO ITEM 8.4" in context
    has_tagged_chunks = "=== CHUNKS COM TAGS ESPECÍFICAS" in context
    
    if has_complete_8_4:
        structure_instruction = """
**ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4:**
Use a estrutura oficial do item 8.4 do Formulário de Referência:
a) Termos e condições gerais dos planos
b) Data de aprovação e órgão responsável
c) Número máximo de ações abrangidas pelos planos
d) Número máximo de opções a serem outorgadas
e) Condições de aquisição de ações
f) Critérios para fixação do preço de aquisição ou exercício
g) Critérios para fixação do prazo de aquisição ou exercício
h) Forma de liquidação
i) Restrições à transferência das ações
j) Critérios e eventos que, quando verificados, ocasionarão a suspensão, alteração ou extinção do plano
k) Efeitos da saída do administrador do cargo na manutenção dos seus direitos no plano

Para cada subitem, extraia e organize as informações encontradas na SEÇÃO COMPLETA DO ITEM 8.4.
"""
    elif has_tagged_chunks:
        structure_instruction = "**PRIORIZE** as informações dos CHUNKS COM TAGS ESPECÍFICAS e organize a resposta de forma lógica usando Markdown."
    else:
        structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
    
    prompt = f"""
Você é um analista financeiro sênior especializado em Formulários de Referência da CVM. 

**PERGUNTA ORIGINAL DO USUÁRIO:**
"{query}"

**CONTEXTO COLETADO DOS DOCUMENTOS:**
{context}

{structure_instruction}

**INSTRUÇÕES PARA O RELATÓRIO FINAL:**
1. Responda diretamente à pergunta do usuário
2. **PRIORIZE** as informações da SEÇÃO COMPLETA DO ITEM 8.4 quando disponível
3. **PRIORIZE** as informações dos CHUNKS COM TAGS ESPECÍFICAS quando disponível
4. Use informações complementares apenas para esclarecer ou expandir pontos específicos
5. Seja detalhado, preciso e profissional
6. Transcreva dados importantes como valores, datas e percentuais
7. Se alguma informação não estiver disponível, indique: "Informação não encontrada nas fontes analisadas"
8. Mantenha a estrutura técnica apropriada para administradores de LTIP

**RELATÓRIO ANALÍTICO FINAL:**
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
    }
    headers = {'Content-Type': 'application/json'}
    
    # USA A FUNÇÃO SEGURA PARA CHAMADAS DE API
    response_data, error_message = safe_api_call(url, payload, headers, timeout=180)
    
    if error_message:
        return f"ERRO ao gerar resposta final: {error_message}"
    
    try:
        return response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"ERRO ao processar resposta final: {e}"

# --- INTERFACE STREAMLIT ---

def main():
    st.title("🚀 Agente com Planejamento Dinâmico - LTIP")
    st.markdown("**Análise inteligente de Formulários de Referência da CVM**")
    
    # --- SIDEBAR - CONFIGURAÇÕES ---
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # CONSERVA O ACESSO À API KEY
        st.subheader("🔑 API Key do Gemini")
        gemini_api_key = st.text_input(
            "Insira sua API Key do Google Gemini",
            type="password",
            help="Sua chave será mantida segura e não será exposta em erros."
        )
        if gemini_api_key:
            st.session_state['gemini_api_key'] = gemini_api_key
            st.success("✅ API Key configurada com segurança!")
        
        st.divider()
        
        # CONSERVA O ACESSO AOS ARQUIVOS FAISS
        st.subheader("📁 Arquivos FAISS")
        google_drive_path = st.text_input(
            "Caminho dos arquivos FAISS",
            value="./data",
            help="Pasta onde estão os arquivos *_faiss_index.bin e *_chunks_map.json"
        )
        if google_drive_path:
            st.session_state['google_drive_path'] = google_drive_path
            
        if os.path.exists(google_drive_path):
            faiss_files = glob.glob(os.path.join(google_drive_path, '*_faiss_index.bin'))
            st.info(f"✅ {len(faiss_files)} arquivos FAISS encontrados")
        else:
            st.error(f"❌ Pasta não encontrada: {google_drive_path}")
        
        st.divider()
        
        # Botão para recarregar
        if st.button("🔄 Recarregar Artefatos", type="primary"):
            # Limpa cache
            if 'loaded_artifacts' in st.session_state:
                del st.session_state['loaded_artifacts']
            if 'embedding_model' in st.session_state:
                del st.session_state['embedding_model']
            if 'company_catalog' in st.session_state:
                del st.session_state['company_catalog']
            st.rerun()

    # --- CARREGAMENTO DOS ARTEFATOS ---
    if 'loaded_artifacts' not in st.session_state:
        if not st.session_state.get('gemini_api_key') and 'GEMINI_API_KEY' not in st.secrets:
            st.warning("⚠️ Por favor, configure a API Key do Gemini na barra lateral ou nos secrets do Streamlit.")
            return
            
        with st.spinner("Carregando artefatos pela primeira vez..."):
            artifacts, model, company_catalog = load_all_artifacts()
        
        if artifacts is None:
            st.error("❌ Falha no carregamento dos artefatos. Verifique as configurações.")
            return
        
        st.session_state['loaded_artifacts'] = artifacts
        st.session_state['embedding_model'] = model
        st.session_state['company_catalog'] = company_catalog

    # --- INTERFACE PRINCIPAL ---
    loaded_artifacts = st.session_state['loaded_artifacts']
    embedding_model = st.session_state['embedding_model']
    company_catalog = st.session_state['company_catalog']
    
    # Exibe informações dos artefatos carregados
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Categorias de Documentos", len(loaded_artifacts))
    with col2:
        st.metric("🏢 Empresas Identificadas", len(company_catalog))
    with col3:
        st.metric("🤖 Modelo Embedding", MODEL_NAME.split('/')[-1])
    
    with st.expander("📋 Ver Empresas no Catálogo"):
        st.write(sorted(company_catalog))
    
    # Exemplos de perguntas
    st.subheader("💡 Exemplos de Perguntas")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Item 8.4 da Vibra", help="Consulta completa do item 8.4"):
            st.session_state['query_exemplo'] = "Descreva detalhadamente o item 8.4 do formulário de referência da Vibra"
    
    with col2:
        if st.button("⚡ Vesting da CCR", help="Consulta específica sobre vesting"):
            st.session_state['query_exemplo'] = "Como funciona o vesting nos planos de ações restritas da CCR?"
    
    with col3:
        if st.button("💰 Liquidação Vale", help="Consulta sobre forma de liquidação"):
            st.session_state['query_exemplo'] = "Qual a forma de liquidação dos planos da Vale?"
    
    st.divider()
    
    # --- ÁREA DE CONSULTA ---
    st.subheader("💬 Faça sua Pergunta")
    
    user_query = st.text_area(
        "Digite sua pergunta:",
        value=st.session_state.get('query_exemplo', ''),
        height=100,
        placeholder="Ex: Descreva a forma de liquidação das ações restritas na Vibra"
    )
    
    # Limpa query_exemplo após usar
    if 'query_exemplo' in st.session_state:
        del st.session_state['query_exemplo']
    
    if st.button("🔍 Analisar", type="primary", disabled=not user_query.strip()):
        if not user_query.strip():
            st.warning("Por favor, digite uma pergunta.")
            return
        
        try:
            # ETAPA 1: Geração do plano
            st.header("📋 Plano de Análise Dinâmico")
            with st.expander("Ver detalhes do planejamento", expanded=True):
                st.write("🔍 Processando pergunta...")
                
                plan_response = create_dynamic_analysis_plan(
                    user_query, 
                    company_catalog, 
                    list(loaded_artifacts.keys())
                )
                
                if plan_response['status'] != 'success':
                    st.error("❌ Erro ao gerar plano de análise")
                    return
                
                plan = plan_response['plan']
                
                # Exibe o plano gerado
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🏢 Empresas Identificadas:**")
                    for empresa in plan.get('empresas', []):
                        st.write(f"- {empresa}")
                
                with col2:
                    st.write("**📝 Tópicos de Análise:**")
                    for i, topico in enumerate(plan.get('topicos', [])[:10]):
                        st.write(f"{i+1}. {topico}")
                    if len(plan.get('topicos', [])) > 10:
                        st.write(f"... e mais {len(plan.get('topicos', [])) - 10} tópicos")
                
                if not plan.get("empresas"):
                    st.error("❌ Não consegui identificar empresas na sua pergunta. Seja mais específico.")
                    return
            
            # Detecta intenção da query (MANTIDO ORIGINAL)
            query_intent = 'item_8_4_query' if ('8.4' in user_query.lower() or '8-4' in user_query.lower() or 
                                               'item 8.4' in user_query.lower() or 'formulário' in user_query.lower()) else 'general_query'
            
            st.info(f"**Estratégia detectada:** {query_intent}")
            
            # ETAPA 2: Execução do plano
            st.header("🔍 Recuperação de Contexto")
            with st.expander("Ver detalhes da busca", expanded=True):
                retrieved_context, sources = execute_dynamic_plan(
                    plan, query_intent, loaded_artifacts, embedding_model
                )
                
                if not retrieved_context.strip():
                    st.warning("⚠️ Não encontrei informações relevantes nos documentos.")
                    return
                
                st.success(f"✅ Contexto recuperado de {len(set(sources))} documento(s)")
            
            # ETAPA 3: Geração da resposta final
            st.header("📄 Resposta Analítica")
            
            with st.spinner("Gerando resposta final..."):
                final_answer = get_final_unified_answer(user_query, retrieved_context)
            
            # Exibe a resposta
            st.markdown(final_answer)
            
            # Exibe fontes consultadas
            st.divider()
            with st.expander(f"📚 Documentos Consultados ({len(set(sources))})"):
                try:
                    unique_sources = sorted(list(set([str(source) for source in sources])))
                    for source in unique_sources:
                        st.write(f"- {source}")
                except Exception as e:
                    st.error(f"Erro ao processar fontes: {e}")
                    if sources:
                        st.write(f"Fontes: {list(set(sources))}")
        
        except Exception as e:
            st.error(f"❌ Erro durante a execução: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
