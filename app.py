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
from catalog_data import company_catalog_rich

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- DICIONÁRIOS E LISTAS DE TÓPICOS (sem alterações) ---
TERMOS_TECNICOS_LTIP = {
    "tratamento de dividendos": ["tratamento de dividendos", "equivalente em dividendos", "dividendos", "juros sobre capital próprio", "proventos", "dividend equivalent", "dividendos pagos em ações", "ajustes por dividendos"],
    "preço de exercício": ["preço de exercício", "strike price", "preço de compra", "preço fixo", "valor de exercício", "preço pré-estabelecido", "preço de aquisição"],
    "forma de liquidação": ["forma de liquidação", "liquidação", "pagamento", "entrega física", "pagamento em dinheiro", "transferência de ações", "settlement"],
    "vesting": ["vesting", "período de carência", "carência", "aquisição de direitos", "cronograma de vesting", "vesting schedule", "período de cliff"],
    "eventos corporativos": ["eventos corporativos", "desdobramento", "grupamento", "dividendos pagos em ações", "bonificação", "split", "ajustes", "reorganização societária"],
    "stock options": ["stock options", "opções de ações", "opções de compra", "SOP", "plano de opções", "ESOP", "opção de compra de ações"],
    "ações restritas": ["ações restritas", "restricted shares", "RSU", "restricted stock units", "ações com restrição", "plano de ações restritas"]
}
AVAILABLE_TOPICS = [
    "termos e condições gerais", "data de aprovação e órgão responsável", "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas", "condições de aquisição de ações", "critérios para fixação do preço de aquisição ou exercício", "preço de exercício", "strike price", "critérios para fixação do prazo de aquisição ou exercício", "forma de liquidação", "liquidação", "pagamento", "restrições à transferência das ações", "critérios e eventos de suspensão/extinção", "efeitos da saída do administrador", "Tipos de Planos", "Condições de Carência", "Vesting", "período de carência", "cronograma de vesting", "Matching", "contrapartida", "co-investimento", "Lockup", "período de lockup", "restrição de venda", "Tratamento de Dividendos", "equivalente em dividendos", "proventos", "Stock Options", "opções de ações", "SOP", "Ações Restritas", "RSU", "restricted shares", "Eventos Corporativos", "IPO", "grupamento", "desdobramento"
]

# --- FUNÇÕES AUXILIARES ---
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
            # Melhorando a busca para não pegar apenas a primeira palavra
            if company_name.split(' ')[0].lower() in document_path.lower():
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
    try:
        nfkd_form = unicodedata.normalize('NFKD', name.lower())
        name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        name = re.sub(r'[.,-]', '', name)
        suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', name).strip() # Manter espaços entre as palavras
    except Exception as e:
        return name.lower()

# --- CACHE PARA CARREGAR ARTEFATOS ---
@st.cache_resource
def load_all_artifacts():
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
    return artifacts, model

# --- FUNÇÕES PRINCIPAIS ---
# NOVO: Esta é a nova função de análise com lógica de scoring.
def create_dynamic_analysis_plan_v2(query, company_catalog_rich, available_indices):
    """
    Gera um plano de ação dinâmico com identificação de empresas baseada em scoring
    para maior precisão, lidando com nomes compostos, apelidos e variações.
    """
    api_key = GEMINI_API_KEY
    # CORREÇÃO: Atualizado o nome do modelo na URL para a versão mais recente e estável.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    # --- LÓGICA DE IDENTIFICAÇÃO DE EMPRESAS POR SCORING ---
    query_lower = query.lower().strip()
    normalized_query_for_scoring = normalize_name(query_lower)
    company_scores = {}
    for company_data in company_catalog_rich:
        canonical_name = company_data["canonical_name"]
        score = 0
        for alias in company_data.get("aliases", []):
            if alias in query_lower:
                score += 10 * len(alias.split())
        name_for_parts = normalize_name(canonical_name)
        parts = name_for_parts.split()
        for part in parts:
            if len(part) >= 2 and part in normalized_query_for_scoring.split():
                score += 1
        if score > 0:
            company_scores[canonical_name] = score
    mentioned_companies = []
    if company_scores:
        sorted_companies = sorted(company_scores.items(), key=lambda item: item[1], reverse=True)
        max_score = sorted_companies[0][1]
        if max_score > 0:
            mentioned_companies = [company for company, score in sorted_companies if score >= max_score * 0.7]

    # --- FIM DA LÓGICA DE IDENTIFICAÇÃO ---
    prompt = f'Você é um consultor de incentivos de longo prazo semelhante a global shares. Sua tarefa é analisar a "Pergunta do Usuário" que serão sempre no contexto de programas de incentivo de longo prazo, remuneração ou item 8 do formulário de referencia  e identificar os tópicos de interesse relacionados a programas de incentivo de longo prazo. Instruções: 1. Identifique os Tópicos: Analise a pergunta para identificar os tópicos de interesse. Se a pergunta for genérica (ex: "resumo dos planos", "análise da empresa"), inclua todos os "Tópicos de Análise Disponíveis". Se for específica (ex: "fale sobre o vesting e dividendos"), inclua apenas os tópicos relevantes. 2. Formate a Saída: Retorne APENAS uma lista JSON de strings contendo os tópicos identificados. Tópicos de Análise Disponíveis: {json.dumps(AVAILABLE_TOPICS, indent=2)} Pergunta do Usuário: "{query}" Tópicos de Interesse (responda APENAS com a lista JSON de strings):'
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        topics = json.loads(json_match.group(0)) if json_match else AVAILABLE_TOPICS
    except Exception:
        topics = AVAILABLE_TOPICS
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}


# Mantenha as constantes no topo do seu script
MAX_CONTEXT_TOKENS = 12000
MAX_CHUNKS_PER_TOPIC = 5

def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """Executa o plano de busca e recupera contexto relevante."""
    full_context = ""
    all_retrieved_docs = set()
    
    if query_intent == 'item_8_4_query':
        # Busca exaustiva no item 8.4
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            if 'item_8_4' in artifacts:
                artifact_data = artifacts['item_8_4']
                chunk_data = artifact_data['chunks']
                
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
                
                full_context += f"=== SEÇÃO COMPLETA DO ITEM 8.4 - {empresa.upper()} ===\n\n"
                for chunk_info in empresa_chunks_8_4:
                    full_context += f"--- Chunk Item 8.4 (Doc: {chunk_info['path']}) ---\n"
                    full_context += f"{chunk_info['text']}\n\n"
                
                full_context += f"=== FIM DA SEÇÃO ITEM 8.4 - {empresa.upper()} ===\n\n"
            
            # Busca complementar
            complementary_indices = [idx for idx in artifacts.keys() if idx != 'item_8_4']
            
            for topico in plan.get("topicos", [])[:10]:
                expanded_terms = expand_search_terms(topico)
                
                for term in expanded_terms[:5]:
                    search_query = f"item 8.4 {term} empresa {empresa}"
                    
                    for index_name in complementary_indices:
                        if index_name in artifacts:
                            artifact_data = artifacts[index_name]
                            index = artifact_data['index']
                            chunk_data = artifact_data['chunks']
                            
                            try:
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
                                    break
                            except Exception as e:
                                st.error(f"Erro na busca semântica: {e}")
                                continue
                    
                    if chunks_found > 0:
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    
    else:
        # Busca geral com tags e expansão de termos
        for empresa in plan.get("empresas", []):
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            # Busca por tags específicas
            target_tags = []
            for topico in plan.get("topicos", []):
                expanded_terms = expand_search_terms(topico)
                target_tags.extend(expanded_terms)
            
            target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
            
            tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
            
            if tagged_chunks:
                full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
                for chunk_info in tagged_chunks:
                    full_context += f"--- Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']}) ---\n"
                    full_context += f"{chunk_info['text']}\n\n"
                    all_retrieved_docs.add(str(chunk_info['path']))
                full_context += f"=== FIM DOS CHUNKS COM TAGS - {empresa.upper()} ===\n\n"
            
            # Busca semântica complementar
            indices_to_search = list(artifacts.keys())
            
            for topico in plan.get("topicos", []):
                expanded_terms = expand_search_terms(topico)
                
                for term in expanded_terms[:3]:
                    search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                    
                    chunks_found = 0
                    for index_name in indices_to_search:
                        if index_name in artifacts:
                            artifact_data = artifacts[index_name]
                            index = artifact_data['index']
                            chunk_data = artifact_data['chunks']
                            
                            try:
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
                                    break
                            except Exception as e:
                                st.error(f"Erro na busca semântica: {e}")
                                continue
                    
                    if chunks_found > 0:
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    
    return full_context, [str(doc) for doc in all_retrieved_docs]


# MANTENDO A FUNÇÃO DE GERAÇÃO DE RESPOSTA ORIGINAL
# MANTENDO A FUNÇÃO DE GERAÇÃO DE RESPOSTA ORIGINAL
def get_final_unified_answer(query, context):
    """Gera a resposta final usando o contexto recuperado."""
    # CORREÇÃO: Atualizado o nome do modelo na URL para a versão mais recente e estável.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    
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
        
    prompt = f'Você é um analista financeiro sênior especializado em Formulários de Referência da CVM. PERGUNTA ORIGINAL DO USUÁRIO: "{query}" CONTEXTO COLETADO DOS DOCUMENTOS: {context} {structure_instruction} INSTRUÇÕES PARA O RELATÓRIO FINAL: 1. Responda diretamente à pergunta do usuário. 2. PRIORIZE informações da SEÇÃO COMPLETA DO ITEM 8.4 ou de CHUNKS COM TAGS ESPECÍFICAS quando disponíveis. 3. Use informações complementares apenas para esclarecer. 4. Seja detalhado, preciso e profissional. 5. Se alguma informação não estiver disponível, indique: "Informação não encontrada nas fontes analisadas". RELATÓRIO ANALÍTICO FINAL:'
    
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"ERRO ao gerar resposta final: {e}"
        


# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
    st.title("🤖 Agente de Análise de Planos de Incentivo Longo Prazo ILP")
    st.markdown("---")

    with st.spinner("Inicializando sistema..."):
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
        st.success("✅ Sistema carregado")
        st.info(f"Modelo: {MODEL_NAME}")

    st.header("💬 Faça sua pergunta")

    with st.expander("💡 Entenda como funciona e veja dicas para perguntas ideais"):
        st.markdown("""
        **Este agente analisa Planos de Incentivo de Longo Prazo (ILPs) usando documentos públicos das empresas listadas.**

        ### Formatos de Pergunta Recomendados

        **1. Perguntas Específicas** *(formato ideal)*
        Combine tópicos + empresas para análises direcionadas:
        - *"Qual a liquidação e dividendos da **Vale**?"*
        - *"Vesting da **Petrobras**"*
        - *"Ajustes de preço da **Ambev**"*
        - *"Período de lockup da **Magalu**"*
        - *"Condições de carência **YDUQS**"*

        **2. Visão Geral (Item 8.4)**
        Solicite a seção completa do Formulário de Referência:
        - *"Item 8.4 da **Vibra**"*
        - *"Resumo 8.4 da **Raia Drogasil**"*
        - *"Formulário completo da **WEG**"*

        **3. Análise Comparativa**
        Compare características entre empresas:
        - *"Liquidação **Localiza** vs **Movida**"*
        - *"Dividendos **Eletrobras** vs **Energisa**"*
        - *"Matching **Natura** vs **Gerdau**"*
        """)

    user_query = st.text_area("Digite sua pergunta:", height=100, placeholder="Ex: Compare o vesting da Vale com a Petrobras")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return

        with st.container():
            st.markdown("---")
            st.subheader("📋 Processo de Análise")

            # --- ETAPA 1: GERAÇÃO DO PLANO ---
            with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
                plan_response = create_dynamic_analysis_plan_v2(user_query, company_catalog_rich, list(loaded_artifacts.keys()))
                plan = plan_response['plan']
                empresas = plan.get('empresas', [])

                if not empresas:
                    st.error("❌ Não consegui identificar empresas na sua pergunta. Tente usar nomes, apelidos ou marcas conhecidas (ex: Magalu, Vivo, Itaú).")
                    return

                st.write(f"**🏢 Empresas identificadas:** {', '.join(empresas)}")
                st.write(f"**📝 Tópicos a analisar:** {len(plan.get('topicos', []))}")
                status.update(label="✅ Plano gerado com sucesso!", state="complete")

            # --- ETAPA 2: LÓGICA DE EXECUÇÃO (com tratamento para comparações) ---
            final_answer = ""
            sources = set()

            # --- MODO COMPARATIVO: Se mais de uma empresa for identificada ---
            if len(empresas) > 1:
                st.info(f"Modo de comparação ativado para {len(empresas)} empresas. Analisando sequencialmente...")
                summaries = []
                for i, empresa in enumerate(empresas):
                    with st.status(f"Analisando {i+1}/{len(empresas)}: {empresa}...", expanded=True):
                        single_company_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
                        query_intent = 'item_8_4_query' if any(term in user_query.lower() for term in ['8.4', 'formulário']) else 'general_query'
                        
                        retrieved_context, retrieved_sources = execute_dynamic_plan(single_company_plan, query_intent, loaded_artifacts, embedding_model)
                        sources.update(retrieved_sources)

                        if "Nenhuma informação" in retrieved_context:
                            summary = f"## Análise para {empresa}\n\nNenhuma informação encontrada nos documentos para os tópicos solicitados."
                        else:
                            # Reutiliza a função get_final_answer para criar um resumo para esta empresa
                            summary_prompt = f"Com base no contexto a seguir sobre a empresa {empresa}, resuma os pontos principais sobre os seguintes tópicos: {', '.join(plan['topicos'])}. Contexto: {retrieved_context}"
                            summary = get_final_unified_answer(summary_prompt, retrieved_context)
                        
                        summaries.append(f"--- RESUMO PARA {empresa.upper()} ---\n\n{summary}")

                # Etapa final de comparação
                with st.status("Gerando relatório comparativo final...", expanded=True):
                    comparison_prompt = f"""Com base nos resumos individuais a seguir, crie um relatório comparativo detalhado e bem estruturado entre as empresas, focando nos pontos levantados na pergunta original do usuário.

Pergunta original do usuário: '{user_query}'

{chr(10).join(summaries)}

Relatório Comparativo Final:"""
                    # Usa o contexto dos resumos para a chamada final
                    final_answer = get_final_unified_answer(comparison_prompt, "\n\n".join(summaries))

            # --- MODO DE ANÁLISE ÚNICA: Se apenas uma empresa for identificada ---
            else:
                with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
                    query_intent = 'item_8_4_query' if any(term in user_query.lower() for term in ['8.4', 'formulário']) else 'general_query'
                    st.write(f"**🎯 Estratégia detectada:** {'Item 8.4 completo' if query_intent == 'item_8_4_query' else 'Busca geral'}")
                    
                    retrieved_context, retrieved_sources = execute_dynamic_plan(plan, query_intent, loaded_artifacts, embedding_model)
                    sources.update(retrieved_sources)
                    
                    if not retrieved_context.strip() or "Nenhuma informação encontrada" in retrieved_context:
                        st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                        return
                    
                    st.write(f"**📄 Contexto recuperado de:** {len(sources)} documento(s)")
                    status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
                
                with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
                    final_answer = get_final_unified_answer(user_query, retrieved_context)
                    status.update(label="✅ Análise concluída!", state="complete")

            # --- ETAPA 3: EXIBIÇÃO DO RESULTADO ---
            st.markdown("---")
            st.subheader("📄 Resultado da Análise")
            with st.container():
                st.markdown(final_answer)

            # Fontes consultadas
            if sources:
                st.markdown("---")
                with st.expander(f"📚 Documentos consultados ({len(sources)})", expanded=False):
                    unique_sources = sorted(list(sources))
                    for i, source in enumerate(unique_sources, 1):
                        st.write(f"{i}. {source}")

if __name__ == "__main__":
    main()
