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
import logging
from functools import lru_cache
from catalog_data import company_catalog_rich

# --- CONFIGURAÇÕES ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- DICIONÁRIOS E LISTAS DE TÓPICOS (sem alterações) ---
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
# --- FUNÇÕES PRINCIPAIS ---
def create_dynamic_analysis_plan_v2(query, company_catalog_rich, available_indices):
    """
    Gera um plano de ação dinâmico.
    ALTERADO: Agora inclui uma busca determinística por tópicos antes de usar o LLM.
    """
    api_key = GEMINI_API_KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    query_lower = query.lower().strip()
    
    # --- ETAPA DE IDENTIFICAÇÃO DE EMPRESAS (Lógica inalterada) ---
    mentioned_companies = []
    companies_found_by_alias = {}
    for company_data in company_catalog_rich:
        for alias in company_data.get("aliases", []):
            if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
                score = len(alias.split())
                canonical_name = company_data["canonical_name"]
                if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                    companies_found_by_alias[canonical_name] = score
    
    if companies_found_by_alias:
        sorted_companies = sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)
        mentioned_companies = [company for company, score in sorted_companies]
    
    if not mentioned_companies:
        # Lógica de fallback para identificação de empresas (inalterada)
        # ... (seu código de fallback aqui) ...
        pass

    # --- ETAPA DE IDENTIFICAÇÃO DE TÓPICOS (LÓGICA ALTERADA) ---
    topics = []
    
    # NOVO: Etapa 1 - Busca Determinística de Tópicos
    found_topics = set()
    # Criar um mapa reverso para encontrar o nome canônico do tópico a partir de um alias
    alias_to_canonical_map = {}
    for canonical, aliases in TERMOS_TECNICOS_LTIP.items():
        for alias in aliases:
            alias_to_canonical_map[alias.lower()] = canonical

    # Verificar se algum alias de tópico está na query
    for alias, canonical_name in alias_to_canonical_map.items():
        # Usar \b (word boundary) para garantir que estamos combinando palavras inteiras
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            found_topics.add(canonical_name)

    # ALTERADO: Se encontrarmos tópicos deterministicamente, usamos eles. Senão, usamos o LLM.
    if found_topics:
        topics = list(found_topics)
        print(f"INFO: Tópicos identificados deterministicamente: {topics}") # Log para debug
    else:
        # Etapa 2 (Fallback) - Usar LLM se nenhum tópico específico for encontrado
        print("INFO: Nenhum tópico específico encontrado, usando LLM para análise geral.") # Log para debug
        prompt = f"""
        Você é um consultor de incentivos de longo prazo. Sua tarefa é identificar os TÓPICOS CENTRAIS de uma pergunta.
        
        Pergunta do usuário: "{query}"
        
        Analise a pergunta e retorne APENAS uma lista em formato JSON com os tópicos mais relevantes da lista abaixo.
        Se a pergunta for muito genérica ou comparativa, selecione os tópicos mais importantes para uma análise geral.
        
        Tópicos Disponíveis: {json.dumps(AVAILABLE_TOPICS)}
        
        Formato da Resposta (apenas JSON): ["Tópico 1", "Tópico 2", ...]
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
                # Fallback final se o LLM não retornar JSON
                topics = AVAILABLE_TOPICS
        except Exception as e:
            print(f"ERRO: Falha ao chamar LLM para tópicos. Usando todos os tópicos. Erro: {e}")
            topics = AVAILABLE_TOPICS
            
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}

####################################################################################

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes de configuração
class Config:
    MAX_CONTEXT_TOKENS = 12000
    MAX_CHUNKS_PER_TOPIC = 5
    SCORE_THRESHOLD_GENERAL = 0.4
    SCORE_THRESHOLD_ITEM_84 = 0.5
    DEDUPLICATION_HASH_LENGTH = 100

def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """
    Executa o plano de busca com controle robusto de tokens e deduplicação.
    """
    full_context = ""
    all_retrieved_docs = set()
    unique_chunks_content = set()  # Para deduplicação baseada em hash
    current_token_count = 0
    chunks_processed = 0

    def estimate_tokens(text):
        """Estima tokens de forma mais precisa."""
        # Aproximação: 1 token ≈ 4 caracteres para português
        return len(text) // 4

    def generate_chunk_hash(chunk_text):
        """Gera hash único para deduplicação de chunks."""
        # Remove espaços e normaliza para comparação
        normalized = re.sub(r'\s+', '', chunk_text.lower())
        return hash(normalized[:Config.DEDUPLICATION_HASH_LENGTH])

    def add_unique_chunk_to_context(chunk_text, source_info):
        """
        Adiciona chunk ao contexto com controle de tokens e deduplicação.
        """
        nonlocal full_context, current_token_count, chunks_processed
        
        # 1. Verificação de deduplicação
        chunk_hash = generate_chunk_hash(chunk_text)
        if chunk_hash in unique_chunks_content:
            logger.debug(f"Chunk duplicado ignorado: {source_info[:50]}...")
            return "DUPLICATE"
        
        # 2. Estimativa de tokens
        estimated_chunk_tokens = estimate_tokens(chunk_text)
        estimated_source_tokens = estimate_tokens(source_info)
        total_estimated_tokens = estimated_chunk_tokens + estimated_source_tokens + 10  # Buffer
        
        # 3. Verificação de limite de tokens
        if current_token_count + total_estimated_tokens > Config.MAX_CONTEXT_TOKENS:
            logger.warning(f"Limite de tokens atingido. Atual: {current_token_count}, Tentando adicionar: {total_estimated_tokens}")
            return "LIMIT_REACHED"
        
        # 4. Verificação de limite de chunks por tópico
        if chunks_processed >= Config.MAX_CHUNKS_PER_TOPIC * len(plan.get("topicos", [])):
            logger.info(f"Limite máximo de chunks atingido: {chunks_processed}")
            return "MAX_CHUNKS_REACHED"
        
        # 5. Adiciona ao contexto
        unique_chunks_content.add(chunk_hash)
        full_context += f"--- {source_info} ---\n{chunk_text}\n\n"
        current_token_count += total_estimated_tokens
        chunks_processed += 1
        
        # 6. Extrai nome do documento para tracking
        if "(Doc: " in source_info and ")" in source_info:
            try:
                doc_name = source_info.split("(Doc: ")[1].split(")")[0]
                all_retrieved_docs.add(doc_name)
            except IndexError:
                logger.warning(f"Não foi possível extrair nome do documento de: {source_info}")
        
        logger.debug(f"Chunk adicionado. Tokens atuais: {current_token_count}, Chunks: {chunks_processed}")
        return "SUCCESS"

    # Processamento principal
    for empresa in plan.get("empresas", []):
        searchable_company_name = normalize_name(empresa).split(' ')[0]
        logger.info(f"Processando empresa: {empresa}")
        
        if query_intent == 'item_8_4_query':
            # --- LÓGICA ESPECÍFICA PARA ITEM 8.4 ---
            full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
            
            if 'item_8_4' in artifacts:
                full_context += f"=== SEÇÃO COMPLETA DO ITEM 8.4 - {empresa.upper()} ===\n\n"
                
                artifact_data = artifacts['item_8_4']
                chunk_data = artifact_data['chunks']
                item_84_chunks_added = 0
                
                for i, mapping in enumerate(chunk_data.get('map', [])):
                    document_path = mapping['document_path']
                    if searchable_company_name in document_path.lower():
                        chunk_text = chunk_data["chunks"][i]
                        
                        result = add_unique_chunk_to_context(
                            chunk_text, 
                            f"Chunk Item 8.4 (Doc: {document_path})"
                        )
                        
                        if result == "LIMIT_REACHED":
                            st.warning(f"⚠️ Limite de tokens atingido para {empresa}. Resultado pode estar incompleto.")
                            break
                        elif result == "MAX_CHUNKS_REACHED":
                            st.info(f"ℹ️ Limite máximo de chunks atingido para {empresa}.")
                            break
                        elif result == "SUCCESS":
                            item_84_chunks_added += 1
                
                logger.info(f"Item 8.4: {item_84_chunks_added} chunks adicionados para {empresa}")
                full_context += f"=== FIM DA SEÇÃO ITEM 8.4 - {empresa.upper()} ===\n\n"
            
            # Busca complementar com controle de tokens
            complementary_indices = [idx for idx in artifacts.keys() if idx != 'item_8_4']
            
            for topico_idx, topico in enumerate(plan.get("topicos", [])[:10]):
                if current_token_count >= Config.MAX_CONTEXT_TOKENS * 0.9:  # Para em 90% do limite
                    logger.warning(f"Parando busca complementar - próximo ao limite de tokens")
                    break
                
                expanded_terms = expand_search_terms(topico)
                
                for term_idx, term in enumerate(expanded_terms[:5]):
                    search_query = f"item 8.4 {term} empresa {empresa}"
                    
                    for index_name in complementary_indices:
                        if index_name in artifacts:
                            try:
                                artifact_data = artifacts[index_name]
                                index = artifact_data['index']
                                chunk_data = artifact_data['chunks']
                                
                                query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                                scores, indices = index.search(query_embedding, 3)
                                
                                chunks_found = 0
                                for i, idx in enumerate(indices[0]):
                                    if idx != -1 and idx < len(chunk_data.get("chunks", [])) and scores[0][i] > Config.SCORE_THRESHOLD_ITEM_84:
                                        document_path = chunk_data["map"][idx]['document_path']
                                        if searchable_company_name in document_path.lower():
                                            chunk_text = chunk_data["chunks"][idx]
                                            
                                            result = add_unique_chunk_to_context(
                                                chunk_text,
                                                f"Contexto COMPLEMENTAR para '{topico}' via '{term}' (Fonte: {index_name}, Score: {scores[0][i]:.3f})"
                                            )
                                            
                                            if result == "LIMIT_REACHED":
                                                logger.warning(f"Limite de tokens atingido na busca complementar")
                                                break
                                            elif result == "SUCCESS":
                                                chunks_found += 1
                                
                                if chunks_found > 0 or current_token_count >= Config.MAX_CONTEXT_TOKENS * 0.8:
                                    break
                                    
                            except Exception as e:
                                logger.error(f"Erro na busca complementar em {index_name}: {e}")
                                continue
                    
                    if chunks_found > 0 or current_token_count >= Config.MAX_CONTEXT_TOKENS * 0.8:
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
        
        else:
            # --- LÓGICA PARA BUSCA GERAL ---
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
                tags_chunks_added = 0
                
                for chunk_info in tagged_chunks:
                    result = add_unique_chunk_to_context(
                        chunk_info['text'], 
                        f"Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']})"
                    )
                    
                    if result == "LIMIT_REACHED":
                        st.warning(f"⚠️ Limite de tokens atingido nos chunks com tags para {empresa}")
                        break
                    elif result == "SUCCESS":
                        tags_chunks_added += 1
                
                logger.info(f"Tags: {tags_chunks_added} chunks adicionados para {empresa}")
                full_context += f"=== FIM DOS CHUNKS COM TAGS - {empresa.upper()} ===\n\n"
            
            # Busca semântica complementar com controle rigoroso
            indices_to_search = list(artifacts.keys())
            
            for topico in plan.get("topicos", []):
                if current_token_count >= Config.MAX_CONTEXT_TOKENS * 0.85:  # Para em 85% para busca geral
                    logger.warning(f"Parando busca semântica - próximo ao limite de tokens")
                    break
                
                expanded_terms = expand_search_terms(topico)
                
                for term in expanded_terms[:3]:
                    search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                    
                    chunks_found = 0
                    for index_name in indices_to_search:
                        if index_name in artifacts:
                            try:
                                artifact_data = artifacts[index_name]
                                index = artifact_data['index']
                                chunk_data = artifact_data['chunks']
                                
                                query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                                scores, indices = index.search(query_embedding, TOP_K_SEARCH)
                                
                                for i, idx in enumerate(indices[0]):
                                    if idx != -1 and scores[0][i] > Config.SCORE_THRESHOLD_GENERAL:
                                        document_path = chunk_data["map"][idx]['document_path']
                                        if searchable_company_name in document_path.lower():
                                            chunk_text = chunk_data["chunks"][idx]
                                            
                                            result = add_unique_chunk_to_context(
                                                chunk_text,
                                                f"Contexto para '{topico}' via '{term}' (Fonte: {index_name}, Score: {scores[0][i]:.3f})"
                                            )
                                            
                                            if result == "LIMIT_REACHED":
                                                logger.warning(f"Limite de tokens atingido na busca semântica")
                                                break
                                            elif result == "SUCCESS":
                                                chunks_found += 1
                                
                                if chunks_found > 0:
                                    break
                                    
                            except Exception as e:
                                logger.error(f"Erro na busca semântica em {index_name}: {e}")
                                continue
                    
                    if chunks_found > 0:
                        break
            
            full_context += f"--- FIM DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
    
    # Verificação final
    if not unique_chunks_content:
        logger.warning("Nenhum chunk único encontrado")
        return "Nenhuma informação única encontrada para os critérios especificados.", set()

    # Log de estatísticas finais
    logger.info(f"Processamento concluído - Tokens: {current_token_count}/{Config.MAX_CONTEXT_TOKENS}, "
                f"Chunks únicos: {len(unique_chunks_content)}, Documentos: {len(all_retrieved_docs)}")
    
    return full_context, all_retrieved_docs



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
        
    prompt = f'Você é um consultor de incentivos de longo prazo e o item 8 do formulário de referencia da CVM. PERGUNTA ORIGINAL DO USUÁRIO: "{query}" CONTEXTO COLETADO DOS DOCUMENTOS: {context} {structure_instruction} INSTRUÇÕES PARA O RELATÓRIO FINAL: 1. Responda diretamente à pergunta do usuário. 2. PRIORIZE informações da SEÇÃO COMPLETA DO ITEM 8.4 ou de CHUNKS COM TAGS ESPECÍFICAS quando disponíveis. 3. Use informações complementares apenas para esclarecer. 4. Seja detalhado, preciso e profissional. 5. Se alguma informação não estiver disponível, indique: "Informação não encontrada nas fontes analisadas". RELATÓRIO ANALÍTICO FINAL:'
    
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
