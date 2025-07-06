# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP - VERSÃO STREAMLIT (HÍBRIDO)
Aplicação web para análise de planos de incentivo de longo prazo, com
capacidades de busca profunda (RAG) e análise agregada (resumo).
"""

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
import logging
from functools import lru_cache


# --- FUNÇÕES AUXILIARES GLOBAIS ---
# Estas funções são usadas pelo fluxo de análise profunda (RAG)

def expand_search_terms(base_term):
    """Expande um termo de busca para incluir sinônimos do dicionário principal."""
    expanded_terms = [base_term.lower()]
    # Usando TERMOS_TECNICOS_LTIP que já está definido globalmente
    for category, terms in TERMOS_TECNICOS_LTIP.items():
        # Verificamos se o 'base_term' ou a 'category' estão relacionados
        if base_term.lower() in (t.lower() for t in terms) or base_term.lower() == category.lower():
            expanded_terms.extend([term.lower() for term in terms])
    return list(set(expanded_terms))

def search_by_tags(artifacts, company_name, target_tags):
    """
    Busca por chunks que contenham tags de tópicos pré-processados,
    considerando tanto 'Tópicos:' quanto 'Item 8.4 - Subitens:'.
    """
    results = []
    # Normaliza o nome da empresa para a busca no caminho do arquivo
    searchable_company_name = unicodedata.normalize('NFKD', company_name.lower()).encode('ascii', 'ignore').decode('utf-8').split(' ')[0]

    for index_name, artifact_data in artifacts.items():
        chunk_data = artifact_data.get('chunks', {})
        for i, mapping in enumerate(chunk_data.get('map', [])):
            document_path = mapping.get('document_path', '')
            
            if searchable_company_name in document_path.lower():
                chunk_text = chunk_data.get("chunks", [])[i]
                for tag in target_tags:
                    # LÓGICA CORRIGIDA: A regex agora procura pela tag após qualquer um dos dois prefixos.
                    pattern = r'(Tópicos:|Item 8.4 - Subitens:).*?' + re.escape(tag)
                    if re.search(pattern, chunk_text, re.IGNORECASE):
                        results.append({
                            'text': chunk_text, 'path': document_path, 'index': i,
                            'source': index_name, 'tag_found': tag
                        })
                        break # Para no primeiro tag encontrado para este chunk
    return results

def normalize_name(name):
    """Normaliza nomes de empresas removendo acentos, pontuação e sufixos comuns."""
    try:
        # Converte para minúsculas e remove acentos
        nfkd_form = unicodedata.normalize('NFKD', name.lower())
        name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        
        # Remove pontuação e caracteres especiais
        name = re.sub(r'[.,-]', '', name)
        
        # Remove sufixos comuns de empresas
        suffixes = [r'\bs\.?a\.?\b', r'\bltda\b', r'\bholding\b', r'\bparticipacoes\b', r'\bcia\b', r'\bind\b', r'\bcom\b']
        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            
        # Remove espaços extras
        return re.sub(r'\s+', ' ', name).strip()
    except Exception as e:
        # Fallback em caso de erro
        return name.lower()

# --- CONFIGURAÇÕES GERAIS ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "Gemini 2.0 Flash-Lite"  # Modelo Gemini unificado
DADOS_PATH = "dados" # Centraliza o caminho para a pasta de dados

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DICIONÁRIOS DE CONHECIMENTO ---

# Dicionário principal para tradução de termos e busca de tópicos
TERMOS_TECNICOS_LTIP = {
    # --- TIPOS DE PLANOS ---
    "Ações Restritas": ["Restricted Shares", "Plano de Ações Restritas", "Outorga de Ações", "ações restritas", "RSU", "Restricted Stock Units"],
    "Opções de Compra de Ações": ["Stock Options", "ESOP", "Plano de Opção de Compra", "Outorga de Opções", "opções", "Plano de Opção", "Plano de Opções", "SOP"],
    "Ações Fantasmas": ["Phantom Shares", "Ações Virtuais"],
    "Opções Fantasmas (SAR)": ["Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
    "Planos com Condição de Performance": ["Performance Shares", "Performance Units", "PSU", "Plano de Desempenho", "Metas de Performance", "performance", "desempenho"],
    "Plano de Compra de Ações (ESPP)": ["Plano de Compra de Ações", "Employee Stock Purchase Plan", "ESPP", "Ações com Desconto"], # (NOVA CATEGORIA)
    "Bônus Diferido": ["Staying Bonus", "Retention Bonus", "Bônus de Permanência", "Bônus de Retenção", "bônus", "Deferred Bonus"],
    "Matching": ["Matching", "Contrapartida", "Co-investimento", "Plano de Matching", "investimento"],

    # --- MECÂNICAS E CICLO DE VIDA ---
    "Outorga": ["Outorga", "Concessão", "Grant", "Grant Date", "Data da Outorga", "Aprovação"],
    "Vesting": ["Vesting", "Período de Carência", "Condições de Carência", "Aquisição de Direitos", "carência", "cronograma de vesting"],
    "Antecipação de Vesting": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação de carência", "antecipação do vesting", "antecipação"],
    "Tranche / Lote": ["Tranche", "Lote", "Parcela do Vesting"],
    "Cliff": ["Cliff Period", "Período de Cliff", "Carência Inicial"],
    "Preço": ["Preço", "Preço de Exercício", "Strike", "Strike Price"],
    "Ciclo de Vida do Exercício": ["Exercício", "Período de Exercício", "pagamento", "liquidação", "vencimento", "expiração", "forma de liquidação"],
    "Lockup": ["Lockup", "Período de Lockup", "Restrição de Venda", "período de restrição"],

    # --- GOVERNANÇA E RISCO ---
    "Governança e Documentos": ["Regulamento", "Regulamento do Plano", "Contrato de Adesão", "Termo de Outorga", "Comitê de Remuneração", "Comitê de Pessoas", "Deliberação"], # (NOVA CATEGORIA)
    "Malus e Clawback": ["Malus", "Clawback", "Redução", "Devolução", "Cláusula de Recuperação", "Forfeiture", "Cancelamento", "Perda do Direito"], # (NOVA CATEGORIA)
    "Estrutura do Plano/Programa": ["Plano", "Planos", "Programa", "Programas", "termos e condições gerais"],
    "Diluição": ["Diluição", "Dilution", "Capital Social"], # (NOVA CATEGORIA)

    # --- PARTICIPANTES E CONDIÇÕES ---
    "Elegíveis": ["Participantes", "Beneficiários", "Elegíveis", "Empregados", "Administradores", "Executivos", "Colaboradores", "Conselheiros"],
    "Condição de Saída": ["Desligamento", "Saída", "Término do Contrato", "Rescisão", "Demissão", "Good Leaver", "Bad Leaver"],
    "Tratamento em Casos Especiais": ["Aposentadoria", "Morte", "Invalidez", "Reforma", "Afastamento"], # (NOVA CATEGORIA)
    "Indicadores": ["TSR", "Total Shareholder Return", "Retorno Total ao Acionista", "CDI", "IPCA", "Selic", "ROIC", "EBITDA", "LAIR", "Lucro", "CAGR", "Metas ESG", "Receita Líquida"],

    # --- EVENTOS E ASPECTOS FINANCEIROS ---
    "Eventos Corporativos": ["IPO", "grupamento", "desdobramento", "cisão", "fusão", "incorporação", "bonificações", "bonificação"],
    "Mudança de Controle": ["Mudança de Controle", "Change of Control", "Evento de Liquidez"], # (NOVA CATEGORIA)
    "Dividendos": ["Dividendos", "Dividendo", "JCP", "Juros sobre capital próprio", "Tratamento de Dividendos", "dividend equivalent", "proventos"],
    "Encargos": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"],
    "Contabilidade e Normas": ["IFRS 2", "CPC 10", "Valor Justo", "Fair Value", "Black-Scholes", "Despesa Contábil", "Volatilidade"] # (NOVA CATEGORIA)
}

# Tópicos para o fallback do LLM na análise profunda (RAG)
AVAILABLE_TOPICS = list(TERMOS_TECNICOS_LTIP.keys()) + [
    "data de aprovação e órgão responsável", "número máximo de ações abrangidas", "número máximo de opções a serem outorgadas",
    "critérios para fixação do preço de aquisição ou exercício", "preço de exercício", "strike price", "restrições à transferência das ações",
    "critérios e eventos de suspensão/extinção", "efeitos da saída do administrador"
]

# --- CARREGAMENTO DE DADOS E CACHING ---

@st.cache_resource
def load_all_artifacts():
    """
    Carrega todos os artefatos, incluindo o novo resumo DETALHADO.
    """
    model = SentenceTransformer(MODEL_NAME)
    artifacts = {}
    index_files = glob.glob(os.path.join(DADOS_PATH, '*_faiss_index.bin'))
    if not index_files:
        logger.error("Nenhum arquivo de índice FAISS encontrado na pasta 'dados'. O RAG não funcionará.")
    else:
        for index_file in index_files:
            category = os.path.basename(index_file).replace('_faiss_index.bin', '')
            chunks_file = os.path.join(DADOS_PATH, f"{category}_chunks_map.json")
            try:
                index = faiss.read_index(index_file)
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                artifacts[category] = {'index': index, 'chunks': chunk_data}
            except FileNotFoundError:
                logger.warning(f"Arquivo de chunks para a categoria '{category}' não encontrado. Pulando.")
                continue
    
    # Carrega o NOVO arquivo de resumo detalhado
    summary_data = None
    summary_file_path = os.path.join(DADOS_PATH, 'resumo_caracteristicas.json')
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        logger.error("Arquivo 'resumo_caracteristicas.json' não encontrado. Buscas agregadas não funcionarão.")
        
    return model, artifacts, summary_data

@st.cache_data
def criar_mapa_de_alias():
    """
    Cria um dicionário que mapeia cada apelido E o próprio nome do tópico ao seu tópico canônico.
    Ex: {'performance': 'Planos com Condição de Performance', 'planos com condição de performance': 'Planos com Condição de Performance'}
    """
    alias_to_canonical = {}
    for canonical_name, aliases in TERMOS_TECNICOS_LTIP.items():
        # 1. Adiciona o próprio nome canônico como um alias para si mesmo
        alias_to_canonical[canonical_name.lower()] = canonical_name
        
        # 2. Adiciona todos os outros sinônimos
        for alias in aliases:
            alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

# --- FUNÇÕES DE LÓGICA DE NEGÓCIO (ROTEADOR E MANIPULADORES) ---

def handle_aggregate_query(query, summary_data, alias_map):
    """
    Lida com perguntas agregadas usando o resumo detalhado para filtros precisos.
    """
    query_lower = query.lower()
    
    # 1. Encontrar TODOS os termos específicos mencionados na pergunta (ex: 'tsr', 'vesting')
    query_keywords = set()
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    
    temp_query = query_lower
    for alias in sorted_aliases:
        # Busca e substitui para não contar sub-palavras de termos já encontrados
        if re.search(r'\b' + re.escape(alias) + r'\b', temp_query):
            query_keywords.add(alias)
            # Remove a palavra encontrada para evitar matches duplicados (ex: "ações" e "ações restritas")
            temp_query = temp_query.replace(alias, "")

    if not query_keywords:
        st.warning("Não consegui identificar um termo técnico conhecido (como 'TSR', 'vesting', 'matching') na sua pergunta.")
        return

    st.info(f"Termos específicos identificados para a busca: **{', '.join(sorted(list(query_keywords)))}**")

    # 2. Filtrar as empresas com base nos termos específicos
    empresas_encontradas = []
    for empresa, data in summary_data.items():
        # Coleta todas as palavras-chave encontradas para uma empresa em um único set
        all_found_keywords_for_company = set()
        # A estrutura agora é: {'Tópico': ['keyword1', 'keyword2']}
        for topic, keywords_list in data.get("topicos_encontrados", {}).items():
            all_found_keywords_for_company.update([k.lower() for k in keywords_list])
            # Adiciona o próprio nome do tópico como uma keyword pesquisável
            all_found_keywords_for_company.add(topic.lower().split(' - ')[-1]) # Trata "Item 8.4 - a"

        # A empresa é uma candidata se TODOS os termos da pergunta estiverem no conjunto de palavras-chave dela
        if query_keywords.issubset(all_found_keywords_for_company):
            empresas_encontradas.append(empresa)

    # 3. Formatar e exibir a resposta
    import pandas as pd

    if not empresas_encontradas:
        st.warning(f"Nenhuma empresa foi encontrada com **TODOS** os termos específicos mencionados.")
        return

    st.success(f"✅ **{len(empresas_encontradas)} empresa(s)** encontrada(s) com os termos específicos: **{', '.join(sorted(list(query_keywords)))}**")
    df = pd.DataFrame(sorted(empresas_encontradas), columns=["Empresa"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def handle_rag_query(query, artifacts, model, company_catalog_rich):
    """
    Lida com perguntas detalhadas e comparativas usando o fluxo RAG completo.
    """
    # ETAPA 1: GERAÇÃO DO PLANO
    with st.status("1️⃣ Gerando plano de análise...", expanded=True) as status:
        # Nota: Idealmente, company_catalog_rich seria carregado uma vez fora.
        # Por simplicidade, mantemos aqui.
        plan_response = create_dynamic_analysis_plan_v2(query, company_catalog_rich, list(artifacts.keys()))
        if plan_response['status'] != "success" or not plan_response['plan']['empresas']:
            st.error("❌ Não consegui identificar empresas na sua pergunta. Tente usar nomes conhecidos (ex: Magalu, Vivo, Itaú).")
            return "Análise abortada.", set()
        
        plan = plan_response['plan']
        empresas = plan.get('empresas', [])
        st.write(f"**🏢 Empresas identificadas:** {', '.join(empresas)}")
        st.write(f"**📝 Tópicos a analisar:** {len(plan.get('topicos', []))}")
        status.update(label="✅ Plano gerado com sucesso!", state="complete")

    # ETAPA 2: LÓGICA DE EXECUÇÃO (com tratamento para comparações)
    final_answer = ""
    sources = set()

    # MODO COMPARATIVO
    if len(empresas) > 1:
        st.info(f"Modo de comparação ativado para {len(empresas)} empresas. Analisando sequencialmente...")
        summaries = []
        for i, empresa in enumerate(empresas):
            with st.status(f"Analisando {i+1}/{len(empresas)}: {empresa}...", expanded=True):
                single_company_plan = {'empresas': [empresa], 'topicos': plan['topicos']}
                query_intent = 'item_8_4_query' if any(term in query.lower() for term in ['8.4', 'formulário']) else 'general_query'
                retrieved_context, retrieved_sources = execute_dynamic_plan(single_company_plan, query_intent, artifacts, model)
                sources.update(retrieved_sources)

                if "Nenhuma informação" in retrieved_context or not retrieved_context.strip():
                    summary = f"## Análise para {empresa.upper()}\n\nNenhuma informação encontrada nos documentos para os tópicos solicitados."
                else:
                    summary_prompt = f"Com base no contexto a seguir sobre a empresa {empresa}, resuma os pontos principais sobre os seguintes tópicos: {', '.join(plan['topicos'])}. Contexto: {retrieved_context}"
                    summary = get_final_unified_answer(summary_prompt, retrieved_context)
                
                summaries.append(f"--- RESUMO PARA {empresa.upper()} ---\n\n{summary}")

        with st.status("Gerando relatório comparativo final...", expanded=True):
            comparison_prompt = f"Com base nos resumos individuais a seguir, crie um relatório comparativo detalhado e bem estruturado entre as empresas, focando nos pontos levantados na pergunta original do usuário.\n\nPergunta original do usuário: '{query}'\n\n" + "\n\n".join(summaries)
            final_answer = get_final_unified_answer(comparison_prompt, "\n\n".join(summaries))
            status.update(label="✅ Relatório comparativo gerado!", state="complete")

    # MODO DE ANÁLISE ÚNICA
    else:
        with st.status("2️⃣ Recuperando contexto relevante...", expanded=True) as status:
            query_intent = 'item_8_4_query' if any(term in query.lower() for term in ['8.4', 'formulário']) else 'general_query'
            st.write(f"**🎯 Estratégia detectada:** {'Item 8.4 completo' if query_intent == 'item_8_4_query' else 'Busca geral'}")
            
            retrieved_context, retrieved_sources = execute_dynamic_plan(plan, query_intent, artifacts, model)
            sources.update(retrieved_sources)
            
            if not retrieved_context.strip() or "Nenhuma informação encontrada" in retrieved_context:
                st.error("❌ Não encontrei informações relevantes nos documentos para a sua consulta.")
                return "Nenhuma informação relevante encontrada.", set()
            
            st.write(f"**📄 Contexto recuperado de:** {len(sources)} documento(s)")
            status.update(label="✅ Contexto recuperado com sucesso!", state="complete")
        
        with st.status("3️⃣ Gerando resposta final...", expanded=True) as status:
            final_answer = get_final_unified_answer(query, retrieved_context)
            status.update(label="✅ Análise concluída!", state="complete")

    return final_answer, sources

# --- FUNÇÕES DE BACKEND (RAG) - com modelo atualizado ---

def create_dynamic_analysis_plan_v2(query, company_catalog_rich, available_indices):
    # Esta função agora é chamada apenas pelo `handle_rag_query`
    api_key = GEMINI_API_KEY
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    query_lower = query.lower().strip()
    
    # Identificação de Empresas
    mentioned_companies = []
    companies_found_by_alias = {}
    if company_catalog_rich:
        for company_data in company_catalog_rich:
            for alias in company_data.get("aliases", []):
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                    score = len(alias.split())
                    canonical_name = company_data["canonical_name"]
                    if canonical_name not in companies_found_by_alias or score > companies_found_by_alias[canonical_name]:
                        companies_found_by_alias[canonical_name] = score
        if companies_found_by_alias:
            sorted_companies = sorted(companies_found_by_alias.items(), key=lambda item: item[1], reverse=True)
            mentioned_companies = [company for company, score in sorted_companies]
    
    if not mentioned_companies:
        return {"status": "error", "plan": {}}
    
    # Identificação de Tópicos
    topics = []
    found_topics = set()
    alias_map = criar_mapa_de_alias() # Reutiliza o mapa de alias
    for alias, canonical_name in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            found_topics.add(canonical_name)

    if found_topics:
        topics = list(found_topics)
    else:
        # Fallback para LLM se nenhum tópico for encontrado
        prompt = f"""Você é um consultor de ILP. Identifique os TÓPICOS CENTRAIS da pergunta: "{query}".
        Retorne APENAS uma lista JSON com os tópicos mais relevantes de: {json.dumps(AVAILABLE_TOPICS)}.
        Se for genérica, selecione tópicos para uma análise geral. Formato: ["Tópico 1", "Tópico 2"]"""
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
                topics = ["Estrutura do Plano/Programa", "Vesting", "Opções de Compra de Ações"]
        except Exception as e:
            logger.error(f"Falha ao chamar LLM para tópicos: {e}")
            topics = ["Estrutura do Plano/Programa", "Vesting", "Opções de Compra de Ações"]
            
    plan = {"empresas": mentioned_companies, "topicos": topics}
    return {"status": "success", "plan": plan}


def execute_dynamic_plan(plan, query_intent, artifacts, model):
    """
    Executa o plano de busca com controle robusto de tokens e deduplicação.
    """
    full_context = ""
    all_retrieved_docs = set()
    unique_chunks_content = set()
    current_token_count = 0
    chunks_processed = 0

    class Config: # Usando uma classe interna para manter as constantes da função
        MAX_CONTEXT_TOKENS = 12000
        MAX_CHUNKS_PER_TOPIC = 5
        SCORE_THRESHOLD_GENERAL = 0.4
        SCORE_THRESHOLD_ITEM_84 = 0.5
        DEDUPLICATION_HASH_LENGTH = 100

    def estimate_tokens(text):
        return len(text) // 4

    def generate_chunk_hash(chunk_text):
        normalized = re.sub(r'\s+', '', chunk_text.lower())
        return hash(normalized[:Config.DEDUPLICATION_HASH_LENGTH])

    def add_unique_chunk_to_context(chunk_text, source_info):
        nonlocal full_context, current_token_count, chunks_processed, unique_chunks_content, all_retrieved_docs
        
        chunk_hash = generate_chunk_hash(chunk_text)
        if chunk_hash in unique_chunks_content:
            logger.debug(f"Chunk duplicado ignorado: {source_info[:50]}...")
            return "DUPLICATE"
        
        estimated_chunk_tokens = estimate_tokens(chunk_text) + estimate_tokens(source_info) + 10
        
        if current_token_count + estimated_chunk_tokens > Config.MAX_CONTEXT_TOKENS:
            logger.warning(f"Limite de tokens atingido. Atual: {current_token_count}")
            return "LIMIT_REACHED"
        
        unique_chunks_content.add(chunk_hash)
        full_context += f"--- {source_info} ---\n{chunk_text}\n\n"
        current_token_count += estimated_chunk_tokens
        chunks_processed += 1
        
        try:
            doc_name = source_info.split("(Doc: ")[1].split(")")[0]
            all_retrieved_docs.add(doc_name)
        except IndexError:
            pass # Ignora se não conseguir extrair o nome
        
        logger.debug(f"Chunk adicionado. Tokens atuais: {current_token_count}")
        return "SUCCESS"

    for empresa in plan.get("empresas", []):
        searchable_company_name = unicodedata.normalize('NFKD', empresa.lower()).encode('ascii', 'ignore').decode('utf-8').split(' ')[0]
        logger.info(f"Processando empresa: {empresa}")

        # Lógica para busca geral (pode ser adaptada para item_8_4 também)
        full_context += f"--- INÍCIO DA ANÁLISE PARA: {empresa.upper()} ---\n\n"
        
        # Busca por tags
        target_tags = []
        for topico in plan.get("topicos", []):
            target_tags.extend(expand_search_terms(topico))
        target_tags = list(set([tag.title() for tag in target_tags if len(tag) > 3]))
        
        # A função search_by_tags precisa estar definida no seu script
        tagged_chunks = search_by_tags(artifacts, empresa, target_tags)
        
        if tagged_chunks:
            full_context += f"=== CHUNKS COM TAGS ESPECÍFICAS - {empresa.upper()} ===\n\n"
            for chunk_info in tagged_chunks:
                add_unique_chunk_to_context(
                    chunk_info['text'], 
                    f"Chunk com tag '{chunk_info['tag_found']}' (Doc: {chunk_info['path']})"
                )

        # Busca semântica complementar
        for topico in plan.get("topicos", []):
            expanded_terms = expand_search_terms(topico)
            for term in expanded_terms[:3]:
                search_query = f"informações sobre {term} no plano de remuneração da empresa {empresa}"
                query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
                
                for index_name, artifact_data in artifacts.items():
                    index = artifact_data['index']
                    chunk_data = artifact_data['chunks']
                    scores, indices = index.search(query_embedding, TOP_K_SEARCH)
                    
                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and scores[0][i] > Config.SCORE_THRESHOLD_GENERAL:
                            document_path = chunk_data["map"][idx]['document_path']
                            if searchable_company_name in document_path.lower():
                                chunk_text = chunk_data["chunks"][idx]
                                add_unique_chunk_to_context(
                                    chunk_text,
                                    f"Contexto para '{topico}' via '{term}' (Fonte: {index_name}, Score: {scores[0][i]:.3f}, Doc: {document_path})"
                                )

    if not unique_chunks_content:
        logger.warning("Nenhum chunk único encontrado para o plano de execução.")
        return "Nenhuma informação única encontrada para os critérios especificados.", set()

    logger.info(f"Processamento concluído - Tokens: {current_token_count}, Chunks únicos: {len(unique_chunks_content)}")
    return full_context, all_retrieved_docs

def get_final_unified_answer(query, context):
    """Gera a resposta final usando o contexto recuperado e a API do Gemini."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    has_complete_8_4 = "=== SEÇÃO COMPLETA DO ITEM 8.4" in context
    has_tagged_chunks = "=== CHUNKS COM TAGS ESPECÍFICAS" in context
    
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
    if has_complete_8_4:
        structure_instruction = """
**ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4:**
Use a estrutura oficial do item 8.4 do Formulário de Referência:
a) Termos e condições gerais; b) Data de aprovação e órgão; c) Máximo de ações; d) Máximo de opções; 
e) Condições de aquisição; f) Critérios de preço; g) Critérios de prazo; h) Forma de liquidação; 
i) Restrições à transferência; j) Suspensão/extinção; k) Efeitos da saída.
Para cada subitem, extraia e organize as informações encontradas na SEÇÃO COMPLETA DO ITEM 8.4.
"""
    elif has_tagged_chunks:
        structure_instruction = "**PRIORIZE** as informações dos CHUNKS COM TAGS ESPECÍFICAS e organize a resposta de forma lógica usando Markdown."
        
    prompt = f"""Você é um consultor especialista em planos de incentivo de longo prazo (ILP) e no item 8 do formulário de referência da CVM.
    
    PERGUNTA ORIGINAL DO USUÁRIO: "{query}"
    
    CONTEXTO COLETADO DOS DOCUMENTOS:
    {context}
    
    {structure_instruction}
    
    INSTRUÇÕES PARA O RELATÓRIO FINAL:
    1. Responda diretamente à pergunta do usuário com base no contexto fornecido.
    2. PRIORIZE informações da "SEÇÃO COMPLETA DO ITEM 8.4" ou de "CHUNKS COM TAGS ESPECÍFICAS" quando disponíveis. Use o resto do contexto para complementar.
    3. Seja detalhado, preciso e profissional na sua linguagem. Use formatação Markdown (negrito, listas) para clareza.
    4. Se uma informação específica pedida não estiver no contexto, declare explicitamente: "Informação não encontrada nas fontes analisadas.". Não invente dados.
    
    RELATÓRIO ANALÍTICO FINAL:
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"ERRO ao gerar resposta final: Ocorreu um problema ao contatar o modelo de linguagem. Detalhes: {e}"


# --- INTERFACE STREAMLIT (Aplicação Principal) ---
def main():
    st.set_page_config(page_title="Agente de Análise LTIP", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    st.markdown("---")

    # Carregamento centralizado dos artefatos
    model, artifacts, summary_data = load_all_artifacts()
    ALIAS_MAP = criar_mapa_de_alias()

    # Tenta carregar o catálogo de empresas, mas não quebra se não encontrar
    try:
        from catalog_data import company_catalog_rich
    except ImportError:
        company_catalog_rich = []
        logger.warning("`catalog_data.py` não encontrado. A identificação de empresas por apelidos será limitada.")

    # Validação dos dados carregados
    if not artifacts:
        st.error("❌ Erro crítico: Nenhum artefato de busca (índices FAISS) foi carregado. A análise profunda está desativada.")
    if not summary_data:
        st.warning("⚠️ Aviso: O arquivo `resumo_caracteristicas.json` não foi encontrado. Análises de 'quais/quantas empresas' estão desativadas.")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Fontes de Documentos (RAG)", len(artifacts) if artifacts else 0)
        st.metric("Empresas no Resumo", len(summary_data) if summary_data else 0)
        
        if summary_data:
            with st.expander("empresas com características identificadas"):
                st.dataframe(sorted(list(summary_data.keys())), use_container_width=True)
        
        st.success("✅ Sistema pronto para análise")
        st.info(f"Embedding Model: `{MODEL_NAME}`")
        st.info(f"Generative Model: `{GEMINI_MODEL}`") # Mostra o modelo Gemini em uso

    # --- Corpo Principal ---
    st.header("💬 Faça sua pergunta")
    
        # Colunas para exemplos de perguntas
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Experimente uma análise agregada:**")
        st.code("Quais empresas possuem planos com matching?")
        st.code("Quantas empresas têm performance?")
        st.code("Quantas empresas têm Stock Options?")
    with col2:
        st.info("**Ou uma análise profunda :**")
        st.code("Compare dividendos da Vale com a Gerdau")
        st.code("Como é o plano Magazine Luiza?")
        st.code("Resumo item 8.4 Movida")
    st.caption("**Principais Termos-Chave:** `Item 8.4`, `Vesting`, `Stock Options`, `Ações Restritas`, `Performance`, `Matching`, `Lockup`, `SAR`, `ESPP`, `Malus e Clawback`, `Dividendos`")

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quantas empresas oferecem ações restritas? ")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            return

        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
        
        # --- O ROTEADOR DE INTENÇÃO EM AÇÃO ---
        final_answer = ""
        sources = set()
        
        query_lower = user_query.lower()
        aggregate_keywords = ["quais", "quantas", "liste", "qual a lista de"]

        # Rota 1: Pergunta agregada
        if any(keyword in query_lower for keyword in aggregate_keywords):
            if not summary_data:
                st.error("A funcionalidade de busca agregada está desativada pois o arquivo `resumo_caracteristicas.json` não foi encontrado.")
            else:
                st.info("Detectada uma pergunta agregada. Buscando no resumo de características...")
                with st.spinner("Analisando resumo..."):
                    # A função `handle_aggregate_query` já renderiza a saída em st.
                    handle_aggregate_query(user_query, summary_data, ALIAS_MAP)

        # Rota 2: Pergunta profunda (RAG)
        else:
            if not artifacts:
                st.error("A funcionalidade de análise profunda está desativada pois os índices de busca não foram encontrados.")
            elif not company_catalog_rich:
                 st.error("A funcionalidade de análise profunda está desativada pois o `catalog_data.py` não foi encontrado.")
            else:
                st.info("Detectada uma pergunta detalhada. Acionando análise profunda (RAG)...")
                final_answer, sources = handle_rag_query(user_query, artifacts, model, company_catalog_rich)
                st.markdown(final_answer) # Renderiza a resposta do RAG

        # Fontes consultadas (apenas para o RAG)
        if sources:
            st.markdown("---")
            with st.expander(f"📚 Documentos consultados na análise profunda ({len(sources)})", expanded=False):
                for i, source in enumerate(sorted(list(sources)), 1):
                    st.write(f"{i}. {source}")

if __name__ == "__main__":
    main()
