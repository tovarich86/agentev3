# -*- coding: utf-8 -*-
"""
AGENTE DE ANÁLISE LTIP - VERSÃO FINAL E FUNCIONAL
"""

# --- 1. Importações e Configurações ---
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
import pandas as pd
from pathlib import Path

try:
    from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
except ImportError:
    st.error("ERRO CRÍTICO: Crie o arquivo 'knowledge_base.py' e cole o 'DICIONARIO_UNIFICADO_HIERARQUICO' nele.")
    st.stop()

# Configurações Gerais
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K_SEARCH = 7
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash-latest" # Usando o modelo mais recente e capaz
DADOS_PATH = Path("dados")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 2. Carregamento de Dados e Funções Auxiliares ---

@st.cache_resource
def load_all_artifacts():
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
                    r = requests.get(url, stream=True)
                    r.raise_for_status()
                    with open(caminho_arquivo, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                except requests.exceptions.RequestException as e:
                    st.error(f"Falha ao baixar {nome_arquivo}: {e}"); st.stop()
    
    _model = SentenceTransformer(MODEL_NAME)
    artifacts = {}
    index_files = glob.glob(str(DADOS_PATH / '*_faiss_index_final.bin'))
    for index_file_path in index_files:
        category = Path(index_file_path).stem.replace('_faiss_index_final', '')
        chunks_file_path = DADOS_PATH / f"{category}_chunks_map_final.json"
        try:
            index = faiss.read_index(index_file_path)
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            artifacts[category] = {'index': index, 'chunks': chunk_data}
        except Exception as e:
            logger.error(f"Erro ao carregar '{category}': {e}"); continue
    
    summary_data = None
    summary_file_path = DADOS_PATH / 'resumo_fatos_e_topicos_final_enriquecido.json'
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo de resumo '{summary_file_path}' não encontrado.")
            
    return _model, artifacts, summary_data

@st.cache_data
def criar_mapa_de_alias():
    alias_to_canonical = {}
    for section, topics in DICIONARIO_UNIFICADO_HIERARQUICO.items():
        for canonical_name, aliases in topics.items():
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical


# --- 3. Funções de Lógica de Negócio (Manipuladores de Query) ---

def handle_analytical_query(query: str, summary_data: dict):
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
    
    st.info(f"Analisando: **{operation.upper()}** para o fato **'{target_fact_key}'**...")
    valores, unidade = [], ''
    for data in summary_data.values():
        if target_fact_key in data.get("fatos_extraidos", {}):
            fact_data = data["fatos_extraidos"][target_fact_key]
            valor = fact_data.get('valor_numerico') or fact_data.get('valor')
            if isinstance(valor, (int, float)):
                valores.append(valor)
                if not unidade and 'unidade' in fact_data: unidade = fact_data['unidade']
    if not valores:
        st.warning(f"Não encontrei dados numéricos para '{target_fact_key}' para calcular."); return True

    resultado, label_metrica = 0, ""
    if operation == 'avg': resultado, label_metrica = np.mean(valores), f"Média de {target_fact_key.replace('_', ' ')}"
    elif operation == 'min': resultado, label_metrica = np.min(valores), f"Mínimo de {target_fact_key.replace('_', ' ')}"
    elif operation == 'max': resultado, label_metrica = np.max(valores), f"Máximo de {target_fact_key.replace('_', ' ')}"
    
    valor_formatado = f"{resultado:.1%}" if target_fact_key == 'desconto_strike_price' else f"{resultado:.1f} {unidade}".strip()
    st.metric(label=label_metrica.title(), value=valor_formatado)
    st.caption(f"Cálculo baseado em {len(valores)} empresas com dados para este fato.")
    return True

def handle_aggregate_query(query: str, summary_data: dict, alias_map: dict):
    query_lower = query.lower()
    query_keywords = set()
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    temp_query = query_lower
    for alias in sorted_aliases:
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', temp_query):
            query_keywords.add(alias.lower()); temp_query = temp_query.replace(alias.lower(), "")
    if not query_keywords: st.warning("Não identifiquei um termo técnico na sua pergunta."); return
    st.info(f"Termos para busca: **{', '.join(sorted(list(query_keywords)))}**")
    empresas_encontradas = []
    for empresa, data in summary_data.items():
        company_terms = set()
        for topics in data.get("topicos_encontrados", {}).values():
            for topic, aliases in topics.items():
                company_terms.add(topic.lower()); company_terms.update([a.lower() for a in aliases])
        if query_keywords.issubset(company_terms): empresas_encontradas.append(empresa)
    if not empresas_encontradas: st.warning(f"Nenhuma empresa encontrada com: `{', '.join(query_keywords)}`."); return
    st.success(f"✅ **{len(empresas_encontradas)} empresa(s)** encontrada(s).")
    df = pd.DataFrame(sorted(empresas_encontradas), columns=["Empresa"])
    st.dataframe(df, use_container_width=True, hide_index=True)

def handle_direct_fact_query(query: str, summary_data: dict, alias_map: dict, company_catalog: list):
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
    st.subheader(f"Fato Direto para: {empresa_encontrada}")
    fato_encontrado = False
    for fact_key, fact_value in empresa_data.get("fatos_extraidos", {}).items():
        if fato_encontrado_alias in fact_key.lower():
            valor, unidade = fact_value.get('valor', ''), fact_value.get('unidade', '')
            st.metric(label=f"Fato: {fact_key.replace('_', ' ').title()}", value=f"{valor} {unidade}".strip())
            fato_encontrado = True; break
    if not fato_encontrado: st.info(f"O tópico '{fato_encontrado_alias}' foi mencionado, mas um fato estruturado não foi extraído.")
    return True

# --- FUNÇÕES DO PIPELINE RAG (VERSÃO FINAL E FUNCIONAL) ---

# --- FUNÇÕES AUXILIARES RESTAURADAS E ADAPTADAS ---

def expand_search_terms(base_term: str, alias_map: dict, knowledge_base: dict) -> list[str]:
    """
    (ADAPTADO DO CÓDIGO ANTIGO) Expande um termo de busca para incluir sinônimos.
    Funciona com a nova estrutura do DICIONARIO_UNIFICADO_HIERARQUICO.
    """
    canonical_name = alias_map.get(base_term.lower())
    if not canonical_name:
        return [base_term]

    expanded_terms = set([canonical_name])
    for section, topics in knowledge_base.items():
        if canonical_name in topics:
            expanded_terms.update(topics[canonical_name])
            expanded_terms.add(canonical_name) # Garante que o nome canônico está lá
            break
            
    return list(expanded_terms)

def search_by_tags(artifacts: dict, company_name: str, target_tags: list[str]) -> list[dict]:
    """
    (ADAPTADO DO CÓDIGO ANTIGO) Busca por chunks que contenham tags de tópicos.
    Agora usa o metadado 'company_name' em vez de buscar no nome do arquivo, o que é mais robusto.
    """
    results = []
    # Converte tags para um formato mais fácil de buscar (ex: ignora case)
    target_tags_lower = {tag.lower() for tag in target_tags}

    for category, artifact_data in artifacts.items():
        chunk_data = artifact_data.get('chunks', {})
        for i, mapping in enumerate(chunk_data.get('map', [])):
            # LÓGICA ADAPTADA: Verifica o metadado da empresa
            if company_name.upper() == mapping.get("company_name", "").upper():
                chunk_text = chunk_data.get("chunks", [])[i]
                # Verifica se alguma das tags está no texto do chunk
                # Isso é uma simplificação, a busca por regex do código antigo era mais específica
                # para "Tópicos:" ou "Item 8.4 - Subitens:". Podemos adicionar se necessário.
                # Por agora, vamos buscar a menção da tag no texto.
                for tag in target_tags_lower:
                    if re.search(r'\b' + re.escape(tag) + r'\b', chunk_text, re.IGNORECASE):
                        results.append({
                            'text': chunk_text,
                            'source_url': mapping.get("source_url", "Fonte Desconhecida"),
                            'tag_found': tag
                        })
                        break # Pára no primeiro tag encontrado para este chunk
    return results

# --- FUNÇÕES DO PIPELINE RAG (VERSÃO HÍBRIDA E DINÂMICA) ---

def create_dynamic_rag_plan(query: str, company_catalog: list, alias_map: dict, knowledge_base: dict) -> dict | None:
    """
    (HÍBRIDO E DINÂMICO - V3) Cria um plano de busca, combinando regras locais com fallback para LLM.
    """
    query_lower = query.lower()
    plan = {"empresas": [], "topicos": []}
    
    # --- ETAPA 1: TENTATIVA DE PLANEJAMENTO LOCAL (RÁPIDO E BARATO) ---
    # Identifica empresas usando o catálogo
    for company_data in company_catalog:
        for alias in company_data.get("aliases", []):
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
                plan["empresas"].append(company_data["canonical_name"])
                break
    plan["empresas"] = sorted(list(set(plan["empresas"])))

    # Identifica tópicos usando o mapa de alias
    for alias, canonical_name in alias_map.items():
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', query_lower):
            plan["topicos"].append(canonical_name)
    plan["topicos"] = sorted(list(set(plan["topicos"])))

    # Se não encontrou empresa, o plano é inválido
    if not plan["empresas"]:
        return None

    # Se encontrou tópicos localmente, o plano está pronto!
    if plan["topicos"]:
        logger.info(f"Plano RAG criado com regras locais: Empresas={plan['empresas']}, Tópicos={plan['topicos']}")
        return plan

    # --- ETAPA 2: FALLBACK PARA LLM (SE NENHUM TÓPICO FOI ENCONTRADO) ---
    logger.warning(f"Nenhum tópico local encontrado para a query. Acionando LLM para planejamento.")
    st.info("Nenhum termo técnico conhecido foi encontrado. Usando IA para interpretar os tópicos da sua pergunta...")

    if not GEMINI_API_KEY:
        st.error("Chave da API Gemini não configurada para o planejamento dinâmico.")
        # Fallback para um plano genérico se a API não estiver disponível
        plan["topicos"] = ["informações gerais do plano de incentivo"]
        return plan

    # Extrai todos os nomes de tópicos canônicos da base de conhecimento
    available_topics = list(knowledge_base.keys())
    for section_topics in knowledge_base.values():
        available_topics.extend(section_topics.keys())
    
    prompt = f"""Você é um assistente especialista em planos de incentivo. Analise a pergunta do usuário e identifique os tópicos centrais que devem ser pesquisados.
    
    **Pergunta do Usuário:** "{query}"

    **Tópicos Disponíveis para Escolha:** {json.dumps(list(set(available_topics)), ensure_ascii=False)}

    **Sua Tarefa:**
    Retorne uma lista JSON com os nomes EXATOS dos tópicos mais relevantes da lista acima que correspondem à pergunta do usuário.
    Se a pergunta for genérica sobre um plano (ex: "como é o plano da vale?"), retorne uma lista com tópicos essenciais como ["Estrutura do Plano/Programa", "Vesting", "Governança e Documentos"].
    O formato da sua resposta deve ser APENAS a lista JSON. Exemplo: ["Vesting", "Lockup", "Dividendos"]
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        text_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        
        # Tenta extrair a lista JSON da resposta
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            plan["topicos"] = json.loads(json_match.group(0))
            logger.info(f"Plano RAG criado via LLM: Empresas={plan['empresas']}, Tópicos={plan['topicos']}")
        else:
            raise ValueError("Resposta do LLM não continha um JSON válido.")
            
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
        logger.error(f"Falha ao usar LLM para planejamento: {e}. Usando tópicos de fallback.")
        st.warning("Falha na interpretação por IA. Usando uma busca genérica.")
        plan["topicos"] = ["informações gerais do plano de incentivo", "Estrutura do Plano/Programa"]
        
    return plan


def execute_hybrid_rag_plan(plan: dict, artifacts: dict, model, alias_map: dict, knowledge_base: dict) -> tuple[str, set]:
    """
    (BUSCA HÍBRIDA) Executa a busca em duas etapas: tags (precisão) e semântica (cobertura).
    """
    full_context, sources, unique_chunks = "", set(), set()
    
    with st.spinner("Executando busca de alta precisão por tags..."):
        # Expande todos os tópicos do plano para seus sinônimos
        all_target_tags = []
        for topico in plan.get("topicos", []):
            all_target_tags.extend(expand_search_terms(topico, alias_map, knowledge_base))
        all_target_tags = list(set(all_target_tags))
        
        tagged_context = ""
        for empresa in plan["empresas"]:
            tagged_results = search_by_tags(artifacts, empresa, all_target_tags)
            if tagged_results:
                tagged_context += f"--- Contexto de Alta Precisão para {empresa.upper()} (Tags Encontradas) ---\n"
                for res in tagged_results:
                    chunk_text = res['text']
                    if chunk_text not in unique_chunks:
                        source_url = res['source_url']
                        tagged_context += f"Fonte (Tag: '{res['tag_found']}'): {os.path.basename(source_url)}\n{chunk_text}\n\n"
                        unique_chunks.add(chunk_text)
                        sources.add(source_url)
        full_context += tagged_context

    with st.spinner("Executando busca semântica para complementar o contexto..."):
        semantic_context = ""
        for empresa in plan["empresas"]:
            search_query = f"informações detalhadas sobre {', '.join(plan['topicos'])} no plano de remuneração da empresa {empresa}"
            query_embedding = model.encode([search_query], normalize_embeddings=True)

            for category, artifact_data in artifacts.items():
                scores, indices = artifact_data['index'].search(query_embedding, TOP_K_SEARCH)
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and scores[0][i] >= 0.35:
                        mapping = artifact_data["chunks"]["map"][idx]
                        if empresa.upper() == mapping.get("company_name", "").upper():
                            chunk_text = artifact_data["chunks"]["chunks"][idx]
                            if chunk_text not in unique_chunks:
                                source_url = mapping.get("source_url", "Fonte Desconhecida")
                                semantic_context += f"Fonte (Semântica): {os.path.basename(source_url)} (Similaridade: {scores[0][i]:.2f})\n{chunk_text}\n\n"
                                unique_chunks.add(chunk_text)
                                sources.add(source_url)
        
        if semantic_context:
            full_context += "--- Contexto Adicional (Busca Semântica Ampla) ---\n" + semantic_context

    return full_context, sources


def get_final_answer_with_dynamic_prompt(query: str, context: str):
    """
    (PROMPT DINÂMICO) Gera a resposta final, adaptando o prompt com base no contexto.
    """
    if not GEMINI_API_KEY:
        st.error("Chave da API Gemini não configurada. Verifique os segredos do Streamlit.")
        return "Erro: Chave da API não encontrada."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    # Lógica do Prompt Dinâmico
    structure_instruction = "Use formatação Markdown (negrito, listas) para clareza e legibilidade."
    if "item 8.4" in query.lower():
        structure_instruction = """
        **ESTRUTURA OBRIGATÓRIA PARA ITEM 8.4:**
        Organize a resposta usando a estrutura oficial do item 8.4 do Formulário de Referência da CVM:
        a) Termos e condições gerais; b) Data de aprovação e órgão; c) Máximo de ações; d) Máximo de opções;
        e) Condições de aquisição; f) Critérios de preço; g) Critérios de prazo; h) Forma de liquidação;
        i) Restrições à transferência; j) Suspensão/extinção; k) Efeitos da saída.
        Para cada subitem, extraia e organize as informações encontradas.
        """
    elif "Contexto de Alta Precisão" in context:
        structure_instruction = "PRIORIZE as informações da seção 'Contexto de Alta Precisão (Tags Encontradas)', pois são as mais relevantes. Use o 'Contexto Adicional' para complementar os detalhes. Organize a resposta de forma lógica usando Markdown."

    prompt = f"""Você é um consultor especialista em planos de remuneração da CVM. Sua tarefa é responder à pergunta do usuário de forma clara, profissional e em português, baseando-se estritamente no contexto fornecido.

    **Instruções Importantes:**
    1.  Use apenas as informações do 'Contexto Coletado'.
    2.  {structure_instruction}
    3.  Se a resposta não estiver no contexto, afirme explicitamente: "A informação não foi encontrada nos documentos analisados.". Não invente dados.

    **Pergunta do Usuário:** "{query}"

    **Contexto Coletado dos Documentos:**
    ---
    {context}
    ---
    
    **Relatório Analítico Detalhado:**
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192} # Aumentado para 8k
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        candidate = response.json().get('candidates', [{}])[0]
        content = candidate.get('content', {}).get('parts', [{}])[0]
        return content.get('text', "Não foi possível gerar uma resposta.")
    except requests.exceptions.RequestException as e:
        logger.error(f"ERRO de requisição ao chamar a API Gemini: {e}")
        return f"Erro de comunicação com a API do Gemini. Detalhes: {e}"
    except Exception as e:
        logger.error(f"ERRO inesperado ao processar resposta do Gemini: {e}")
        return f"Ocorreu um erro inesperado ao processar a resposta. Detalhes: {e}"


def handle_rag_query(query: str, artifacts: dict, model, company_catalog: list, alias_map: dict, knowledge_base: dict):
    """
    (ORQUESTRADOR ATUALIZADO) Orquestra o pipeline RAG Híbrido e Dinâmico.
    """
    with st.status("Gerando plano de análise RAG...") as status:
        plan = create_dynamic_rag_plan(query, company_catalog, alias_map, knowledge_base)
        if not plan:
            st.error("Não consegui identificar empresas conhecidas na sua pergunta para realizar a análise.")
            return set()
        status.update(label=f"Plano gerado. Analisando para: {', '.join(plan['empresas'])}...")

    # A execução agora é a HÍBRIDA
    context, sources = execute_hybrid_rag_plan(plan, artifacts, model, alias_map, knowledge_base)
    
    if not context:
        st.warning("Não encontrei informações relevantes nos documentos para esta consulta.")
        return set()
    
    with st.spinner("Gerando relatório final com base no contexto coletado..."):
        # A geração de resposta agora usa o PROMPT DINÂMICO
        final_answer = get_final_answer_with_dynamic_prompt(query, context)
        st.markdown(final_answer)
        
    return sources
    **Relatório Analítico Detalhado:**
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        # Extrai o texto da resposta da API
        candidate = response.json().get('candidates', [{}])[0]
        content = candidate.get('content', {}).get('parts', [{}])[0]
        return content.get('text', "Não foi possível gerar uma resposta.")
    except requests.exceptions.RequestException as e:
        logger.error(f"ERRO de requisição ao chamar a API Gemini: {e}")
        return f"Erro de comunicação com a API do Gemini. Detalhes: {e}"
    except Exception as e:
        logger.error(f"ERRO inesperado ao processar resposta do Gemini: {e}")
        return f"Ocorreu um erro inesperado ao processar a resposta. Detalhes: {e}"

def handle_rag_query(query: str, artifacts: dict, model, company_catalog: list, alias_map: dict):
    """
    (VERSÃO CORRIGIDA) Orquestra o pipeline RAG, chamando a função de API correta.
    """
    with st.status("Gerando plano de análise RAG...") as status:
        plan = create_rag_plan(query, company_catalog, alias_map)
        if not plan:
            st.error("Não consegui identificar empresas conhecidas na sua pergunta para realizar a análise.")
            return set()
        status.update(label=f"Plano gerado. Analisando para: {', '.join(plan['empresas'])}...")

    with st.spinner("Recuperando e analisando informações..."):
        context, sources = execute_rag_plan(plan, artifacts, model)
        if not context:
            st.warning("Não encontrei informações relevantes nos documentos para esta consulta.")
            return set()
        
        # Chamada para a função de API correta e funcional
        final_answer = get_final_unified_answer(query, context)
        st.markdown(final_answer)
        
    return sources

@st.cache_data
def criar_mapa_de_alias(knowledge_base: dict):
    """
    (VERSÃO CORRIGIDA) Cria um dicionário que mapeia cada apelido E o próprio nome do tópico 
    ao seu tópico canônico, recebendo a base de conhecimento como argumento.
    """
    alias_to_canonical = {}
    for section, topics in knowledge_base.items():
        for canonical_name, aliases in topics.items():
            alias_to_canonical[canonical_name.lower()] = canonical_name
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical_name
    return alias_to_canonical

def main():
    st.set_page_config(
        page_title="Agente de Análise LTIP", 
        page_icon="🤖", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    st.title("🤖 Agente de Análise de Planos de Incentivo (ILP)")
    
    # --- Carregamento Centralizado de Dados e Artefatos ---
    with st.spinner("Carregando modelos, índices e base de conhecimento..."):
        model, artifacts, summary_data = load_all_artifacts()
    
    try:
        from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO
        logger.info("Base de conhecimento 'knowledge_base.py' carregada.")
    except ImportError:
        st.error("ERRO CRÍTICO: Crie o arquivo 'knowledge_base.py' e cole o 'DICIONARIO_UNIFICADO_HIERARQUICO' nele.")
        st.stop()
        
    ALIAS_MAP = criar_mapa_de_alias(DICIONARIO_UNIFICADO_HIERARQUICO)

    try:
        from catalog_data import company_catalog_rich
        logger.info("Catálogo de empresas 'catalog_data.py' carregado.")
    except ImportError:
        logger.warning("`catalog_data.py` não encontrado. Criando catálogo dinâmico a partir do resumo.")
        if summary_data:
            company_catalog_rich = [{"canonical_name": name, "aliases": [name.split(' ')[0].lower(), name.lower()]} for name in summary_data.keys()]
        else:
            company_catalog_rich = []
            st.warning("Catálogo de empresas não pôde ser criado pois o arquivo de resumo também está ausente.")

    # --- Sidebar com Informações do Sistema (Inspirado no script original) ---
    with st.sidebar:
        st.header("📊 Informações do Sistema")
        st.metric("Fontes de Documentos (RAG)", len(artifacts) if artifacts else "N/A")
        st.metric("Empresas no Resumo", len(summary_data) if summary_data else "N/A")
        
        if summary_data:
            with st.expander("Empresas com dados para análise rápida"):
                empresas_df = pd.DataFrame(sorted(list(summary_data.keys())), columns=["Nome da Empresa"])
                st.dataframe(empresas_df, use_container_width=True, hide_index=True)
        
        st.success("✅ Sistema pronto para análise")
        st.info(f"**Modelo de Embedding:**\n`{MODEL_NAME}`")
        st.info(f"**Modelo Generativo:**\n`{GEMINI_MODEL}`")

    # --- Bloco de Orientação ao Usuário (Inspirado no script original) ---
    st.header("💬 Faça sua pergunta")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Experimente análises rápidas (sem RAG):**")
        st.code("Quais empresas possuem planos com matching?") # Agregada
        st.code("Qual o desconto médio oferecido?") # Analítica
        st.code("Qual o período de vesting da Movida?") # Fato Direto
    with col2:
        st.info("**Ou uma análise profunda (com RAG):**")
        st.code("Compare as políticas de dividendos da Vale e Gerdau") # Comparativa
        st.code("Como é o tratamento de desligamento no plano da Magazine Luiza?") # Detalhada
        st.code("Resumo completo do item 8.4 da Vivo") # Estruturada

    st.caption("**Principais Termos-Chave:** `Item 8.4`, `Vesting`, `Stock Options`, `Ações Restritas`, `Performance`, `Matching`, `Lockup`, `SAR`, `ESPP`, `Malus e Clawback`, `Dividendos`, `Good Leaver`, `Bad Leaver`")

    user_query = st.text_area("Sua pergunta:", height=100, placeholder="Ex: Quantas empresas oferecem ações restritas e possuem cláusula de clawback?")

    if st.button("🔍 Analisar", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Por favor, digite uma pergunta.")
            st.stop()

        st.markdown("---")
        st.subheader("📋 Resultado da Análise")
        
        # --- ROTEADOR DE INTENÇÃO DE 4 NÍVEIS COM FEEDBACK PARA O USUÁRIO ---
        query_lower = user_query.lower()
        analytical_keywords = ['medio', 'média', 'típico', 'menor', 'mínimo', 'maior', 'máximo']
        aggregate_keywords = ["quais", "quantas", "liste", "mostre"]
        direct_fact_pattern = r'qual\s*(?:é|o|a)\s*.*\s*d[aeo]\s*'
        sources = set()

        if any(keyword in query_lower for keyword in analytical_keywords):
            st.info("Detectada uma pergunta **analítica (média, mínimo, máximo)**. Buscando nos dados pré-processados...")
            if not handle_analytical_query(user_query, summary_data):
                st.warning("A análise rápida não encontrou dados numéricos. Acionando a análise profunda (RAG) para uma resposta mais completa...")
                sources = handle_rag_query(user_query, artifacts, model, company_catalog_rich, ALIAS_MAP, DICIONARIO_UNIFICADO_HIERARQUICO)
        
        elif any(keyword in query_lower for keyword in aggregate_keywords):
            st.info("Detectada uma pergunta **agregada (quais, quantas)**. Buscando na lista de empresas...")
            if not summary_data:
                st.error("A funcionalidade de busca agregada está desativada pois o arquivo de resumo não foi encontrado.")
            else:
                handle_aggregate_query(user_query, summary_data, ALIAS_MAP)
        
        elif re.search(direct_fact_pattern, query_lower) and any(comp["canonical_name"].lower() in query_lower for comp in company_catalog_rich):
            st.info("Detectada uma pergunta de **fato direto**. Buscando nos fatos extraídos...")
            if not handle_direct_fact_query(user_query, summary_data, ALIAS_MAP, company_catalog_rich):
                st.warning("Não encontrei um fato estruturado. Acionando a análise profunda (RAG) para buscar no texto completo...")
                sources = handle_rag_query(user_query, artifacts, model, company_catalog_rich, ALIAS_MAP, DICIONARIO_UNIFICADO_HIERARQUICO)
        
        else:
            st.info("Detectada uma pergunta **detalhada ou comparativa**. Acionando a análise profunda (RAG)...")
            if not artifacts:
                 st.error("A funcionalidade de análise profunda está desativada pois os índices de busca não foram encontrados.")
            else:
                sources = handle_rag_query(user_query, artifacts, model, company_catalog_rich, ALIAS_MAP, DICIONARIO_UNIFICADO_HIERARQUICO)

        # --- Exibição das Fontes Consultadas (Apenas para o RAG) ---
        if sources:
            st.markdown("---")
            with st.expander(f"📚 Fontes consultadas na análise profunda ({len(sources)})", expanded=False):
                # Usando um DataFrame para uma visualização mais limpa
                sources_df = pd.DataFrame(sorted(list(sources)), columns=["Documento"])
                st.dataframe(sources_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
