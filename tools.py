# tools.py (Versão Final e Definitiva)

"""
Módulo de ferramentas para o Agente de Análise de Planos de Incentivo.

Este módulo contém a lógica para as capacidades avançadas do agente, incluindo:
- Busca híbrida por empresas com base em um tópico.
- Reclassificação de chunks para máxima precisão de contexto.
- Extração de resumos detalhados para um tópico em uma empresa específica.
- Orquestração de uma análise temática completa para identificar padrões de mercado.
"""

import faiss
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import requests
import logging
import random
logger = logging.getLogger(__name__)

# Habilidades: 'thematic', 'listing', 'statistic', 'comparison'
CAPABILITY_MAP = {
    # Tipos de Plano (Geralmente não têm estatísticas diretas)
    "AcoesRestritas": ["listing", "thematic", "comparison", "example_deep_dive"],
    "OpcoesDeCompra": ["listing", "thematic", "comparison", "example_deep_dive"],
    "AcoesFantasmas": ["listing", "thematic", "comparison"],
    "Matching Coinvestimento": ["listing", "thematic", "comparison"],

    # Mecânicas (Alguns têm estatísticas claras)
    "Vesting": ["statistic", "thematic", "listing", "comparison", "example_deep_dive"],
    "Lockup": ["statistic", "thematic", "listing", "comparison"],
    "PrecoDesconto": ["statistic", "listing", "thematic"],
    "VestingAcelerado": ["listing", "thematic", "comparison"],
    "Outorga": ["thematic", "listing"],

    # Governança e Risco
    "MalusClawback": ["listing", "thematic", "comparison", "example_deep_dive"],
    "Diluicao": ["statistic", "listing", "thematic"],
    "OrgaoDeliberativo": ["listing", "thematic"],

    # Participantes e Condições
    "Elegibilidade": ["listing", "thematic", "comparison"],
    "CondicaoSaida": ["thematic", "listing", "comparison", "example_deep_dive"],

    # Indicadores de Performance
    "TSR Relativo": ["listing", "thematic", "comparison", "example_deep_dive"],
    "TSR Absoluto": ["listing", "thematic", "comparison"],
    "ESG": ["listing", "thematic"],
    "GrupoDeComparacao": ["thematic", "listing"],

    # Eventos Financeiros
    "DividendosProventos": ["listing", "thematic", "comparison", "example_deep_dive"],
    "MudancaDeControle": ["listing", "thematic", "comparison"],
}

# MAPEIA HABILIDADES PARA TEMPLATES DE PERGUNTAS GARANTIDAS
QUESTION_TEMPLATES = {
    "thematic": [
        "Analise os modelos típicos de **{topic}** encontrados nos planos das empresas.",
        "Quais são as abordagens mais comuns para **{topic}** no mercado brasileiro?",
        "Descreva os padrões de mercado relacionados a **{topic}**."
    ],
    "listing": [
        "Quais empresas na base de dados possuem planos com **{topic}**?",
        "Gere uma lista de companhias que mencionam **{topic}** em seus documentos.",
        "Liste as empresas que adotam práticas de **{topic}**."
    ],
    "statistic": [
        "Qual o valor médio ou mais comum para o tópico de **{topic}** entre as empresas?",
        "Apresente as estatísticas (média, mediana, máximo) para **{topic}**.",
        "Qual a prevalência de **{topic}** nos planos analisados?"
    ],
    "comparison": [
        "Compare como a Vale e a Petrobras abordam o tópico de **{topic}** em seus planos.",
        "Quais as principais diferenças entre os planos da Magazine Luiza e da Localiza sobre **{topic}**?",
    ],
    # NOVO TIPO: Uma forma inteligente de responder perguntas de "definição"
    # com exemplos reais, o que aciona o RAG perfeitamente.
    "example_deep_dive": [
        "Como o plano da Vale define e aplica o conceito de **{topic}**?",
        "Descreva em detalhes como a Hypera trata a questão de **{topic}** em seu plano de remuneração.",
    ]
}

def _create_company_lookup_map(company_catalog_rich: list) -> dict:
    """
    Cria um dicionário de mapeamento reverso: {nome_variante.lower(): canonical_name}.
    """
    lookup_map = {}
    if not company_catalog_rich:
        return lookup_map
        
    for company_data in company_catalog_rich:
        canonical_name = company_data.get("canonical_name")
        if not canonical_name:
            continue
        
        # Adiciona o próprio nome canônico e todos os apelidos ao mapa
        all_names_to_map = [canonical_name] + company_data.get("aliases", [])
        
        for name in all_names_to_map:
            lookup_map[name.lower()] = canonical_name
            
    return lookup_map

def get_final_unified_answer(query: str, context: str) -> str:
    # Acessamos as variáveis de configuração através do Streamlit
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.0-flash-lite" # Usando o modelo mais recente recomendado

    if not GEMINI_API_KEY:
        return "Erro: A chave da API do Gemini não foi configurada."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    # Lógica da sua função original...
    structure_instruction = "Organize a resposta de forma lógica e clara usando Markdown."
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
        # Adiciona verificação para o caso de a resposta não ter o formato esperado
        candidates = response.json().get('candidates', [])
        if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
            return candidates[0]['content']['parts'][0]['text'].strip()
        else:
            logger.error(f"Resposta inesperada da API Gemini: {response.json()}")
            return "Ocorreu um erro ao processar a resposta do modelo de linguagem."
            
    except Exception as e:
        logger.error(f"ERRO ao gerar resposta final com LLM: {e}")
        return f"Ocorreu um erro ao contatar o modelo de linguagem. Detalhes: {str(e)}"

# --- FUNÇÃO DE SUGESTÃO (AGORA CORRIGIDA) ---

def suggest_alternative_query(failed_query: str) -> str:
    """
    Motor de sugestão robusto que usa os mapas enriquecidos para gerar
    sugestões variadas, relevantes e garantidas de funcionar.
    """
    # ... imports locais e extração de tópicos (como na versão anterior) ...
    from tools import get_final_unified_answer, _create_alias_to_canonical_map, _get_all_canonical_topics_from_text
    from knowledge_base import DICIONARIO_UNIFICADO_HIERARQUICO

    print("---EXECUTANDO MOTOR DE SUGESTÃO ROBUSTO (v2)---")

    alias_map, _ = _create_alias_to_canonical_map(DICIONARIO_UNIFICADO_HIERARQUICO)
    topics_found = _get_all_canonical_topics_from_text(failed_query.lower(), alias_map)
    
    safe_suggestions = []
    context_for_llm = ""

    if topics_found:
        primary_topic = topics_found[0]
        context_for_llm = f"O usuário demonstrou interesse no tópico de '{primary_topic}'."
        print(f"---TÓPICO IDENTIFICADO: '{primary_topic}'---")

        capabilities = CAPABILITY_MAP.get(primary_topic, ["listing", "thematic"])
        print(f"---HABILIDADES ENCONTRADAS PARA O TÓPICO: {capabilities}---")
        
        # Gera uma sugestão para cada capacidade, escolhendo uma variação aleatória do template
        for cap in capabilities:
            template_variations = QUESTION_TEMPLATES.get(cap)
            if template_variations:
                # Escolhe aleatoriamente um dos templates para aquela habilidade
                chosen_template = random.choice(template_variations)
                safe_suggestions.append(chosen_template.format(topic=primary_topic))
    else:
        # Se nenhum tópico foi encontrado, oferecer sugestões gerais e seguras
        context_for_llm = "O usuário fez uma pergunta genérica ou sobre um tópico não reconhecido."
        print("---NENHUM TÓPICO IDENTIFICADO. USANDO SUGESTÕES GERAIS E SEGURAS.---")
        
        # --- BLOCO DE SUGESTÕES CORRIGIDO ---
        safe_suggestions = [
            # MANTÉM: Esta pergunta aciona a ferramenta de listagem (qualitativa)
            "Liste as empresas que utilizam TSR Relativo como métrica de performance.",
            
            # SUBSTITUIÇÃO: Troca a pergunta quantitativa por uma qualitativa poderosa
            "Analise os modelos típicos de planos de Ações Restritas (RSU), o tipo mais comum no mercado.",
            
            # MANTÉM: Esta pergunta aciona o RAG para uma empresa específica (qualitativa)
            "Como funciona o plano de vesting da Vale?"
        ]
        # 

    # Limita o número de sugestões para não sobrecarregar o usuário
    safe_suggestions = safe_suggestions[:3]

    # O refinamento com o LLM continua sendo uma boa prática para naturalidade
    prompt = f"""
    Você é um assistente de IA prestativo. A pergunta de um usuário falhou porque não identificamos a empresa mencionada.
    PERGUNTA ORIGINAL: "{failed_query}"
    CONTEXTO: {context_for_llm}
    Eu gerei uma lista de perguntas "seguras" que eu sei responder. Sua tarefa é **apresentá-las de forma clara e convidativa** para o usuário. Você pode fazer pequenas melhorias para soar mais natural, mas **mantenha a intenção original** de cada pergunta intacta.

    PERGUNTAS SEGURAS PARA APRESENTAR:
    {json.dumps(safe_suggestions, indent=2, ensure_ascii=False)}

    Apresente o resultado como uma lista de marcadores em Markdown.
    """
    
    final_suggestions = get_final_unified_answer(prompt, "")
    return final_suggestions
    
def _create_alias_to_canonical_map(kb: dict) -> tuple[dict, dict]:
    """
    Cria dois mapeamentos a partir da knowledge base:
    1. alias_map: De qualquer alias (em minúsculas) para seu tópico canônico formatado.
    2. canonical_topics: De um tópico canônico para a lista de todos os seus aliases.
    """
    alias_map = {}
    canonical_topics = defaultdict(list)
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            canonical_name = topic_name_raw.replace('_', ' ')
            all_aliases = aliases + [canonical_name]
            for alias in all_aliases:
                alias_map[alias.lower()] = canonical_name 
            canonical_topics[canonical_name] = all_aliases
    return alias_map, canonical_topics

def _get_all_canonical_topics_from_text(text: str, alias_map: dict) -> list[str]:
    """
    Função aprimorada que encontra TODOS os tópicos canônicos relevantes.
    Ela busca por aliases exatos e também por termos gerais contidos em nomes de tópicos mais longos.
    """
    found_topics = set()
    text_lower = text.lower()

    # 1. Busca por aliases exatos (mais precisa)
    for alias, canonical_topic in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
            found_topics.add(canonical_topic)

    # 2. Busca por termos curtos/genéricos dentro de outros tópicos (mais abrangente)
    matched_aliases = {alias for alias in alias_map if re.search(r'\b' + re.escape(alias) + r'\b', text_lower)}
    all_canonical_names = alias_map.values()

    for alias in matched_aliases:
        for canonical_name in all_canonical_names:
            if alias in canonical_name.lower() and len(alias) < len(canonical_name):
                found_topics.add(canonical_name)
                
    return sorted(list(found_topics))

def _find_companies_by_exact_tag(canonical_topic: str, artifacts: dict, kb: dict) -> set[str]:
    """
    Busca empresas que possuem chunks com a tag de tópico exata.
    """
    topic_tag_name = None
    for section, topics in kb.items():
        for topic_name_raw, _ in topics.items():
            if topic_name_raw.replace('_', ' ') == canonical_topic:
                topic_tag_name = topic_name_raw
                break
        if topic_tag_name:
            break
            
    if not topic_tag_name:
        return set()

    found_companies = set()
    for artifact_data in artifacts.values():
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        for mapping in chunk_map:
            if topic_tag_name in mapping.get("topics_in_doc", []):
                company_name = mapping.get("company_name")
                if company_name:
                    found_companies.add(company_name)
    return found_companies

# --- FERRAMENTA DE RE-RANKING ---

def rerank_with_cross_encoder(query: str, chunks: list[dict], cross_encoder_model: CrossEncoder, top_n: int = 7) -> list[dict]:
    """
    Usa um modelo Cross-Encoder para reordenar uma lista de chunks e retorna os N melhores.
    """
    if not chunks:
        return []
    
    pairs = [[query, chunk.get('text', '')] for chunk in chunks]
    scores = cross_encoder_model.predict(pairs, show_progress_bar=False)
    
    for i, chunk in enumerate(chunks):
        chunk['relevance_score'] = scores[i]
        
    reranked_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0.0), reverse=True)
    
    return reranked_chunks[:top_n]

# --- FERRAMENTAS PRINCIPAIS ---

def find_companies_by_topic(topic: str, artifacts: dict, model: SentenceTransformer, kb: dict, top_k: int = 20) -> list[str]:
    """
    Ferramenta de Listagem: Busca por um tópico usando uma estratégia híbrida.
    """
    print(f"Buscando empresas para o tópico canônico: {topic}")
    
    _, canonical_map = _create_alias_to_canonical_map(kb)
    search_terms = canonical_map.get(topic, [topic])

    companies_from_tags = _find_companies_by_exact_tag(topic, artifacts, kb)
    
    companies_from_vector = set()
    search_query = f"informações e detalhes sobre {', '.join(search_terms)}"
    query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
    
    for artifact_data in artifacts.values():
        index = artifact_data.get('index')
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        if not index or not chunk_map: continue
        
        _, indices = index.search(query_embedding, top_k)
        for idx in indices[0]:
            if idx != -1:
                company_name = chunk_map[idx].get("company_name")
                if company_name: companies_from_vector.add(company_name)

    final_companies = sorted(list(companies_from_tags.union(companies_from_vector)))
    print(f"Encontradas {len(final_companies)} empresas no total para o tópico '{topic}'.")
    return final_companies

def get_summary_for_topic_at_company(
    company: str, 
    topic: str, 
    query: str,
    kb: dict, 
    artifacts: dict, 
    model: SentenceTransformer, 
    cross_encoder_model: CrossEncoder,
    execute_dynamic_plan_func: callable, 
    get_final_unified_answer_func: callable
) -> str:
    """
    Ferramenta de Extração: Busca, re-ranqueia e resume um tópico para uma empresa específica.
    """
    plan = {"empresas": [company], "topicos": [topic]}
    
    context, _ = execute_dynamic_plan_func(
        query, plan, artifacts, model, cross_encoder_model, kb, is_summary_plan=False
    )
    
    if not context:
        return "Não foi possível encontrar detalhes específicos sobre este tópico para esta empresa."
        
    summary_prompt = f"""
    Com base no contexto fornecido sobre a empresa {company}, resuma em detalhes as regras e o funcionamento do plano relacionadas ao tópico: '{topic}'.
    Seja direto e foque apenas nas informações relevantes para o tópico.

    CONTEXTO:
    {context}
    """
    summary = get_final_unified_answer_func(summary_prompt, context)
    return summary

def analyze_topic_thematically(
    topic: str, 
    query: str, 
    artifacts: dict, 
    model: SentenceTransformer,
    cross_encoder_model: CrossEncoder,
    kb: dict, 
    execute_dynamic_plan_func: callable, 
    get_final_unified_answer_func: callable
) -> str:
    """
    Ferramenta de Orquestração: Realiza uma análise temática completa de um tópico.
    """
    print(f"Iniciando análise temática para o tópico: {topic}")
    
    companies_to_analyze = find_companies_by_topic(topic, artifacts, model, kb)
    
    if not companies_to_analyze:
        return f"Não foram encontradas empresas com informações suficientes sobre o tópico '{topic}' para realizar uma análise temática."
    
    print(f"Analisando o tópico '{topic}' para {len(companies_to_analyze)} empresas...")

    company_summaries = []
    with ThreadPoolExecutor(max_workers=len(companies_to_analyze)) as executor:
        futures = {
            executor.submit(
                get_summary_for_topic_at_company, 
                company, topic, query, kb, artifacts, model, cross_encoder_model,
                execute_dynamic_plan_func, get_final_unified_answer_func
            ): company for company in companies_to_analyze
        }
        for future in futures:
            company = futures[future]
            try:
                summary_text = future.result()
                company_summaries.append({"empresa": company, "resumo_do_plano": summary_text})
            except Exception as e:
                print(f"Erro ao analisar a empresa {company}: {e}")
                company_summaries.append({"empresa": company, "resumo_do_plano": f"Erro ao processar a análise: {e}"})

    synthesis_context = json.dumps(company_summaries, indent=2, ensure_ascii=False)
    
    synthesis_prompt = f"""
    Você é um consultor especialista em remuneração e planos de incentivo.
    Sua tarefa é responder à pergunta original do usuário: "{query}"
    Para isso, analise o CONTEXTO JSON abaixo, que contém resumos dos planos de várias empresas sobre o tópico '{topic}'.

    CONTEXTO:
    {synthesis_context}

    INSTRUÇÕES PARA O RELATÓRIO TEMÁTICO:
    1.  **Introdução:** Comece com um parágrafo que resume suas principais descobertas.
    2.  **Identificação de Padrões:** Analise todos os resumos e identifique de 2 a 4 "modelos" ou "padrões" comuns.
    3.  **Descrição dos Padrões:** Para cada padrão, descreva-o e liste as empresas que o seguem.
    4.  **Exceções e Casos Únicos:** Destaque abordagens diferentes ou inovadoras.
    5.  **Conclusão:** Finalize com uma breve conclusão sobre as práticas de mercado.

    Seja analítico, estruturado e use Markdown para formatar sua resposta de forma clara.
    """
    
    final_report = get_final_unified_answer_func(synthesis_prompt, synthesis_context)
    return final_report
