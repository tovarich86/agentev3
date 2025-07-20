# tools.py (Versão Final - Agente de Ferramentas)

"""
Módulo de ferramentas para o Agente de Análise de Planos de Incentivo.

Este módulo contém a lógica para as capacidades avançadas do agente, incluindo:
- Busca híbrida por empresas com base em um tópico.
- Extração de resumos detalhados para um tópico em uma empresa específica.
- Orquestração de uma análise temática completa para identificar padrões de mercado.
"""

import faiss
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# --- Funções Auxiliares Internas ---

def _create_alias_to_canonical_map(kb: dict) -> tuple[dict, dict]:
    """
    Cria dois mapeamentos a partir da knowledge base:
    1. alias_map: De qualquer alias (em minúsculas) para seu tópico canônico formatado (ex: "Matching Coinvestimento").
    2. canonical_topics: De um tópico canônico para a lista de todos os seus aliases.
    """
    alias_map = {}
    canonical_topics = defaultdict(list)
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            canonical_name = topic_name_raw.replace('_', ' ')
            # Adiciona o próprio nome canônico à lista de aliases para busca
            all_aliases = aliases + [canonical_name]
            for alias in all_aliases:
                alias_map[alias.lower()] = canonical_name
            canonical_topics[canonical_name] = all_aliases
    return alias_map, canonical_topics

def _get_canonical_topic_from_text(text: str, alias_map: dict) -> str | None:
    """
    Encontra o primeiro tópico canônico que corresponde a um alias em um texto de entrada.
    Prioriza aliases mais longos para evitar correspondências parciais (ex: "TSR" vs "TSR Relativo").
    """
    # Itera sobre os aliases do mais longo para o mais curto
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', text.lower()):
            return alias_map[alias]
    return None

def _find_companies_by_exact_tag(canonical_topic: str, artifacts: dict, kb: dict) -> set[str]:
    """
    Busca empresas que possuem chunks com a tag de tópico exata (ex: [topico:Matching_Coinvestimento]).
    """
    # Mapeia o nome canônico formatado (ex: "Matching Coinvestimento") de volta para o formato de tag (ex: "Matching_Coinvestimento")
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
    # A busca pela tag é uma checagem de substring, o que é extremamente rápido.
    for artifact_data in artifacts.values():
        all_chunks_text = artifact_data.get('chunks', {}).get('chunks', [])
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        for i, chunk_text in enumerate(all_chunks_text):
            if f"topico:{topic_tag_name}" in chunk_text:
                company_name = chunk_map[i].get("company_name")
                if company_name:
                    found_companies.add(company_name)
    return found_companies

# --- Ferramentas Principais (Para serem chamadas pelo app.py) ---

def find_companies_by_topic(topic: str, artifacts: dict, model: SentenceTransformer, kb: dict, top_k: int = 20) -> list[str]:
    """
    Ferramenta de Listagem: Busca por um tópico em todos os documentos usando uma estratégia híbrida
    e retorna uma lista de nomes de empresas únicas que possuem chunks relevantes.
    """
    print(f"Buscando empresas para o tópico canônico: {topic}")
    
    _, canonical_map = _create_alias_to_canonical_map(kb)
    search_terms = canonical_map.get(topic, [topic])

    # 1. Busca por Tag (Alta Precisão)
    companies_from_tags = _find_companies_by_exact_tag(topic, artifacts, kb)
    
    # 2. Busca Vetorial (Alto Alcance)
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

    # 3. Unificar Resultados
    final_companies = sorted(list(companies_from_tags.union(companies_from_vector)))
    print(f"Encontradas {len(final_companies)} empresas no total para o tópico '{topic}'.")
    return final_companies

def get_summary_for_topic_at_company(
    company: str, 
    topic: str, 
    kb: dict, 
    artifacts: dict, 
    model: SentenceTransformer, 
    execute_dynamic_plan_func: callable, 
    get_final_unified_answer_func: callable
) -> str:
    """
    Ferramenta de Extração: Busca e resume um tópico para uma empresa específica.
    """
    plan = {"empresas": [company], "topicos": [topic]}
    context, _ = execute_dynamic_plan_func(plan, artifacts, model, kb)
    
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
    kb: dict, 
    execute_dynamic_plan_func: callable, 
    get_final_unified_answer_func: callable
) -> str:
    """
    Ferramenta de Orquestração: Realiza uma análise temática completa de um tópico.
    """
    print(f"Iniciando análise temática para o tópico: {topic}")
    
    # Etapa 1: ENCONTRAR empresas relevantes
    companies_to_analyze = find_companies_by_topic(topic, artifacts, model, kb)
    
    if not companies_to_analyze:
        return f"Não foram encontradas empresas com informações suficientes sobre o tópico '{topic}' para realizar uma análise temática."
    
    print(f"Analisando o tópico '{topic}' para {len(companies_to_analyze)} empresas...")

    # Etapa 2: EXTRAIR detalhes de cada empresa em paralelo
    company_summaries = []
    with ThreadPoolExecutor(max_workers=len(companies_to_analyze)) as executor:
        futures = {
            executor.submit(
                get_summary_for_topic_at_company, 
                company, topic, kb, artifacts, model, 
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

    # Etapa 3: SINTETIZAR um relatório final a partir dos dados extraídos
    synthesis_context = json.dumps(company_summaries, indent=2, ensure_ascii=False)
    
    synthesis_prompt = f"""
    Você é um consultor especialista em remuneração e planos de incentivo.
    Sua tarefa é responder à pergunta original do usuário: "{query}"
    Para isso, analise o CONTEXTO JSON abaixo, que contém resumos dos planos de várias empresas sobre o tópico '{topic}'.

    CONTEXTO:
    {synthesis_context}

    INSTRUÇÕES PARA O RELATÓRIO TEMÁTICO:
    1.  **Introdução:** Comece com um parágrafo que resume suas principais descobertas.
    2.  **Identificação de Padrões:** Analise todos os resumos e identifique de 2 a 4 "modelos" ou "padrões" comuns de como as empresas estruturam seus planos para este tópico.
    3.  **Descrição dos Padrões:** Para cada padrão identificado, descreva-o em detalhes e liste as empresas do contexto que seguem aquele padrão. Use subtítulos para cada padrão.
    4.  **Exceções e Casos Únicos:** Se houver alguma empresa com uma abordagem muito diferente ou inovadora, destaque-a em uma seção separada.
    5.  **Conclusão:** Finalize com uma breve conclusão sobre as práticas de mercado para o tópico '{topic}'.

    Seja analítico, estruturado e use Markdown para formatar sua resposta de forma clara.
    """
    
    final_report = get_final_unified_answer_func(synthesis_prompt, synthesis_context)
    return final_report
