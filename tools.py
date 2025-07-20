# tools.py (Versão 3.0 - Com Análise Temática)

import faiss
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# --- Funções Auxiliares (do passo anterior, sem alterações) ---

def _create_alias_to_canonical_map(kb: dict) -> tuple[dict, dict]:
    alias_map = {}
    canonical_topics = defaultdict(list)
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            all_aliases = aliases + [topic_name_raw.replace('_', ' ')]
            for alias in all_aliases:
                alias_map[alias.lower()] = topic_name_raw
            canonical_topics[topic_name_raw] = all_aliases
    return alias_map, canonical_topics

def _get_canonical_topic(term: str, alias_map: dict) -> str | None:
    return alias_map.get(term.lower())

def _find_companies_by_exact_tag(canonical_topic: str, artifacts: dict) -> set[str]:
    found_companies = set()
    tag_to_find = f"[topico:{canonical_topic}]"
    for artifact_data in artifacts.values():
        all_chunks_text = artifact_data.get('chunks', {}).get('chunks', [])
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        for i, chunk_text in enumerate(all_chunks_text):
            if tag_to_find in chunk_text:
                company_name = chunk_map[i].get("company_name")
                if company_name:
                    found_companies.add(company_name)
    return found_companies

def expand_search_terms(base_term: str, kb: dict) -> list[str]:
    # ... (código sem alterações)
    return list(expanded_terms)


# --- Ferramenta de Listagem (do passo anterior, sem alterações) ---

def find_companies_by_topic(topic: str, artifacts: dict, model: SentenceTransformer, kb: dict, top_k: int = 20) -> list[str]:
    # ... (código sem alterações)
    return final_companies


# --- NOVAS FERRAMENTAS PARA ANÁLISE TEMÁTICA ---

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
    Ferramenta de extração: Busca e resume um tópico para uma empresa específica.
    Retorna o texto do resumo ou uma mensagem de "não encontrado".
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
    Ferramenta de orquestração: Realiza uma análise temática completa de um tópico.
    """
    print(f"Iniciando análise temática para o tópico: {topic}")
    
    # 1. ENCONTRAR: Identifica as empresas relevantes
    companies_to_analyze = find_companies_by_topic(topic, artifacts, model, kb)
    
    if not companies_to_analyze:
        return f"Não foram encontradas empresas com informações suficientes sobre o tópico '{topic}' para realizar uma análise temática."
    
    print(f"Analisando o tópico '{topic}' para {len(companies_to_analyze)} empresas...")

    # 2. EXTRAIR: Busca os detalhes para cada empresa em paralelo
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

    # 3. SINTETIZAR: Gera o relatório final com base nos resumos extraídos
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
