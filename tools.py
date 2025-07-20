# tools.py (Versão 2.0 - Busca Híbrida)

import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# --- Funções Auxiliares ---

def _create_alias_to_canonical_map(kb: dict) -> tuple[dict, dict]:
    """Cria um mapeamento de qualquer alias para seu tópico canônico e seção."""
    alias_map = {}
    canonical_topics = defaultdict(list)
    for section, topics in kb.items():
        for topic_name_raw, aliases in topics.items():
            # Adiciona o próprio nome canônico à lista de aliases para busca
            all_aliases = aliases + [topic_name_raw.replace('_', ' ')]
            for alias in all_aliases:
                alias_map[alias.lower()] = topic_name_raw
            canonical_topics[topic_name_raw] = all_aliases
    return alias_map, canonical_topics

def _get_canonical_topic(term: str, alias_map: dict) -> str | None:
    """Encontra o nome do tópico canônico (ex: Matching_Coinvestimento) a partir de um termo de busca."""
    return alias_map.get(term.lower())

def _find_companies_by_exact_tag(canonical_topic: str, artifacts: dict) -> set[str]:
    """Busca empresas que possuem chunks com a tag de tópico exata."""
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
    """Mantida para compatibilidade, mas a nova lógica usa o mapa canônico."""
    base_term_lower = base_term.lower()
    expanded_terms = {base_term_lower}
    for section, topics in kb.items():
        for topic, aliases in topics.items():
            all_terms_in_group = {alias.lower() for alias in aliases} | {topic.lower().replace('_', ' ')}
            if base_term_lower in all_terms_in_group:
                expanded_terms.update(all_terms_in_group)
    return list(expanded_terms)


# --- Ferramenta Principal (Lógica Aprimorada) ---

def find_companies_by_topic(topic: str, artifacts: dict, model: SentenceTransformer, kb: dict, top_k: int = 20) -> list[str]:
    """
    Busca por um tópico em todos os documentos usando uma ESTRATÉGIA HÍBRIDA e retorna
    uma lista de nomes de empresas únicas que possuem chunks relevantes.
    """
    print(f"Buscando empresas para o tópico (termo original): {topic}")
    
    # 1. Mapear o termo do usuário para o tópico canônico do dicionário
    alias_map, canonical_map = _create_alias_to_canonical_map(kb)
    canonical_topic = _get_canonical_topic(topic, alias_map)
    
    if not canonical_topic:
        print(f"Tópico '{topic}' não encontrado no dicionário. Realizando apenas busca vetorial.")
        # Se não acharmos um tópico, fazemos a busca vetorial como antes
        search_terms = expand_search_terms(topic, kb)
    else:
        print(f"Tópico canônico identificado: {canonical_topic}")
        # Se achamos, usamos todos os aliases para uma busca vetorial mais rica
        search_terms = canonical_map[canonical_topic]

    # --- ESTRATÉGIA DE BUSCA HÍBRIDA ---
    
    # 2. Busca por Tag (Alta Precisão)
    companies_from_tags = set()
    if canonical_topic:
        companies_from_tags = _find_companies_by_exact_tag(canonical_topic, artifacts)
        print(f"Encontradas {len(companies_from_tags)} empresas via busca por tag: {companies_from_tags}")

    # 3. Busca Vetorial (Alto Alcance)
    companies_from_vector = set()
    search_query = f"informações e detalhes sobre {', '.join(search_terms)}"
    query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
    
    for doc_type, artifact_data in artifacts.items():
        index = artifact_data.get('index')
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        
        if not index or not chunk_map:
            continue
            
        scores, indices = index.search(query_embedding, top_k)
        
        for idx in indices[0]:
            if idx != -1:
                company_name = chunk_map[idx].get("company_name")
                if company_name:
                    companies_from_vector.add(company_name)
    print(f"Encontradas {len(companies_from_vector)} empresas via busca vetorial.")

    # 4. Unificar Resultados
    final_companies = sorted(list(companies_from_tags.union(companies_from_vector)))
    print(f"Total de empresas únicas encontradas: {len(final_companies)}")
    
    return final_companies
