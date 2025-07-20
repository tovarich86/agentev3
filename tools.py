# tools.py (Versão 3.1 - Corrigido e Aprimorado)

import faiss
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# --- Funções Auxiliares ---

def _create_alias_to_canonical_map(kb: dict) -> tuple[dict, dict]:
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

def _get_canonical_topic_from_text(text: str, alias_map: dict) -> str | None:
    """Encontra o primeiro tópico canônico que corresponde a um alias no texto."""
    # Itera sobre os aliases do mais longo para o mais curto para evitar correspondências parciais
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', text.lower()):
            return alias_map[alias]
    return None

def _find_companies_by_exact_tag(canonical_topic: str, artifacts: dict, kb: dict) -> set[str]:
    """Busca empresas que possuem chunks com a tag de tópico exata."""
    # Mapeia o nome canônico formatado de volta para o formato de tag (com underscore)
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
    tag_to_find = f"[topico:{topic_tag_name}]"
    
    for artifact_data in artifacts.values():
        all_chunks_text = artifact_data.get('chunks', {}).get('chunks', [])
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        for i, chunk_text in enumerate(all_chunks_text):
            # A busca agora é mais específica para o formato da tag
            if f"topico:{topic_tag_name}" in chunk_text:
                company_name = chunk_map[i].get("company_name")
                if company_name:
                    found_companies.add(company_name)
    return found_companies

# --- Ferramentas ---

def find_companies_by_topic(topic: str, artifacts: dict, model: SentenceTransformer, kb: dict, top_k: int = 20) -> list[str]:
    print(f"Buscando empresas para o tópico canônico: {topic}")
    
    _, canonical_map = _create_alias_to_canonical_map(kb)
    search_terms = canonical_map.get(topic, [topic])

    # Busca por Tag
    companies_from_tags = _find_companies_by_exact_tag(topic, artifacts, kb)
    
    # Busca Vetorial
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

    # Unificar Resultados
    final_companies = sorted(list(companies_from_tags.union(companies_from_vector)))
    print(f"Total de empresas únicas encontradas: {len(final_companies)}")
    return final_companies

# ... (outras ferramentas como analyze_topic_thematically podem ser adicionadas aqui depois) ...
