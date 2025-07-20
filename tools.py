# tools.py

import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# Esta função é uma cópia da que está em app.py, para ser usada pela ferramenta.
# Em uma refatoração maior, ela poderia ir para um arquivo de "utils".
def expand_search_terms(base_term: str, kb: dict) -> list[str]:
    base_term_lower = base_term.lower()
    expanded_terms = {base_term_lower}
    for section, topics in kb.items():
        for topic, aliases in topics.items():
            all_terms_in_group = {alias.lower() for alias in aliases} | {topic.lower().replace('_', ' ')}
            if base_term_lower in all_terms_in_group:
                expanded_terms.update(all_terms_in_group)
    return list(expanded_terms)


def find_companies_by_topic(topic: str, artifacts: dict, model: SentenceTransformer, kb: dict, top_k: int = 20) -> list[str]:
    """
    Busca por um tópico em todos os documentos e retorna uma lista de nomes de empresas únicas
    que possuem chunks relevantes sobre aquele tópico.

    Args:
        topic (str): O tópico a ser buscado (ex: "Matching", "TSR Relativo").
        artifacts (dict): O dicionário contendo os índices FAISS e os chunks.
        model (SentenceTransformer): O modelo de embedding.
        kb (dict): A knowledge base (dicionário) para expandir os termos de busca.
        top_k (int): O número de chunks a serem recuperados na busca vetorial.

    Returns:
        list[str]: Uma lista ordenada de nomes de empresas únicas.
    """
    print(f"Buscando empresas para o tópico: {topic}")
    found_companies = set()
    
    # Usa a função de expansão para criar queries de busca mais ricas
    search_terms = expand_search_terms(topic, kb)
    
    # Cria uma query de busca genérica
    search_query = f"informações e detalhes sobre {', '.join(search_terms)}"
    query_embedding = model.encode([search_query], normalize_embeddings=True).astype('float32')
    
    # Itera sobre todas as categorias de documentos no artefato
    for doc_type, artifact_data in artifacts.items():
        index = artifact_data.get('index')
        chunk_map = artifact_data.get('chunks', {}).get('map', [])
        
        if not index or not chunk_map:
            continue
            
        # Realiza a busca vetorial
        scores, indices = index.search(query_embedding, top_k)
        
        # Processa os resultados
        for idx in indices[0]:
            if idx != -1: # FAISS retorna -1 para resultados não encontrados
                company_name = chunk_map[idx].get("company_name")
                if company_name:
                    found_companies.add(company_name)

    print(f"Empresas encontradas: {sorted(list(found_companies))}")
    return sorted(list(found_companies))
