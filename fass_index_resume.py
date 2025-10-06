# 1. Força a atualização do torch e transformers para as versões mais recentes e seguras
!pip install --upgrade torch torchvision torchaudio "transformers>=4.42.0" "sentence-transformers>=3.0.0"

# 2. Instala a versão da GPU do FAISS a partir do repositório confiável da NVIDIA
!pip install faiss-gpu-cu12 --pre -f https://pypi.nvidia.com

DICIONARIO_UNIFICADO_HIERARQUICO = {
    "FormularioReferencia_Item_8_4": {
        "a_TermosGerais": {"aliases": ["termos e condições gerais", "objetivos do plano", "elegíveis", "principais regras"], "subtopicos": {}},
        "b_Aprovacao": {"aliases": ["data de aprovação", "órgão responsável", "assembleia geral"], "subtopicos": {}},
        "c_MaximoAcoes": {"aliases": ["número máximo de ações abrangidas", "diluição máxima"], "subtopicos": {}},
        "d_MaximoOpcoes": {"aliases": ["número máximo de opções a serem outorgadas", "limite de opções"], "subtopicos": {}},
        "e_CondicoesAquisicao": {"aliases": ["condições de aquisição de ações", "metas de desempenho", "tempo de serviço"], "subtopicos": {}},
        "f_CriteriosPreco": {"aliases": ["critérios para fixação do preço de aquisição", "preço de exercício", "preço fixo previamente estabelecido"], "subtopicos": {}},
        "g_CriteriosPrazo": {"aliases": ["critérios para fixação do prazo de aquisição", "prazo de exercício"], "subtopicos": {}},
        "h_FormaLiquidacao": {"aliases": ["forma de liquidação", "pagamento em dinheiro", "entrega física das ações", "entrega de ações"], "subtopicos": {}},
        "i_RestricoesTransferencia": {"aliases": ["restrições à transferência", "períodos de bloqueio", "lockup", "bloqueio", "período de restrição à negociação"], "subtopicos": {}},
        "j_SuspensaoExtincao": {"aliases": ["suspensão, alteração ou extinção do plano", "mudanças nas políticas"], "subtopicos": {}},
        "k_EfeitosSaida": {"aliases": ["efeitos da saída do administrador", "regras de desligamento", "aposentadoria", "demissão"], "subtopicos": {}},
    },
    "TiposDePlano": {
        "AcoesRestritas": {
            "aliases": ["Ações Restritas", "Restricted Shares", "RSU"],
            "subtopicos": {
                "PerformanceShares": {
                    "aliases": ["Performance Shares", "PSU", "Ações de Performance"],
                    "subtopicos": {}
                }
            }
        },
        "OpcoesDeCompra": {
            "aliases": ["Opções de Compra", "Stock Options", "ESOP", "SOP"],
            "subtopicos": {}
        },
        "PlanoCompraAcoes_ESPP": {
            "aliases": ["Plano de Compra de Ações", "Employee Stock Purchase Plan", "ESPP"],
            "subtopicos": {
                "Matching_Coinvestimento": {
                    "aliases": ["Matching", "Contrapartida", "Co-investimento", "Plano de Matching"],
                    "subtopicos": {}
                }
            }
        },
        "AcoesFantasmas": {
            "aliases": ["Ações Fantasmas", "Phantom Shares", "Ações Virtuais"],
            "subtopicos": {}
        },
        "OpcoesFantasmas_SAR": {
            "aliases": ["Opções Fantasmas", "Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
            "subtopicos": {}
        },
        "BonusRetencaoDiferido": {
            "aliases": ["Bônus de Retenção", "Bônus de Permanência", "Staying Bonus", "Retention Bonus", "Deferred Bonus"],
            "subtopicos": {}
        }
    },
    "MecanicasCicloDeVida": {
        "Outorga": {"aliases": ["Outorga", "Concessão", "Grant", "Grant Date"], "subtopicos": {}},
        "Vesting": {"aliases": ["Vesting", "Período de Carência", "Aquisição de Direitos", "cronograma de vesting", "Vesting Gradual"], "subtopicos": {}},
        "VestingCliff": {"aliases": ["Cliff", "Cliff Period", "Período de Cliff", "Carência Inicial"], "subtopicos": {}},
        "VestingAcelerado": {"aliases": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação do vesting"], "subtopicos": {}},
        "VestingTranche": {"aliases": ["Tranche", "Lote", "Parcela do Vesting", 'Parcela', "Aniversário"], "subtopicos": {}},
        "PrecoExercicio": {"aliases": ["Preço de Exercício", "Strike", "Strike Price"], "subtopicos": {}},
        "PrecoDesconto": {"aliases": ["Desconto de", "preço com desconto", "desconto sobre o preço"], "subtopicos": {}},
        "CicloExercicio": {"aliases": ["Exercício", "Período de Exercício", "pagamento", "liquidação", "vencimento", "expiração"], "subtopicos": {}},
        "Lockup": {"aliases": ["Lockup", "Período de Lockup", "Restrição de Venda"], "subtopicos": {}},
    },
    "GovernancaRisco": {
        "DocumentosPlano": {"aliases": ["Regulamento", "Regulamento do Plano", "Contrato de Adesão", "Termo de Outorga"], "subtopicos": {}},
        "OrgaoDeliberativo": {"aliases": ["Comitê de Remuneração", "Comitê de Pessoas", "Deliberação do Conselho", "Conselho de Administração"], "subtopicos": {}},
        "MalusClawback": {"aliases": ["Malus", "Clawback", "Cláusula de Recuperação", "Forfeiture", "SOG", "Stock Ownership Guidelines"], "subtopicos": {}},
        "Diluicao": {"aliases": ["Diluição", "Dilution", "Capital Social", "Fator de Diluição"], "subtopicos": {}},
        "NaoConcorrencia": {
            "aliases": ["Non-Compete", "Não-Competição", "Garden-leave", "obrigação de não concorrer", "proibição de competição"],
            "subtopicos": {}
        }
    },
    "ParticipantesCondicoes": {
        "Elegibilidade": {"aliases": ["Participantes", "Beneficiários", "Elegíveis", "Empregados", "Administradores", "Colaboradores", "Executivos", "Diretores", "Gerentes", "Conselheiros"], "subtopicos": {}},
        "CondicaoSaida": {"aliases": ["Desligamento", "Saída", "Término do Contrato", "Rescisão", "Demissão", "Good Leaver", "Bad Leaver"], "subtopicos": {}},
        "CasosEspeciais": {"aliases": ["Aposentadoria", "Morte", "Invalidez", "Afastamento"], "subtopicos": {}},
    },
    "IndicadoresPerformance": {
        "ConceitoGeral_Performance": {
            "aliases": ["Plano de Desempenho", "Metas de Performance", "critérios de desempenho", "metas", "indicadores de performance", "metas", "performance"],
            "subtopicos": {
                "Financeiro": {
                    "aliases": [
                        "ROIC", "EBITDA", "LAIR", "Lucro", "CAGR", "Receita Líquida",
                        "fluxo de caixa", "geração de caixa", "Free Cash Flow", "FCF",
                        "lucros por ação", "Earnings per Share", "EPS", "redução de dívida",
                        "Dívida Líquida / EBITDA", "capital de giro", "retorno sobre investimentos",
                        "retorno sobre capital", "Return on Investment", "ROCE",
                        "margem bruta", "margem operacional", "lucro líquido",
                        "lucro operacional", "receita operacional", "vendas líquidas",
                        "valor econômico agregado", "custo de capital", "WACC",
                        "Weighted Average Capital Cost", "retorno sobre ativo",
                        "retorno sobre ativo líquido", "rotatividade de ativos líquidos",
                        "rotatividade do estoque", "despesas de capital", "dívida financeira bruta",
                        "receita operacional líquida", "lucros por ação diluídos",
                        "lucros por ação básicos", "rentabilidade", "Enterprise Value", "EV",
                        "Valor Teórico da Companhia", "Valor Teórico Unitário da Ação",
                        "Economic Value Added", "EVA", "NOPAT", "Net Operating Profit After Tax",
                        "Capital Total Investido", "CAGR EBITDA per Share", "Equity Value"
                    ],
                    "subtopicos": {}
                },
                "Mercado": {
                    "aliases": ["CDI", "IPCA", "Selic"],
                    "subtopicos": {}
                },
                "TSR": {
                    "aliases": ["TSR", "Total Shareholder Return", "Retorno Total ao Acionista"],
                    "subtopicos": {
                        "TSR_Absoluto": {"aliases": ["TSR Absoluto"], "subtopicos": {}},
                        "TSR_Relativo": {"aliases": ["TSR Relativo", "Relative TSR", "TSR versus", "TSR comparado a"], "subtopicos": {}}
                    }
                },
                "ESG": {
                    "aliases": ["Metas ESG", "ESG", "Neutralização de Emissões", "Redução de Emissões", "Igualdade de Gênero", "objetivos de desenvolvimento sustentável", "IAGEE", "ICMA"],
                    "subtopicos": {}
                },
                "Operacional": {
                    "aliases": ["produtividade", "eficiência operacional", "desempenho de entrega", "desempenho de segurança", "qualidade", "satisfação do cliente", "NPS", "conclusão de aquisições", "expansão comercial", "crescimento"],
                    "subtopicos": {}
                }
            }
        },
        "GrupoDeComparacao": {"aliases": ["Peer Group", "Empresas Comparáveis", "Companhias Comparáveis"], "subtopicos": {}}
    },
    "EventosFinanceiros": {
        "EventosCorporativos": {"aliases": ["grupamento", "desdobramento", "cisão", "fusão", "incorporação", "bonificação"], "subtopicos": {}},
        "MudancaDeControle": {"aliases": ["Mudança de Controle", "Change of Control", "Transferência de Controle"], "subtopicos": {}},
        "DividendosProventos": {"aliases": ["Dividendos", "JCP", "Juros sobre capital próprio", "dividend equivalent", "proventos"], "subtopicos": {}},
        "EventosDeLiquidez": {
            "aliases": ["Evento de Liquidez", "liquidação antecipada", "saída da companhia", "transação de controle", "reorganização societária", "desinvestimento", "deslistagem", "Operação Relevante"],
            "subtopicos": {
                "IPO_OPI": {"aliases": ["IPO", "Oferta Pública Inicial", "Oferta Publica Inicial", "abertura do capital"], "subtopicos": {}},
                "AlienacaoControle": {"aliases": ["Alienação de Controle", "alienação de mais de 50% de ações ordinárias", "venda de controle", "transferência de controle acionário", "venda ou permuta de Ações"], "subtopicos": {}},
                "FusaoAquisicaoVenda": {"aliases": ["Fusão", "Aquisição", "Incorporação", "Venda da Companhia", "venda, locação, arrendamento, cessão, licenciamento, transferência ou qualquer outra forma de disposição da totalidade ou de parte substancial dos ativos"], "subtopicos": {}},
                "InvestimentoRelevante": {"aliases": ["investimento primário de terceiros", "aumento de capital", "Capitalização da Companhia"], "subtopicos": {}}
            }
        }
    },
    "AspectosFiscaisContabeis": {
        "TributacaoEncargos": {"aliases": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"], "subtopicos": {}},
        "NormasContabeis": {"aliases": ["IFRS 2", "CPC 10", "Valor Justo", "Fair Value", "Black-Scholes", "Despesa Contábil", "Volatilidade"], "subtopicos": {}},
    }
}

# enrich_and_overwrite.py
#
# PASSO 1 (Revisado) do pipeline de processamento.
# Este script lê o arquivo de dados principal, enriquece-o com metadados
# de 'setor' e 'controle_acionario' e SOBRESCREVE o arquivo original
# com a versão enriquecida, mantendo o nome do arquivo para os próximos passos.

import json
import re
import logging
import pandas as pd
from pathlib import Path

# --- Configurações ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

PROJECT_DRIVE_PATH = Path('/content/drive/MyDrive/Projeto_CVM_RAG')

# ATENÇÃO: O arquivo de entrada e saída são os mesmos para sobrescrita.
MAIN_DATA_FILE = PROJECT_DRIVE_PATH / "cvm_documentos_texto_final.json"
CATALOGO_EMPRESAS_CSV = PROJECT_DRIVE_PATH / "Identificação empresas.csv"

# --- FUNÇÕES ---

def normalize_company_name(name: str) -> str:
    """Normaliza o nome da empresa para uma chave de consulta consistente."""
    if not isinstance(name, str): return ""
    name = re.sub(r'\s+S\.?/?A\.?$', '', name.upper().strip(), flags=re.IGNORECASE)
    return name

def create_enrichment_catalog(csv_path: str) -> dict:
    """Cria um catálogo a partir do CSV para o enriquecimento dos dados."""
    try:
        df = pd.read_csv(csv_path, sep=';')
        catalog = {}
        for _, row in df.iterrows():
            nome_atual = row['Nome_Empresarial'].upper().strip()
            company_data = {
                "harmonized_name": nome_atual,
                "setor": row['Setor_Atividade'],
                "controle_acionario": row['Especie_Controle_Acionario']
            }
            key_atual = normalize_company_name(nome_atual)
            if key_atual:
                catalog[key_atual] = company_data
            nome_anterior = row['Nome_Empresarial_Anterior']
            if isinstance(nome_anterior, str) and nome_anterior.strip():
                key_anterior = normalize_company_name(nome_anterior)
                if key_anterior:
                    catalog[key_anterior] = company_data
        logging.info(f"Catálogo de enriquecimento criado com {len(catalog)} chaves.")
        return catalog
    except Exception as e:
        logging.error(f"Erro ao criar o catálogo de enriquecimento: {e}")
        return {}

# --- FUNÇÃO PRINCIPAL ---

def enrich_and_overwrite_documents(data_file_path, catalog):
    """
    Lê o JSON de documentos, enriquece os dados com metadados e
    sobrescreve o arquivo original.
    """
    try:
        logging.info(f"Lendo o arquivo de documentos de entrada: {data_file_path}")
        with open(data_file_path, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)

        enriched_data = {}
        enriched_count = 0
        not_found_count = 0

        for url, doc_data in documents_data.items():
            original_name = doc_data.get("company_name", "")
            if not original_name:
                enriched_data[url] = doc_data
                continue

            normalized_name = normalize_company_name(original_name)
            company_info = catalog.get(normalized_name)

            if company_info:
                doc_data["company_name"] = company_info["harmonized_name"]
                doc_data["setor"] = company_info["setor"]
                doc_data["controle_acionario"] = company_info["controle_acionario"]
                enriched_count += 1
            else:
                doc_data["company_name"] = original_name.upper().strip()
                doc_data["setor"] = "Não Identificado"
                doc_data["controle_acionario"] = "Não Identificado"
                not_found_count += 1

            enriched_data[url] = doc_data

        logging.info(f"Processamento concluído. Documentos enriquecidos: {enriched_count}. Não encontrados: {not_found_count}.")

        logging.info(f"Sobrescrevendo o arquivo com dados enriquecidos: {data_file_path}")
        with open(data_file_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, ensure_ascii=False, indent=4)

        logging.info("Arquivo de dados mestre enriquecido e salvo com sucesso!")

    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)


# --- BLOCO DE EXECUÇÃO ---
if __name__ == "__main__":
    if not MAIN_DATA_FILE.exists() or not CATALOGO_EMPRESAS_CSV.exists():
        logging.error("Erro: Um ou mais arquivos de entrada não foram encontrados. Verifique os caminhos.")
    else:
        # Passo 1: Criar o catálogo de enriquecimento a partir do CSV
        enrichment_catalog = create_enrichment_catalog(CATALOGO_EMPRESAS_CSV)

        if enrichment_catalog:
            # Passo 2: Aplicar o catálogo para harmonizar e enriquecer, sobrescrevendo o arquivo principal
            enrich_and_overwrite_documents(MAIN_DATA_FILE, enrichment_catalog)


# FASE 1 (v6.0 - Otimização com Data e Hierarquia)
#
# PASSO 3 (Final) do pipeline de processamento.
# Este script lê o arquivo de dados JÁ ENRIQUECIDO e gera os índices FAISS
# e os arquivos de chunks para o agente RAG, com metadados de data e hierarquia.

import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import torch
from pathlib import Path
from collections import defaultdict

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Configurações ---
PROJECT_DRIVE_PATH = Path('/content/drive/MyDrive/Projeto_CVM_RAG')
PROJECT_DRIVE_PATH.mkdir(parents=True, exist_ok=True)

INPUT_JSON_FILE = PROJECT_DRIVE_PATH / "cvm_documentos_texto_final.json"

# --- RECOMENDAÇÃO DE MODELO ---
# 'paraphrase-multilingual-mpnet-base-v2' é um ótimo ponto de partida.
# Para maior precisão em português, considere testar 'neuralmind/bert-base-portuguese-cased'.
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE = 32 # Ajuste conforme a memória da sua GPU

# Assumindo que DICIONARIO_UNIFICADO_HIERARQUICO está carregado.

# --- FUNÇÕES AUXILIARES (INALTERADAS) ---

def classify_document_by_url(url):
    """Classifica o tipo de documento com base em padrões na URL."""
    if 'frmExibirArquivoFRE' in url:
        return 'item_8_4'
    return 'outros_documentos'

def split_text_semantic(text, max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide o texto em chunks, tentando respeitar os parágrafos."""
    if not isinstance(text, str) or not text: return []
    paragraphs = text.split('\n\n')
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph: continue
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk: chunks.append(current_chunk.strip())
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            start = 0
            while start < len(chunk):
                final_chunks.append(chunk[start:start + max_chunk_size])
                start += max_chunk_size - overlap
        else:
            final_chunks.append(chunk)
    return final_chunks

# --- FUNÇÕES PARA ENRIQUECIMENTO (ADICIONADAS OU MODIFICADAS) ---

def _recursive_topic_finder(text, sub_dict, path_so_far, found_items):
    """Função auxiliar recursiva para encontrar tópicos em dicionários aninhados."""
    for topic_key, topic_data in sub_dict.items():
        current_path = path_so_far + [topic_key]
        # Ordena para garantir que aliases mais longos (e.g., "Opções de Compra") sejam checados antes dos mais curtos ("Compra")
        sorted_aliases = sorted(topic_data.get("aliases", []), key=len, reverse=True)
        for alias in sorted_aliases:
            if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
                found_items.add(tuple(current_path))
                break # Encontrou o tópico, vai para o próximo do dicionário
        if "subtopicos" in topic_data and topic_data["subtopicos"]:
            _recursive_topic_finder(text, topic_data["subtopicos"], current_path, found_items)

def find_topics_recursive(text, dictionary_hierarchical):
    """Inicia a busca recursiva por tópicos hierárquicos em um texto."""
    if not isinstance(dictionary_hierarchical, dict): return []
    found_paths = set()
    for section, topics in dictionary_hierarchical.items():
        _recursive_topic_finder(text, topics, [section], found_paths)
    # Retorna os caminhos para serem processados depois
    return [list(path) for path in found_paths]

def extract_date_from_url(url: str) -> str:
    """Extrai a data de referência da URL, se possível."""
    match = re.search(r'data=(.+?)&', url)
    if match:
        return match.group(1).strip()
    return "N/A"

# --- FUNÇÃO PRINCIPAL (ATUALIZADA E OTIMIZADA) ---

def process_and_create_indices(all_docs, embedding_model, dictionary_to_use, model_name_str):
    """
    Orquestra a criação de chunks, embeddings e índices FAISS.
    (v6.0) - Enriquecimento com DATA DE REFERÊNCIA e HIERARQUIA DE TÓPICOS.
    """
    categorized_docs = defaultdict(dict)
    for source_url, doc_data in all_docs.items():
        category = classify_document_by_url(source_url)
        if isinstance(doc_data, dict):
            categorized_docs[category][source_url] = doc_data

    logging.info("--- Iniciando criação de índices (v6.0 - Otimização com Data e Hierarquia) ---")
    for category, docs in categorized_docs.items():
        logging.info(f"--- Processando categoria: '{category}' ({len(docs)} documentos) ---")
        all_chunks, chunk_map = [], []

        for source_url, doc_data in docs.items():
            doc_text = doc_data.get("text", "")
            company_name = doc_data.get("company_name", "N/A")
            setor = doc_data.get("setor", "Não Identificado")
            controle = doc_data.get("controle_acionario", "Não Identificado")
            document_date = doc_data.get("data_referencia", "Não Identificado")

            if not doc_text: continue

            doc_chunks = split_text_semantic(doc_text)
            for chunk_index, chunk in enumerate(doc_chunks):

                # --- Construção do Prefixo de Metadados Otimizado ---

                # 1. Metadados de Data e Categoria
                metadata_prefix_date = f"[data_documento:{document_date}]"
                metadata_prefix_cat = f"[empresa:{company_name}] [setor:{setor}] [controle:{controle}]"

                # 2. Metadados de Tópicos (com hierarquia)
                topic_paths_as_lists = find_topics_recursive(chunk, dictionary_to_use)
                topic_paths_as_strings = ["/".join(path) for path in topic_paths_as_lists]
                topic_tags_list = []
                for path in topic_paths_as_lists:
                    for i, level_name in enumerate(path):
                        # ✨ OTIMIZAÇÃO 2: Criar tag para cada nível da hierarquia
                        topic_tags_list.append(f"[cat_nv{i+1}:{level_name}]")

                # Remove duplicatas mantendo a ordem (importante para hierarquia)
                unique_topic_tags = list(dict.fromkeys(topic_tags_list))
                metadata_prefix_topics = " ".join(unique_topic_tags)

                # 3. Combinação para o Chunk Final
                full_prefix = f"{metadata_prefix_date} {metadata_prefix_cat} {metadata_prefix_topics}".strip()
                enriched_chunk = f"{full_prefix} {chunk}"
                all_chunks.append(enriched_chunk)

                # --- Adicionar metadados completos ao chunk_map ---
                chunk_map.append({
                    "company_name": company_name,
                    "source_url": source_url,
                    "chunk_id": f"{source_url}::{chunk_index}",
                    "document_date": document_date, # 🔑 Data adicionada para re-ranking
                    "setor": setor,
                    "controle_acionario": controle,
                    "topics_in_chunk": topic_paths_as_strings, # Mantém caminhos para análise
                    "chunk_text": chunk # Armazena o texto original sem metadados
                })

        if not all_chunks:
            logging.warning(f"Nenhum chunk válido foi gerado para a categoria '{category}'. Pulando.")
            continue

        logging.info(f"-> {len(all_chunks)} chunks únicos criados para a categoria '{category}'.")
        logging.info("-> Gerando embeddings...")
        embeddings = embedding_model.encode(
            all_chunks,
            show_progress_bar=True,
            normalize_embeddings=True, # Normalizar é crucial para busca com L2/Cosseno
            batch_size=BATCH_SIZE
        )
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]

        logging.info("-> Criando índice FAISS...")
        # IndexFlatL2 é um bom padrão. A distância L2 funciona bem com embeddings normalizados.
        index = faiss.IndexFlatL2(dimension)
        index_final = faiss.IndexIDMap(index)
        ids = np.array(range(len(all_chunks)))
        index_final.add_with_ids(embeddings, ids)
        logging.info(f"-> Índice criado com {index_final.ntotal} vetores.")

        faiss_file = PROJECT_DRIVE_PATH / f"{category}_faiss_index.bin"
        chunks_file = PROJECT_DRIVE_PATH / f"{category}_chunks_map.json"

        faiss.write_index(index_final, str(faiss_file))
        logging.info(f"-> Índice salvo em '{faiss_file}'")

        # Salva o mapa de chunks, que agora é a sua "base de dados" de metadados
        with open(str(chunks_file), 'w', encoding='utf-8') as f:
            # Note a alteração aqui, salvando apenas o chunk_map diretamente
            json.dump(chunk_map, f, ensure_ascii=False, indent=4)
        logging.info(f"-> Mapa de chunks salvo em '{chunks_file}'")


# --- BLOCO DE EXECUÇÃO (CORRIGIDO E COMPLETO) ---
if __name__ == "__main__":
    try:
        # --- CONFIGURAÇÕES PRINCIPAIS ---
        MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        PROJECT_DRIVE_PATH = Path('/content/drive/MyDrive/Projeto_CVM_RAG')
        PROJECT_DRIVE_PATH.mkdir(parents=True, exist_ok=True)
        INPUT_JSON_FILE = PROJECT_DRIVE_PATH / "cvm_documentos_texto_final.json"

        # --- DEFINA O SEU DICIONÁRIO HIERÁRQUICO AQUI ---
        # Ele precisa ser acessível globalmente ou carregado.
        DICIONARIO_UNIFICADO_HIERARQUICO = {
            "Documentos Financeiros": {
                "Demonstrativos Financeiros": {
                    "aliases": ["DFC", "Demonstração do Fluxo de Caixa", "Balanço Patrimonial", "BP"],
                    "subtopicos": {
                        "Ativo Circulante": {"aliases": ["Ativo circulante"]},
                        "Passivo Circulante": {"aliases": ["Passivo circulante"]}
                    }
                },
                "Relatório de Gestão": {
                    "aliases": ["Relatório da Administração", "Relatório de Gestão"],
                    "subtopicos": {}
                }
            },
            "Informações de Mercado": {
                "Governança Corporativa": {
                    "aliases": ["Governança Corporativa", "Conselho de Administração"],
                    "subtopicos": {}
                }
            }
        }
        # --- FIM DAS CONFIGURAÇÕES ---

        # Configuração de logging que não interfere com a barra de progresso.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if 'DICIONARIO_UNIFICADO_HIERARQUICO' not in globals():
            raise NameError("A variável 'DICIONARIO_UNIFICADO_HIERARQUICO' não foi definida.")

        if not INPUT_JSON_FILE.exists():
            raise FileNotFoundError(f"Arquivo de entrada enriquecido não encontrado: {INPUT_JSON_FILE}")

        logging.info(f"Lendo arquivo JSON enriquecido: {INPUT_JSON_FILE}")
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Usando dispositivo: {device} para geração de embeddings.")

        logging.info(f"Carregando modelo de embedding '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME, device=device)

        # Chamada correta da função, passando MODEL_NAME como o 4º argumento.
        process_and_create_indices(documents_data, model, DICIONARIO_UNIFICADO_HIERARQUICO, MODEL_NAME)

        logging.info(f"--- Processamento de índices com  concluído com sucesso! ---")

    except (FileNotFoundError, NameError) as e:
        logging.error(f"ERRO DE CONFIGURAÇÃO OU ARQUIVO: {e}")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)

# extracao_logica_unificada.py (Versão Otimizada)
#
# DESCRIÇÃO:
# Versão ajustada e otimizada da lógica de extração de tópicos.
# A alteração principal corrige um gargalo de performance ao garantir que a
# limpeza do texto de entrada seja feita apenas uma vez, antes do início da
# busca recursiva. A lógica de detecção e os resultados permanecem idênticos,
# mas a execução é significativamente mais rápida e robusta.

import re
import unicodedata
from typing import List, Set, Tuple, Dict, Any

def normalizar_texto(texto: str) -> str:
    """
    (Técnica 1) Converte para minúsculas e remove acentos para uma comparação robusta.
    """
    if not isinstance(texto, str):
        return ""
    # Converte para minúsculas
    texto = texto.lower()
    # Remove acentos (decomposição NFD)
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# --- FUNÇÃO RECURSIVA OTIMIZADA ---
def _recursive_topic_finder_unificada(
    texto_limpo_para_busca: str,  # AJUSTE 1: Recebe o texto já limpo
    sub_dict: Dict[str, Any],
    path_so_far: List[str],
    found_items: Set[Tuple[Tuple[str, ...], str]]
):
    """
    Função auxiliar recursiva que navega pelo dicionário.
    Esta versão é otimizada para não reprocessar o texto.
    """
    # A linha que limpava o texto foi REMOVIDA daqui, evitando reprocessamento.

    for topic_key, topic_data in sub_dict.items():
        current_path = path_so_far + [topic_key]

        # Ordena os aliases do maior para o menor para evitar correspondências parciais
        sorted_aliases = sorted(topic_data.get("aliases", []), key=len, reverse=True)

        for alias in sorted_aliases:
            alias_normalizado = normalizar_texto(alias)

            # A busca continua usando o texto limpo e a lógica de palavra inteira (\b)
            if re.search(r'\b' + re.escape(alias_normalizado) + r'\b', texto_limpo_para_busca):
                found_items.add((tuple(current_path), alias))

        if "subtopicos" in topic_data and topic_data["subtopicos"]:
            # AJUSTE 2: Passa o mesmo texto já limpo para a próxima chamada recursiva
            _recursive_topic_finder_unificada(texto_limpo_para_busca, topic_data["subtopicos"], current_path, found_items)

# --- FUNÇÃO PRINCIPAL OTIMIZADA ---
def encontrar_topicos_unificado(texto_original: str, dicionario_hierarquico: Dict) -> List[Tuple[Tuple[str, ...], str]]:
    """
    Função principal e unificada para encontrar tópicos hierárquicos e seus aliases.
    Esta é a função que seus outros scripts devem chamar.
    """
    if not isinstance(texto_original, str) or not isinstance(dicionario_hierarquico, dict):
        return []

    # --- OTIMIZAÇÃO APLICADA AQUI ---
    # Etapa 1: Normalização (minúsculas, sem acentos)
    texto_normalizado = normalizar_texto(texto_original)
    # Etapa 2: Limpeza de pontuação (feita apenas UMA VEZ para todo o texto)
    # Isso resolve o problema de encontrar termos como "(ROIC)" ou "TSR)".
    texto_limpo_para_busca = re.sub(r'[^\w\s]', ' ', texto_normalizado)

    found_items = set()
    # Itera sobre as seções do dicionário (ex: "TiposDePlano", "IndicadoresPerformance")
    for section, topics in dicionario_hierarquico.items():
        # AJUSTE 3: Passa o texto já limpo e pré-processado para a busca recursiva
        _recursive_topic_finder_unificada(texto_limpo_para_busca, topics, [section], found_items)

    return list(found_items)

# gerar_faiss_e_chunks_v7.py
#
# PASSO 3 (v7.1 - Versão para Colab com Dicionário Externo)
# Este script lê o arquivo de dados ENRIQUECIDO, gera os índices FAISS
# e os arquivos de chunks para o agente RAG, utilizando uma lógica de
# extração de tópicos centralizada e aprimorada.

import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import torch
from pathlib import Path
from collections import defaultdict

# A lógica unificada é importada da célula/arquivo executado anteriormente
# from extracao_logica_unificada import encontrar_topicos_unificado

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Configurações do Projeto ---
PROJECT_DRIVE_PATH = Path('/content/drive/MyDrive/Projeto_CVM_RAG')
PROJECT_DRIVE_PATH.mkdir(parents=True, exist_ok=True)

INPUT_JSON_FILE = PROJECT_DRIVE_PATH / "cvm_documentos_texto_final.json"
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE = 32 # Ajuste conforme a memória da sua GPU

# --- FUNÇÕES AUXILIARES ---

def classify_document_by_url(url):
    """Classifica o tipo de documento com base em padrões na URL."""
    if 'frmExibirArquivoFRE' in url:
        return 'item_8_4'
    return 'outros_documentos'

def split_text_semantic(text, max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide o texto em chunks, tentando respeitar os parágrafos."""
    if not isinstance(text, str) or not text: return []
    paragraphs = text.split('\n\n')
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph: continue
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk: chunks.append(current_chunk.strip())

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            start = 0
            while start < len(chunk):
                final_chunks.append(chunk[start:start + max_chunk_size])
                start += max_chunk_size - overlap
        else:
            final_chunks.append(chunk)
    return final_chunks

# --- FUNÇÃO PRINCIPAL ---

def process_and_create_indices(all_docs, embedding_model, dictionary_to_use):
    """
    Orquestra a criação de chunks, embeddings e índices FAISS.
    """
    categorized_docs = defaultdict(dict)
    for source_url, doc_data in all_docs.items():
        category = classify_document_by_url(source_url)
        if isinstance(doc_data, dict):
            categorized_docs[category][source_url] = doc_data

    logging.info("--- Iniciando criação de índices (v7.1 - Lógica Unificada) ---")
    for category, docs in categorized_docs.items():
        logging.info(f"--- Processando categoria: '{category}' ({len(docs)} documentos) ---")
        all_chunks, chunk_map = [], []

        for source_url, doc_data in docs.items():
            doc_text = doc_data.get("text", "")
            company_name = doc_data.get("company_name", "N/A")
            setor = doc_data.get("setor", "Não Identificado")
            controle = doc_data.get("controle_acionario", "Não Identificado")
            document_date = doc_data.get("data_referencia", "Não Identificado")

            if not doc_text: continue

            doc_chunks = split_text_semantic(doc_text)
            for chunk_index, chunk in enumerate(doc_chunks):

                # 1. Metadados de Data e Categoria
                metadata_prefix_date = f"[data_documento:{document_date}]"
                cat_parts = []
                if company_name and company_name != "N/A":
                    cat_parts.append(f"[empresa:{company_name}]")
                if setor and setor not in ["Não Identificado", "NaN"]:
                    cat_parts.append(f"[setor:{setor}]")
                if controle and controle not in ["Não Identificado", "NaN"]:
                    cat_parts.append(f"[controle:{controle}]")
                metadata_prefix_cat = " ".join(cat_parts)

                # 2. Metadados de Tópicos (com hierarquia) via LÓGICA UNIFICADA
                topic_path_alias_tuples = encontrar_topicos_unificado(chunk, dictionary_to_use)
                topic_paths_as_lists = [list(path) for path, alias in topic_path_alias_tuples]
                topic_paths_as_strings = sorted(list(set(["/".join(path) for path in topic_paths_as_lists])))

                topic_tags_list = []
                for path in topic_paths_as_lists:
                    for i, level_name in enumerate(path):
                        topic_tags_list.append(f"[cat_nv{i+1}:{level_name}]")

                unique_topic_tags = sorted(list(dict.fromkeys(topic_tags_list)))
                metadata_prefix_topics = " ".join(unique_topic_tags)

                # 3. Combinação para o Chunk Final
                full_prefix = f"{metadata_prefix_date} {metadata_prefix_cat} {metadata_prefix_topics}".strip()
                enriched_chunk = f"{full_prefix} {chunk}"
                all_chunks.append(enriched_chunk)

                # 4. Adicionar metadados completos ao chunk_map
                chunk_map.append({
                    "company_name": company_name,
                    "source_url": source_url,
                    "chunk_id": f"{source_url}::{chunk_index}",
                    "document_date": document_date,
                    "setor": setor,
                    "controle_acionario": controle,
                    "topics_in_chunk": topic_paths_as_strings,
                    "chunk_text": chunk
                })

        if not all_chunks:
            logging.warning(f"Nenhum chunk válido foi gerado para a categoria '{category}'. Pulando.")
            continue

        logging.info(f"-> {len(all_chunks)} chunks únicos criados para a categoria '{category}'.")
        logging.info("-> Gerando embeddings...")
        embeddings = embedding_model.encode(
            all_chunks, show_progress_bar=True, normalize_embeddings=True, batch_size=BATCH_SIZE
        )
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]

        logging.info("-> Criando índice FAISS...")
        index = faiss.IndexFlatL2(dimension)
        index_final = faiss.IndexIDMap(index)
        ids = np.array(range(len(all_chunks)))
        index_final.add_with_ids(embeddings, ids)
        logging.info(f"-> Índice criado com {index_final.ntotal} vetores.")

        faiss_file = PROJECT_DRIVE_PATH / f"{category}_faiss_index.bin"
        chunks_file = PROJECT_DRIVE_PATH / f"{category}_chunks_map.json"

        faiss.write_index(index_final, str(faiss_file))
        logging.info(f"-> Índice salvo em '{faiss_file}'")

        with open(str(chunks_file), 'w', encoding='utf-8') as f:
            json.dump(chunk_map, f, ensure_ascii=False, indent=4)
        logging.info(f"-> Mapa de chunks salvo em '{chunks_file}'")


# --- BLOCO DE EXECUÇÃO ---
if __name__ == "__main__":
    try:
        # Verifica se o dicionário foi definido em uma célula anterior
        if 'DICIONARIO_UNIFICADO_HIERARQUICO' not in globals() or not DICIONARIO_UNIFICADO_HIERARQUICO:
            raise NameError("A variável 'DICIONARIO_UNIFICADO_HIERARQUICO' não foi definida ou está vazia. Execute a célula do dicionário primeiro.")
        if not INPUT_JSON_FILE.exists():
            raise FileNotFoundError(f"Arquivo de entrada enriquecido não encontrado: {INPUT_JSON_FILE}")

        logging.info(f"Lendo arquivo JSON enriquecido: {INPUT_JSON_FILE}")
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Usando dispositivo: {device} para geração de embeddings.")

        logging.info(f"Carregando modelo de embedding '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME, device=device)

        process_and_create_indices(documents_data, model, DICIONARIO_UNIFICADO_HIERARQUICO)

        logging.info(f"--- Processamento de índices concluído com sucesso! ---")

    except (FileNotFoundError, NameError) as e:
        logging.error(f"ERRO DE CONFIGURAÇÃO OU ARQUIVO: {e}")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)

# gerar_resumo_v7.4_extracao_robusta.py
#
# DESCRIÇÃO:
# Versão baseada na v7.3 otimizada, com foco em garantir a robustez da extração
# de fatos (TSR, ROIC, etc.), resolvendo especificamente o problema de
# pontuações adjacentes às palavras-chave.

import json
import re
import logging
from pathlib import Path
import unicodedata
from typing import List, Set, Tuple, Dict, Any
from collections import defaultdict

# --- Configurações ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

PROJECT_DRIVE_PATH = Path('/content/drive/MyDrive/Projeto_CVM_RAG')
INPUT_JSON_FILE = PROJECT_DRIVE_PATH / "cvm_documentos_texto_final.json"
# Atualizado o nome do arquivo de saída
OUTPUT_SUMMARY_FILE = PROJECT_DRIVE_PATH / "resumo_planos_granulares_v7.4.json"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- DICIONÁRIO ---
# O script assume que a variável 'DICIONARIO_UNIFICADO_HIERARQUICO'
# já foi definida em uma célula anterior no ambiente do Colab.


# ==============================================================================
# OTIMIZAÇÃO: PRÉ-COMPILAÇÃO DAS EXPRESSÕES REGULARES
# ==============================================================================
def _compile_fact_regexes() -> Dict[str, re.Pattern]:
    """Pré-compila regex para serem reutilizadas, melhorando a performance."""
    logging.info("Compilando expressões regulares para extração de fatos...")
    num_pattern = r'(\d{1,3}(?:[.,]\d{3})*|\d+|um|uma|dois|duas|tres|três|quatro|cinco|seis|sete|oito|nove|dez)'
    unit_pattern_anomesdias = r'\b(anos?|meses|dias)\b'

    return {
        'vesting': re.compile(
            fr'(?:vesting|periodo\s+de\s+carencia|prazo\s+de\s+carencia|periodo\s+de\s+aquisicao|prazo\s+de\s+aquisicao)'
            fr'[\s\S]{{0,250}}?(?:de\s+)?{num_pattern}\s*\(?[\s\S]{{0,100}}?{unit_pattern_anomesdias}', re.IGNORECASE
        ),
        'lockup': re.compile(
            fr'(?:lock-up|periodo\s+de\s+restricao|restricao\s+a\s+venda)'
            fr'[\s\S]{{0,250}}?(?:de\s+)?{num_pattern}\s*\(?[\s\S]{{0,100}}?{unit_pattern_anomesdias}', re.IGNORECASE
        ),
        'diluicao': re.compile(
            r'(?:diluicao|limite|nao\s+exceda|representativas\s+de|no\s+maximo)'
            r'[\s\S]{0,250}?\b(\d{1,3}(?:[.,]\d{1,2})?)\s*%', re.IGNORECASE
        ),
        'desconto': re.compile(
            r'(?:desconto|desagio|abatimento|reducao)[\s\S]{0,100}?\b(\d{1,3}(?:[.,]\d{1,2})?)\s*%', re.IGNORECASE
        ),
        'malus_clawback': re.compile(r'\b(malus|clawback|clausula\s+de\s+recuperacao|forfeiture|nao\s+concorrencia)\b', re.IGNORECASE),
        'dividendos': re.compile(r'(?:dividendos|jcp|juros\s+sobre\s+capital\s+proprio|proventos)[\s\S]{0,200}?(?:durante|no\s+periodo\s+de)\s*(?:carencia|vesting)', re.IGNORECASE),
        # Padrões para TSR e ROIC, que buscam a frase completa ou a sigla.
        'tsr': re.compile(r'\b(retorno\s+total\s+d[oa]s\s+acionistas|total\s+shareholder\s+return|tsr)\b', re.IGNORECASE),
        'roic': re.compile(r'\b(retorno\s+sobre\s+o\s+capital\s+investido|return\s+on\s+invested\s+capital|roic)\b', re.IGNORECASE),
    }

REGEX_FATOS_COMPILADOS = _compile_fact_regexes()


# ==============================================================================
# FUNÇÕES AUXILIARES (LÓGICA ORIGINAL v7 com OTIMIZAÇÕES)
# ==============================================================================

def split_text_semantic(text, max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not isinstance(text, str) or not text: return []
    paragraphs = text.split('\n\n')
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph: continue
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk: chunks.append(current_chunk.strip())
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            start = 0
            while start < len(chunk):
                final_chunks.append(chunk[start:start + max_chunk_size])
                start += max_chunk_size - overlap
        else:
            final_chunks.append(chunk)
    return final_chunks

def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def _recursive_topic_finder_unificada(
    texto_limpo_para_busca: str,
    sub_dict: Dict[str, Any],
    path_so_far: List[str],
    found_items: Set[Tuple[Tuple[str, ...], str]]
):
    for topic_key, topic_data in sub_dict.items():
        current_path = path_so_far + [topic_key]
        sorted_aliases = sorted(topic_data.get("aliases", []), key=len, reverse=True)
        for alias in sorted_aliases:
            alias_normalizado = normalizar_texto(alias)
            if re.search(r'\b' + re.escape(alias_normalizado) + r'\b', texto_limpo_para_busca):
                found_items.add((tuple(current_path), alias))
        if "subtopicos" in topic_data and topic_data["subtopicos"]:
            _recursive_topic_finder_unificada(texto_limpo_para_busca, topic_data["subtopicos"], current_path, found_items)

def encontrar_topicos_unificado(texto_original: str, dicionario_hierarquico: Dict) -> List[Tuple[Tuple, str]]:
    if not isinstance(texto_original, str) or not isinstance(dicionario_hierarquico, dict): return []
    texto_normalizado = normalizar_texto(texto_original)
    texto_limpo_para_busca = re.sub(r'[^\w\s]', ' ', texto_normalizado)
    found_items = set()
    for section, topics in dicionario_hierarquico.items():
        _recursive_topic_finder_unificada(texto_limpo_para_busca, topics, [section], found_items)
    return list(found_items)

def _converter_palavra_para_int(palavra: str) -> int:
    if not isinstance(palavra, str): return 0
    palavra_limpa = normalizar_texto(palavra).strip()
    mapeamento = {'um': 1, 'uma': 1, 'dois': 2, 'duas': 2, 'tres': 3, 'quatro': 4, 'cinco': 5, 'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9, 'dez': 10}
    if palavra_limpa in mapeamento: return mapeamento[palavra_limpa]
    try:
        return int(float(palavra_limpa.replace(',', '.')))
    except (ValueError, TypeError):
        return 0

def build_nested_dict_from_paths(path_alias_tuples):
    nested_dict = {}
    for path, alias in path_alias_tuples:
        d = nested_dict
        for key in path[:-1]: d = d.setdefault(key, {})
        leaf_key = path[-1]
        leaf_node = d.setdefault(leaf_key, {})
        leaf_node.setdefault('_aliases', set()).add(alias)
    def cleanup_dict(d):
        result = {}
        for k, v in d.items():
            sub_dict = {sk: sv for sk, sv in v.items() if sk != '_aliases'}
            aliases = sorted(list(v.get('_aliases', set())))
            cleaned_sub = cleanup_dict(sub_dict)
            if cleaned_sub and aliases: result[k] = {"_aliases": aliases, **cleaned_sub}
            elif cleaned_sub: result[k] = cleaned_sub
            elif aliases: result[k] = aliases
        return result
    return cleanup_dict(nested_dict)

# --- FUNÇÃO DE EXTRAÇÃO DE FATOS ATUALIZADA ---
def extract_structured_facts(text: str) -> dict:
    """
    Extrai fatos estruturados de um texto, usando texto pré-processado para
    garantir a detecção correta mesmo com pontuação.
    """
    facts = {}
    # Etapa 1: Normalização básica (minúsculas, sem acentos)
    text_normalized = normalizar_texto(text)
    # Etapa 2: Limpeza de pontuação para busca robusta de palavras-chave
    text_limpo_para_busca = re.sub(r'[^\w\s]', ' ', text_normalized)

    # Extração de fatos quantitativos (usam o texto normalizado para preservar a estrutura)
    try:
        if match := REGEX_FATOS_COMPILADOS['vesting'].search(text_normalized):
            valor_num = _converter_palavra_para_int(match.group(1))
            unidade = match.group(2).lower().rstrip('s')
            valor_em_anos = valor_num if unidade == 'ano' else valor_num / 12.0 if unidade == 'mes' else valor_num / 365.0
            if 0 < valor_em_anos < 20: facts['periodo_vesting'] = {'presente': True, 'valor': round(valor_em_anos, 2), 'unidade': 'ano'}
    except Exception as e: logging.warning(f"Alerta na extração de Fato (Vesting): {e}")

    try:
        if match := REGEX_FATOS_COMPILADOS['lockup'].search(text_normalized):
            valor_num = _converter_palavra_para_int(match.group(1))
            unidade = match.group(2).lower().rstrip('s')
            valor_em_anos = valor_num if unidade == 'ano' else valor_num / 12.0 if unidade == 'mes' else valor_num / 365.0
            if 0 < valor_em_anos < 20: facts['periodo_lockup'] = {'presente': True, 'valor': round(valor_em_anos, 2), 'unidade': 'ano'}
    except Exception as e: logging.warning(f"Alerta na extração de Fato (Lock-up): {e}")

    try:
        if match := REGEX_FATOS_COMPILADOS['diluicao'].search(text_normalized):
            valor_percentual = float(match.group(1).replace(',', '.')) / 100.0
            if 0 < valor_percentual < 0.5: facts['diluicao_maxima_percentual'] = {'presente': True, 'valor': valor_percentual, 'unidade': 'percentual'}
    except Exception as e: logging.warning(f"Alerta na extração de Fato (Diluição Percentual): {e}")

    try:
        if match := REGEX_FATOS_COMPILADOS['desconto'].search(text_normalized):
            valor_str = match.group(1).replace(',', '.')
            facts['desconto_strike_price'] = {'presente': True, 'valor_numerico': float(valor_str) / 100, 'tipo': 'explícito'}
    except Exception as e: logging.warning(f"Alerta na extração de Fato (Desconto): {e}")

    # --- LÓGICA DE BUSCA ROBUSTA APLICADA AQUI ---
    # Fatos de presença/ausência usam o texto limpo, resolvendo o problema da pontuação.
    if REGEX_FATOS_COMPILADOS['malus_clawback'].search(text_limpo_para_busca): facts['malus_clawback_presente'] = {'presente': True}
    if REGEX_FATOS_COMPILADOS['dividendos'].search(text_limpo_para_busca): facts['dividendos_durante_carencia'] = {'presente': True}
    if REGEX_FATOS_COMPILADOS['tsr'].search(text_limpo_para_busca): facts['t_s_r_presente'] = {'presente': True}
    if REGEX_FATOS_COMPILADOS['roic'].search(text_limpo_para_busca): facts['r_o_i_c_presente'] = {'presente': True}

    return facts

# ==============================================================================
# FUNÇÃO PRINCIPAL (LÓGICA ORIGINAL v7 INTACTA)
# ==============================================================================
def generate_granular_summary_file(all_docs, dictionary_to_use):
    logging.info("--- Iniciando geração de resumo granular (v7.4 - Extração Robusta) ---")

    empresas_features = {}
    tipos_de_plano_dict = {"TiposDePlano": dictionary_to_use.get("TiposDePlano", {})}

    for i, (source_url, doc_data) in enumerate(all_docs.items()):
        company_name = doc_data.get("company_name")
        doc_text = doc_data.get("text", "")

        logging.info(f"Processando documento {i+1}/{len(all_docs)}: {company_name}")
        if not company_name or not doc_text: continue

        if company_name not in empresas_features:
            empresas_features[company_name] = {
                "setor": doc_data.get("setor", "Não Identificado"),
                "controle_acionario": doc_data.get("controle_acionario", "Não Identificado"),
                "planos_identificados": {},
                "fatos_extraidos_agregados": {}
            }

        found_plans_tuples = encontrar_topicos_unificado(doc_text, tipos_de_plano_dict)
        identified_plan_names = {path[-1] for path, alias in found_plans_tuples}
        if not identified_plan_names:
            identified_plan_names = {"PlanoGeralNaoIdentificado"}

        doc_all_topics = encontrar_topicos_unificado(doc_text, dictionary_to_use)

        doc_chunks = split_text_semantic(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
        empresas_features[company_name]["fatos_extraidos_agregados"] = {}
        for chunk in doc_chunks:
            # A chamada agora usa a função de extração atualizada
            chunk_facts = extract_structured_facts(chunk)
            if chunk_facts:
                empresas_features[company_name]["fatos_extraidos_agregados"].update(chunk_facts)

        for plan_name in identified_plan_names:
            plan_entry = empresas_features[company_name]["planos_identificados"].setdefault(
                plan_name,
                { "documentos_fonte": set(), "topicos_encontrados_com_alias": set(), "fatos_extraidos": {} }
            )
            plan_entry["documentos_fonte"].add(source_url)
            plan_entry["topicos_encontrados_com_alias"].update(doc_all_topics)
            plan_entry["fatos_extraidos"].update(empresas_features[company_name]["fatos_extraidos_agregados"])

    logging.info("Finalizando e formatando o JSON de saída...")
    final_output = {}
    for company, data in empresas_features.items():
        final_output[company] = {
            "setor": data["setor"],
            "controle_acionario": data["controle_acionario"],
            "fatos_extraidos": data["fatos_extraidos_agregados"],
            "planos_identificados": {}
        }
        for plan_name, plan_data in data["planos_identificados"].items():
            nested_topics = build_nested_dict_from_paths(plan_data["topicos_encontrados_com_alias"])
            final_output[company]["planos_identificados"][plan_name] = {
                "documentos_fonte": sorted(list(plan_data["documentos_fonte"])),
                "topicos_encontrados": nested_topics,
                "fatos_extraidos": plan_data["fatos_extraidos"]
            }

    with open(OUTPUT_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    logging.info(f"✅ Resumo granular salvo com sucesso em '{OUTPUT_SUMMARY_FILE}'")

# ==============================================================================
# BLOCO DE EXECUÇÃO
# ==============================================================================
def main():
    try:
        if 'DICIONARIO_UNIFICADO_HIERARQUICO' not in globals() or not DICIONARIO_UNIFICADO_HIERARQUICO:
            raise NameError("A variável 'DICIONARIO_UNIFICADO_HIERARQUICO' não foi definida ou está vazia. Execute a célula do dicionário primeiro.")
        if not INPUT_JSON_FILE.exists():
            raise FileNotFoundError(f"Arquivo de entrada não encontrado: {INPUT_JSON_FILE}")

        logging.info(f"Lendo arquivo JSON enriquecido: {INPUT_JSON_FILE}")
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)

        generate_granular_summary_file(documents_data, DICIONARIO_UNIFICADO_HIERARQUICO)
    except (FileNotFoundError, NameError) as e:
        logging.error(f"ERRO DE CONFIGURAÇÃO OU ARQUIVO: {e}")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)

# Para rodar, chame a função main() em uma célula do notebook após definir o dicionário.
main()