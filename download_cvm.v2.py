# -*- coding: utf-8 -*-
"""
SCRIPT UNIFICADO DE DOWNLOAD E PRÉ-PROCESSAMENTO (V7.1 - ATUALIZAÇÃO INTELIGENTE COM DATA PADRÃO FRE)

Versão aprimorada que:
1. Carrega o JSON existente para evitar reprocessamento.
2. Adiciona datas de referência a registros antigos que não as possuam.
3. Atribui uma data de referência padrão ("2025-06-30") para todos os documentos FRE.
4. Verifica e processa APENAS os novos documentos das planilhas.
"""

import pandas as pd
import requests
import re
import time
import os
import base64
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from functools import wraps
import fitz  # PyMuPDF
import logging
import json

# --- Importações do Selenium ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURAÇÕES ---
BASE_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()
CSV_URL = BASE_DIR / "fre_cia_aberta_2025_otimizado.csv"
PLANOS_URL = BASE_DIR / "tabela_consolidada_cvm_otimizado.xlsx"
OUTPUT_JSON_FILE = BASE_DIR / 'cvm_documentos_texto_final.json'
REQUEST_DELAY = 1.5
TIMEOUT_SELENIUM = 40
SIMILARITY_THRESHOLD = 0.95
FRE_DEFAULT_DATE = "2025-06-30" # <-- DATA PADRÃO PARA DOCUMENTOS FRE

# --- FUNÇÕES AUXILIARES E DE PROCESSAMENTO (sem alterações) ---

def retry_on_failure(retries=3, delay=5):
    """Decorator para tentar executar uma função múltiplas vezes em caso de falha."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    logging.warning(f"Exceção na tentativa {i + 1}/{retries} para {func.__name__}: {e}")
                
                if i < retries - 1:
                    logging.info(f"Tentativa {i + 1}/{retries} falhou. Aguardando {delay} segundos...")
                    time.sleep(delay)
            logging.error(f"ERRO FINAL: Todas as {retries} tentativas falharam para a função {func.__name__}")
            return None
        return wrapper
    return decorator

def clean_and_normalize_text(text):
    """Aplica uma série de limpezas e normalizações no texto extraído."""
    if not text: return ""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\n{2,}', '<PARAGRAPH>', text); text = text.replace('\n', ' '); text = text.replace('<PARAGRAPH>', '\n\n')
    text = re.sub(r'Página\s+\d+\s+de\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r' +', ' ', text); text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def calculate_text_similarity(text1, text2):
    """Calcula a similaridade de Jaccard entre dois textos."""
    set1 = set(text1.lower().split()); set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2)); union = len(set1.union(set2))
    return intersection / union if union != 0 else 1.0

def setup_selenium_driver():
    """Configura e retorna uma instância do driver do Chrome."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    logging.info("Configurando o navegador (Selenium)...")
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        return driver
    except Exception as e:
        logging.error(f"Erro ao configurar o Selenium: {e}")
        return None

def normalize_company_name(name):
    """Normaliza o nome da empresa para consistência."""
    if pd.isna(name): return None
    return re.sub(r"\s+(S\.?A\.?|S/A|SA)$", " S.A.", str(name).upper().strip())

# --- FUNÇÕES UNIFICADAS DE FETCH E EXTRAÇÃO (sem alterações) ---

@retry_on_failure(retries=3, delay=5)
def fetch_and_extract_fre(url):
    """Baixa o conteúdo de um FRE para a memória e extrai o texto."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    element = soup.find('input', id='hdnConteudoArquivo')
    if element and element.get('value'):
        pdf_bytes = base64.b64decode(element.get('value'))
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    logging.warning(f"Conteúdo Base64 não encontrado na URL: {url}")
    return None

@retry_on_failure(retries=3, delay=5)
def fetch_and_extract_ipe_with_selenium(driver, url):
    """Usa Selenium para baixar um PDF para a memória e extrair o texto."""
    driver.get(url)
    WebDriverWait(driver, TIMEOUT_SELENIUM).until(EC.presence_of_element_located((By.ID, "pdfViewer")))
    time.sleep(2)
    
    javascript = """
        var iframe = document.getElementById('pdfViewer');
        var url = iframe.src;
        var callback = arguments[arguments.length - 1];
        if (!url.startsWith('blob:')) { callback({error: 'Not a blob URL'}); return; }
        fetch(url).then(response => response.blob()).then(blob => {
            var reader = new FileReader();
            reader.onload = function() { callback({data: reader.result}); };
            reader.onerror = function() { callback({error: 'Read error'}); };
            reader.readAsDataURL(blob);
        }).catch(error => { callback({error: 'Fetch error: ' + error.toString()}); });
    """
    result_dict = driver.execute_async_script(javascript)
    
    if result_dict and 'data' in result_dict:
        _, encoded_data = result_dict['data'].split(',', 1)
        pdf_bytes = base64.b64decode(encoded_data)
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    else:
        logging.error(f"Erro no Javascript para extrair PDF da URL: {url}. Detalhes: {result_dict.get('error', 'Sem detalhes')}")
        return None

# --- NOVA FUNÇÃO PRINCIPAL COM LÓGICA DE ATUALIZAÇÃO ---

def main_intelligent_update():
    logging.info("Iniciando processo de atualização inteligente.")
    
    # 1. Carregar dados existentes ou iniciar um novo dicionário
    all_processed_data = {}
    if OUTPUT_JSON_FILE.exists():
        logging.info(f"Carregando dados existentes de '{OUTPUT_JSON_FILE}'...")
        try:
            with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                all_processed_data = json.load(f)
            logging.info(f"{len(all_processed_data)} documentos já processados.")
        except json.JSONDecodeError:
            logging.warning("Arquivo JSON existente está corrompido. Começando do zero.")
    else:
        logging.info("Nenhum arquivo JSON encontrado. Um novo será criado.")

    processed_urls = set(all_processed_data.keys())
    
    # 2. Carregar planilhas de origem
    try:
        df_fre = pd.read_csv(CSV_URL, sep=';', dtype=str, encoding="latin-1").sort_values("VERSAO", ascending=False).drop_duplicates("DENOM_CIA")
        df_planos = pd.read_excel(PLANOS_URL, dtype=str)
    except FileNotFoundError as e:
        logging.error(f"Erro: Arquivo de dados não encontrado. Verifique o caminho: '{e.filename}'")
        return

    # Normalizar nomes e datas
    df_fre["DENOM_CIA"] = df_fre["DENOM_CIA"].apply(normalize_company_name)
    df_planos["Empresa"] = df_planos["Empresa"].apply(normalize_company_name)
    df_planos['Data referencia'] = pd.to_datetime(df_planos['Data referencia'], errors='coerce')
    df_planos.dropna(subset=['Data referencia', 'Link'], inplace=True)
    df_planos.sort_values(by=['Empresa', 'Data referencia'], ascending=[True, False], inplace=True)

    # 3. Preencher datas faltantes em dados existentes (Backfill)
    logging.info("\n--- Verificando e preenchendo datas de referência faltantes ---")
    date_map = {row['Link']: row['Data referencia'] for _, row in df_planos.iterrows()}
    updated_count = 0
    for url, info in all_processed_data.items():
        if 'data_referencia' not in info or info['data_referencia'] is None:
            # Se for um documento IPE (da planilha de planos)
            if url in date_map and pd.notna(date_map[url]):
                info['data_referencia'] = str(date_map[url].date())
                updated_count += 1
            # Se for um documento FRE, usa a data padrão
            elif "frmExibirArquivoFRE.aspx" in url:
                info['data_referencia'] = FRE_DEFAULT_DATE
                updated_count += 1
                
    if updated_count > 0:
        logging.info(f"-> {updated_count} registros existentes foram atualizados com a data de referência.")

    # 4. Processar NOVOS Planos de Remuneração (IPE)
    logging.info("\n--- Verificando e processando NOVOS Planos de Remuneração (IPE) ---")
    new_ipe_docs = []
    for _, row in df_planos.iterrows():
        if row["Link"] not in processed_urls:
            new_ipe_docs.append(row)
    
    if new_ipe_docs:
        logging.info(f"Encontrados {len(new_ipe_docs)} novos documentos IPE para processar.")
        driver = setup_selenium_driver()
        if driver:
            for row in new_ipe_docs:
                company_name = row["Empresa"]
                source_url = row["Link"]
                data_ref = row["Data referencia"]

                logging.info(f"Processando novo plano para: {company_name} (Data: {data_ref.date()})")
                raw_text = fetch_and_extract_ipe_with_selenium(driver, source_url)
                if not raw_text:
                    logging.warning("-> Falha ao extrair texto. Pulando.")
                    continue
                
                clean_text = clean_and_normalize_text(raw_text)
                all_processed_data[source_url] = {
                    "text": clean_text,
                    "company_name": company_name,
                    "data_referencia": str(data_ref.date())
                }
                logging.info(f"-> Sucesso! Novo documento IPE adicionado para {company_name}.")
            driver.quit()
        else:
            logging.error("Driver do Selenium não iniciado. O download dos novos planos foi abortado.")
    else:
        logging.info("Nenhum novo documento IPE encontrado.")

    # 5. Processar NOVOS Itens 8.4 (FRE)
    logging.info("\n--- Verificando e processando NOVOS Itens 8.4 (FRE) ---")
    new_fre_docs_count = 0
    for _, row in df_fre.iterrows():
        link_doc = row["LINK_DOC"]
        if not isinstance(link_doc, str) or not link_doc.startswith('http'):
            continue
        doc_number = parse_qs(urlparse(link_doc).query).get("NumeroSequencialDocumento", [None])[0]
        if not doc_number: continue
        
        source_url = f"https://www.rad.cvm.gov.br/ENET/frmExibirArquivoFRE.aspx?NumeroSequencialDocumento={doc_number}&CodigoGrupo=8000&CodigoQuadro=8120"
        
        if source_url not in processed_urls:
            new_fre_docs_count += 1
            company_name = row["DENOM_CIA"]
            logging.info(f"Processando novo Item 8.4 para: {company_name}")
            
            raw_text = fetch_and_extract_fre(source_url)
            if not raw_text:
                logging.warning("-> Falha ao extrair texto. Pulando.")
                continue
                
            clean_text = clean_and_normalize_text(raw_text)
            all_processed_data[source_url] = {
                "text": clean_text,
                "company_name": company_name,
                "data_referencia": FRE_DEFAULT_DATE # <-- Atribui a data padrão
            }
            logging.info(f"-> Sucesso! Novo documento FRE adicionado para {company_name}.")

    if new_fre_docs_count == 0:
        logging.info("Nenhum novo documento FRE encontrado.")


    # 6. Salvar o resultado final
    if not all_processed_data:
        logging.warning("Nenhum documento para salvar. Encerrando.")
        return

    logging.info(f"\nSalvando dados de texto finais em '{OUTPUT_JSON_FILE}'...")
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Sucesso! Total de {len(all_processed_data)} documentos únicos salvos.")
    except Exception as e:
        logging.error(f"ERRO ao salvar o arquivo JSON final: {e}")


if __name__ == "__main__":
    # Agora o script chama a nova função inteligente por padrão
    main_intelligent_update()