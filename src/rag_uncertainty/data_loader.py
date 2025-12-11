import logging
import gc
import random
import os
import json
import orjson
import glob
from typing import Optional, List
from datasets import load_dataset
from tqdm import tqdm
from datasets.exceptions import DatasetGenerationError

logger = logging.getLogger("rag_uncertainty")

def load_jsonl(target_file):
    total_size = os.path.getsize(target_file)
    data = []
    with open(target_file, "rb") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Loading JSONL") as pbar:
            for line in f:
                text = orjson.loads(line)["text"]
                data.append(text)
                pbar.update(len(line))
                
    return data

def load_data(
    data_name: str,
    articles_num: Optional[int] = None,
    chunk_size: int = 300,
    overlap: int = 50,
    seed: int = 42,
    cache_dir: str = "data",
) -> List[str]:
    """
    Loads text data. Priority:
    1. specific cached file (if articles_num is set)
    2. ANY cached file matching data_name (if articles_num is None)
    3. Download and process from Hugging Face
    """
    os.makedirs(cache_dir, exist_ok=True)
    sanitized_name = data_name.replace('/', '_')
    
    # 1. Attempt to Load from Cache
    target_file = None
    
    if articles_num is not None:
        # Case A: Specific count -> look for exact match
        specific_path = os.path.join(cache_dir, f"{sanitized_name}_{articles_num}.jsonl")
        if os.path.exists(specific_path):
            target_file = specific_path
    else:
        # Case B: Take the first available file json file
        existing_files = glob.glob(os.path.join(cache_dir, f"{sanitized_name}_*.jsonl"))
        if existing_files:
            target_file = existing_files[0] # Just take the first one found

    if target_file:
        logger.info(f"Loading cached data from: {target_file}")
        return load_jsonl(target_file)

    # --- 2. Download and Process (Cache Missing) ---
    logger.info(f"No suitable cache found. Downloading '{data_name}'.")
    random.seed(seed)

    target_name = '20231101.en'
    try:
        dataset = load_dataset(data_name, streaming=False, name=target_name, split="train")
    except DatasetGenerationError:
        logger.warning("Dataset generation error. Retrying with forced redownload.")
        dataset = load_dataset(data_name, split="train", streaming=False, name=target_name, download_mode="force_redownload")
    except Exception as e:
        logger.error(f"Failed to load dataset '{data_name}': {e}")
        return []

    total = len(dataset)
    
    if articles_num is None:
        articles_num = total
        
        cache_path = os.path.join(cache_dir, f"{data_name.replace('/', '_')}_{articles_num}.jsonl")
        if os.path.exists(cache_path):
            logger.info(f"Found cached full dataset at {cache_path}. Loading...")
            with open(cache_path, "r", encoding="utf-8") as f:
                return [json.loads(line)["text"] for line in f]
    else:
        cache_path = os.path.join(cache_dir, f"{data_name.replace('/', '_')}_{articles_num}.jsonl")

    if total > articles_num:
        logger.info(f"Sampling {articles_num} articles from {total}...")
        indices = random.sample(range(total), k=articles_num)
        iterator = (dataset[i] for i in indices)
    else:
        logger.info(f"Using all {total} articles...")
        iterator = iter(dataset)

    documents, count = [], 0
    try:
        for i, item in enumerate(tqdm(iterator, total=articles_num, desc="Processing dataset")):
            try:
                text = item.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    documents.append(text[start:end])
                    start += chunk_size - overlap
                count += 1
                if i % 1000 == 0:
                    gc.collect()
            except Exception as inner_e:
                logger.warning(f"Skipping record {i}: {inner_e}")
                continue
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — returning partial results.")
    except Exception as outer_e:
        logger.error(f"Fatal error while iterating dataset: {outer_e}")

    try:
        logger.info(f"Saving {len(documents)} chunks to {cache_path}...")
        with open(cache_path, "w", encoding="utf-8") as f:
            for chunk in documents:
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
        logger.info("Cache saved successfully.")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return documents
