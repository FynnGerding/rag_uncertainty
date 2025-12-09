from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, List
import logging
import gc
import random
import os
import json
from datasets.exceptions import DatasetGenerationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def data(
    data_name: str,
    articles_num: Optional[int] = None,  # Defaults to None (All)
    chunk_size: int = 300,
    overlap: int = 50,
    seed: int = 42,
    cache_dir: str = "data",
) -> List[str]:
    """
    Loads and chunks a Hugging Face dataset without streaming.
    Caches processed chunks to disk for reuse.
    
    Args:
        articles_num (int, optional): Number of articles to process. 
                                      If None, processes the entire dataset.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset = None
    if articles_num is not None:
        cache_path = os.path.join(cache_dir, f"{data_name.replace('/', '_')}_{articles_num}.jsonl")
        if os.path.exists(cache_path):
            logging.info(f"Found cached dataset at {cache_path}. Loading...")
            with open(cache_path, "r", encoding="utf-8") as f:
                return [json.loads(line)["text"] for line in f]
    
    random.seed(seed)
    logging.info(f"Loading dataset '{data_name}' (train split, non-streaming)...")

    target_name = '20231101.en'
    try:
        dataset = load_dataset(data_name, streaming=False, name=target_name, split="train")
    except DatasetGenerationError:
        logging.warning("Dataset generation error. Retrying with forced redownload...")
        dataset = load_dataset(data_name, split="train", streaming=False, name=target_name, download_mode="force_redownload")
    except Exception as e:
        logging.error(f"Failed to load dataset '{data_name}': {e}")
        return []

    total = len(dataset)
    
    if articles_num is None:
        articles_num = total
        
        cache_path = os.path.join(cache_dir, f"{data_name.replace('/', '_')}_{articles_num}.jsonl")
        if os.path.exists(cache_path):
            logging.info(f"Found cached full dataset at {cache_path}. Loading...")
            with open(cache_path, "r", encoding="utf-8") as f:
                return [json.loads(line)["text"] for line in f]
    else:
        cache_path = os.path.join(cache_dir, f"{data_name.replace('/', '_')}_{articles_num}.jsonl")

    if total > articles_num:
        logging.info(f"Sampling {articles_num} articles from {total}...")
        indices = random.sample(range(total), k=articles_num)
        iterator = (dataset[i] for i in indices)
    else:
        logging.info(f"Using all {total} articles...")
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
                logging.warning(f"Skipping record {i}: {inner_e}")
                continue
    except KeyboardInterrupt:
        logging.warning("Interrupted by user — returning partial results.")
    except Exception as outer_e:
        logging.error(f"Fatal error while iterating dataset: {outer_e}")

    try:
        logging.info(f"Saving {len(documents)} chunks to {cache_path}...")
        with open(cache_path, "w", encoding="utf-8") as f:
            for chunk in documents:
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
        logging.info("Cache saved successfully.")
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")

    return documents