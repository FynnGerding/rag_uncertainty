from datasets import load_dataset

def data(articles_num):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    texts = []
    for i, item in enumerate(dataset):
        if i >= articles_num:
            break
        texts.append(item["text"])

    documents = []
    for text in texts:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + 300, len(text))
            chunks.append(text[start:end])
            start += 300 - 50
        documents.extend(chunks)

    return documents