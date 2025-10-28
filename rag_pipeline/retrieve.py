from rank_bm25 import BM25Okapi

def build_bm25(documents):
    tokenized_docs = [doc.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

def retrieve(query, documents, bm25, k):
    scores = bm25.get_scores(query.split())
    ranked_indices = scores.argsort()[-k:][::-1]  # top k sorted
    top_k_documents = []

    for i in ranked_indices:
        top_k_documents.append(documents[i])

    return top_k_documents
