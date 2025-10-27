def retrieve(query, model, documents, k):

    scores = model.get_scores(query.split())
    top_results = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_results]