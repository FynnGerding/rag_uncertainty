import rag_pipeline.retrievers as retrievers
def answer_question(query, tokenizer, model, bm25, documents, k):

    context = "\n".join(retrievers.retrieve(query, documents, bm25, k))

    # Build prompt
    prompt = f"""Use the context to answer the user question.

    Context:
    {context}

    Question: {query}
    Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    pred = model.generate(**inputs)

    return tokenizer.decode(pred[0], skip_special_tokens=True)
