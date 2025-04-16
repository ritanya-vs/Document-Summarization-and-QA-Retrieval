from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------- Step 1: Corpus ----------
corpus = [
    "Paris is the capital of France. It is known for the Eiffel Tower and its art museums.",
    "The Moon orbits around the Earth and causes tides due to its gravitational pull.",
    "Photosynthesis allows plants to convert sunlight into energy using chlorophyll.",
    "The Great Wall of China was built to protect Chinese states from invasions.",
    "Python is a popular programming language known for its simplicity and readability."
]

# ---------- Step 2: Load Retriever Model ----------
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = retriever_model.encode(corpus, convert_to_tensor=True)

# ---------- Step 3: Retrieve Top Relevant Documents ----------
def retrieve_documents(question, top_k=2):
    question_embedding = retriever_model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    results = [corpus[hit['corpus_id']] for hit in hits[0]]
    return results

# ---------- Step 4: Load Reader Model ----------
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def get_answer(question, context):
    result = qa_pipeline({
        "question": question,
        "context": context
    })
    return result['answer']

# ---------- Step 5: Full QA Pipeline ----------
def answer_question(question, top_k=2):
    docs = retrieve_documents(question, top_k=top_k)
    answers = []
    for doc in docs:
        ans = get_answer(question, doc)
        answers.append({
            'answer': ans,
            'context': doc
        })
    return answers

# ---------- Step 6: Try it Out ----------
if __name__ == "__main__":
    question = "How do plants make energy?"
    responses = answer_question(question)

    for i, r in enumerate(responses):
        print(f"Result {i+1}:")
        print("Answer:", r['answer'])
        print("From:", r['context'])
        print()
