from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer('models/fine-tuned-retriever')

def answer_question(question, context):
    sentences = context.split(". ")  
    
    context_embeddings = sbert_model.encode(sentences) 
    question_embedding = sbert_model.encode([question])  
    
    scores = util.cos_sim(question_embedding, context_embeddings)
    
    best_sentence_index = scores.argmax()
    
    if scores[0][best_sentence_index] > 0.5:
        return sentences[best_sentence_index]  
    else:
        return "Sorry, no relevant answer found."

