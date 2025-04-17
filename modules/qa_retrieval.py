from sentence_transformers import SentenceTransformer, util

# Load the fine-tuned sentence-transformer model
sbert_model = SentenceTransformer('models/fine-tuned-retriever')

def answer_question(question, context):
    # Split the context into sentences
    sentences = context.split(". ")  # You can adjust this depending on how sentences are structured
    
    # Create embeddings for the context and the question
    context_embeddings = sbert_model.encode(sentences)  # One embedding for each sentence
    question_embedding = sbert_model.encode([question])  # One embedding for the question
    
    # Calculate cosine similarity between the question and each sentence
    scores = util.cos_sim(question_embedding, context_embeddings)
    
    # Find the index of the sentence with the highest score
    best_sentence_index = scores.argmax()
    
    # Return the best sentence if the similarity score is above the threshold, else return a fallback message
    if scores[0][best_sentence_index] > 0.5:
        return sentences[best_sentence_index]  # Return the most relevant sentence
    else:
        return "Sorry, no relevant answer found."

