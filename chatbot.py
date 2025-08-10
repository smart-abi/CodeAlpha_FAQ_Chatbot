import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# first-time run only: download NLTK tokenizer
# nltk.download('punkt')  # uncomment and run once if needed

# Load FAQs
with open('data/faqs.json', 'r', encoding='utf-8') as f:
    faqs = json.load(f)

questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]

# Create vectorizer and fit on known questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def chatbot_response(user_input, threshold=0.3):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, question_vectors)
    idx = similarity.argmax()
    score = similarity[0][idx]
    if score >= threshold:
        return answers[idx]
    else:
        return "Sorry, I don't understand your question. Could you rephrase?"

if __name__ == '__main__':
    print("FAQ Chatbot (type 'quit' to exit)")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                break
            print("Bot:", chatbot_response(user_input))
    except KeyboardInterrupt:
        print("\nGoodbye!")
