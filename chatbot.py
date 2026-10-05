import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CollegeFAQChatbot:
    def __init__(self, csv_file="college_faq.csv", threshold=0.25):
        self.csv_file = csv_file
        self.threshold = threshold

        # Load CSV
        self.data = pd.read_csv(csv_file)

        # Debug print
        print("CSV Columns:", self.data.columns.tolist())

        # Take question and answer columns
        self.questions = self.data["question"].astype(str).tolist()
        self.answers = self.data["answer"].astype(str).tolist()

        # Train TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_input):
        if not user_input.strip():
            return "Please enter a question."

        # Convert user question into vector
        user_vector = self.vectorizer.transform([user_input])

        # Find similarity
        similarity = cosine_similarity(user_vector, self.question_vectors)
        best_match_index = similarity.argmax()
        best_score = similarity[0][best_match_index]

        # If no good match
        if best_score < self.threshold:
            return "Sorry, I couldn't find a proper answer for that. Please ask a college-related question."

        return self.answers[best_match_index]