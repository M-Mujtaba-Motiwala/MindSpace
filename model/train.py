import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv(r"enhanced_mental_health_chatbot_dataset (1).csv")
df.columns = df.columns.str.strip().str.lower()

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["cleaned"] = df["user_input"].apply(clean_text)

X = df["cleaned"]
y = df["intent"]

# Vectorizer (NLP part)
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 🔥 Neural Network
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)
model.fit(X_vec, y)

# Save
pickle.dump(model, open("model/intent_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Neural model trained!")
