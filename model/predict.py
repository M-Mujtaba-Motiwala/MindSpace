import streamlit as st
from transformers import pipeline

LABEL_TO_INTENT = {
    "joy": "emotional_support",
    "sadness": "depression_symptoms",
    "anger": "stress_reaction",
    "fear": "anxiety_symptoms",
    "love": "emotional_support",
    "surprise": "mental_health_faq",
}


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")


def predict_intent(text):
    classifier = load_classifier()
    result = classifier(text)[0]
    label = result["label"]
    intent = LABEL_TO_INTENT.get(label, "mental_health_faq")
    return intent, label  # (intent, raw_emotion)
