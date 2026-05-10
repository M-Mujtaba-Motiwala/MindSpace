import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def build_index(_df):
    """
    Pre-compute embeddings for every user_input row in the dataset.
    Uses the user_input column so we compare user utterances to user utterances.
    Falls back to bot_response if user_input is absent.
    Returns (embeddings np.ndarray, index list of row positions).
    """
    encoder = load_encoder()
    col = "user_input" if "user_input" in _df.columns else "bot_response"
    texts = _df[col].fillna("").tolist()
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False,
                                normalize_embeddings=True)
    return embeddings, col


def semantic_retrieve(user_text, df, intent):
    """
    Return the df row whose user_input (or bot_response) is semantically
    closest to user_text, restricted to rows matching `intent`.
    Falls back to random sample if no match found.
    """
    subset = df[df["intent"] == intent]
    if subset.empty:
        return None

    encoder = load_encoder()
    embeddings, col = build_index(df)

    query_vec = encoder.encode([user_text], normalize_embeddings=True)[0]

    # Only consider rows in the intent subset
    idx = subset.index.tolist()
    sub_embs = embeddings[idx]

    scores = sub_embs @ query_vec          # cosine similarity (already normalized)
    best_local = int(np.argmax(scores))
    best_global = idx[best_local]

    return df.loc[best_global]
