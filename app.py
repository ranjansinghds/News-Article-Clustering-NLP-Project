import streamlit as st
import pandas as pd
import re
import string
import joblib


# Page Configuration
st.set_page_config(
    page_title="News Article Clustering",
    page_icon="📰",
    layout="centered"
)


# Load Models
@st.cache_resource
def load_models():

    tfidf = joblib.load("tfidf_vectorizer.pkl")
    kmeans = joblib.load("kmeans_model.pkl")

    return tfidf, kmeans


tfidf, kmeans = load_models()


# Text Cleaning Function

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text



# Get Cluster Keywords

def get_cluster_keywords(model, vectorizer, cluster_number):

    terms = vectorizer.get_feature_names_out()

    center = model.cluster_centers_[cluster_number]

    top_indices = center.argsort()[-10:][::-1]

    keywords = [
        terms[index]
        for index in top_indices
    ]

    return keywords


# Cluster Topic Names

# Change these after checking your actual
# cluster keywords from your notebook.

cluster_topics = {
    0: "Sports",
    1: "Business",
    2: "World / Politics",
    3: "Technology"
}


# Title

st.title("📰 News Article Topic Clustering")

st.write(
    """
    This application uses Natural Language Processing
    and K-Means clustering to automatically group news
    articles into different topics.
    """
)


# Sidebar

st.sidebar.title("About the Project")

st.sidebar.write(
    """
    **Techniques Used**

    • Text Cleaning  
    • NLP  
    • TF-IDF  
    • K-Means Clustering  
    • Silhouette Score  
    • Dimensionality Reduction  

    The model works without using predefined
    news categories during training.
    """
)



# User Input

st.subheader("Enter a News Article")

article = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste a news article here..."
)


# Prediction

if st.button("🔍 Predict Topic"):

    if article.strip() == "":

        st.warning(
            "Please enter a news article first."
        )

    else:

        # Clean text
        cleaned_article = clean_text(article)

        # Convert text into TF-IDF
        article_vector = tfidf.transform(
            [cleaned_article]
        )

        # Predict cluster
        cluster = kmeans.predict(
            article_vector
        )[0]

        # Get topic
        topic = cluster_topics.get(
            cluster,
            f"Cluster {cluster}"
        )

        # Get keywords
        keywords = get_cluster_keywords(
            kmeans,
            tfidf,
            cluster
        )

        # Display Results

        st.success(
            f"Detected Topic: {topic}"
        )

        st.metric(
            "Cluster",
            cluster
        )

        st.subheader("🔑 Important Keywords")

        st.write(
            ", ".join(keywords)
        )

        # Show Cleaned Text

        with st.expander("View Processed Text"):

            st.write(
                cleaned_article
            )


# Footer

st.divider()

st.caption(
    "News Article Clustering | NLP + Unsupervised Learning"
)
