import streamlit as st
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sample data
queries = [
    "What is the capital of France?", 
    "Retrieve document about climate change.",
    "Explain quantum computing.",
    "Get me the latest research on AI.",
    "How does photosynthesis work?",
    "Find documents on machine learning trends."
]
query_labels = ["general", "retrieval", "general", "retrieval", "general", "retrieval"]

# Preprocessing and feature extraction
query_vectorizer = TfidfVectorizer()
query_classifier = LogisticRegression()

# Pipeline for query classification
query_pipeline = Pipeline([
    ('vectorizer', query_vectorizer),
    ('classifier', query_classifier)
])

# Train the query classifier
query_pipeline.fit(queries, query_labels)

# Sample data for document collection classification
doc_queries = [
    "Retrieve document about climate change.",
    "Get me the latest research on AI.",
    "Find documents on machine learning trends."
]
doc_labels = ["environment", "technology", "technology"]

# Pipeline for document collection classification
doc_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the document collection classifier
doc_pipeline.fit(doc_queries, doc_labels)

# Streamlit interface
st.title("Intelligent Query Routing System")

user_query = st.text_input("Enter your query:")

if st.button("Classify and Route"):
    if user_query:
        query_type = query_pipeline.predict([user_query])[0]
        
        if query_type == "retrieval":
            doc_collection = doc_pipeline.predict([user_query])[0]
            response = f"Query identified as requiring document retrieval from the '{doc_collection}' collection."
        else:
            response = "Query identified as a general query. Generating response..."
        
        st.write(response)
    else:
        st.write("Please enter a query.")
