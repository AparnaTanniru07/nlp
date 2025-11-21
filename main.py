# 1. Setup
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re # Import re module
import numpy as np # Import numpy

# 2. Load Data

feedback_data = [
    'This is great feedback.',
    'Another piece of feedback.',
    'I love this product!',
    'This is terrible.',
    'Could be better, but still good.',
    'Absolutely fantastic experience.'
    "Excellent product! Fast delivery and great customer service.",
    "Item arrived damaged. Very disappointed with packaging.",
    "Amazing quality! Will definitely order again.",
    "Shipping took too long. Product is okay but expected faster.",
    "Love it! Exceeded my expectations. Highly recommend!",
    "Package was crushed during transit. Item broken.",
    "Great product but delivery was delayed by 2 weeks.",
    "Outstanding quality and quick shipping. Very satisfied!",
    "Good value for money. Satisfied with the purchase.",
    "Not as described. Expected better quality.",
    "Super fast delivery! Product works perfectly.",
    "Customer support was slow to respond but resolved my issue.",
    "Affordable and reliable. Will buy again.",
    "Product was missing parts. Needs better QC.",
    "Fantastic experience overall!",
    "Color was slightly different than shown in pictures.",
    "Very durable and well-built product.",
    "Received wrong item. Had to return.",
    "Amazing customer service. Issue resolved quickly.",
    "Mediocre product. Not worth the price."
]
  # Your feedback list

# Placeholder for undefined variables
model_name = 'all-MiniLM-L6-v2' # Example model name
random_state = 42 # Example random state

# 3. Create Processor
class FeedbackProcessor:
    def __init__(self):
       self.model = SentenceTransformer(model_name)
       self.random_state = random_state

    def _basic_clean(self, text):
        text = text.lower().strip()
        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    def clean_and_embed(self, texts):
        """
        texts: list/Series of strings
        returns: np.ndarray of embeddings
        """
        # Ensure list of strings
        cleaned = [self._basic_clean(str(t)) for t in texts]
        embeddings = self.model.encode(cleaned, show_progress_bar=False)
        return np.array(embeddings)
        # Clean \u2192 Preprocess \u2192 Embed

    def cluster(self, embeddings, n_clusters):
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans.fit(embeddings)
        return kmeans.labels_

# 4. Process and Analyze
processor = FeedbackProcessor()
embeddings = processor.clean_and_embed(feedback_data)
clusters = processor.cluster(embeddings, n_clusters=3)

# 5. Display Results
for cluster_id in range(3):
    print(f"\nCluster {cluster_id}:")
    # Show feedback in this cluster
    # Example of how to show feedback in clusters (assuming feedback_data is a list of original texts)
    for i, label in enumerate(clusters):
        if label == cluster_id:
            print(f"- {feedback_data[i]}")
