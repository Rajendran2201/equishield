import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import os

def create_and_save_vectorizer():
    """
    Create, fit, and save the TF-IDF vectorizer properly.
    This function ensures the vectorizer is fully fitted before saving.
    """
    print("Starting vectorizer training process...")
    
    # Load the dataset
    try:
        print(f"Current working directory: {os.getcwd()}")
        df = pd.read_csv("processed_headlines_with_bias.csv")
        print(f"Dataset loaded successfully with {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Clean and prepare data
    df = df.dropna(subset=["cleaned_headline"])
    print(f"After dropping NaN values: {len(df)} rows remain")
    
    # Ensure we have text data to work with
    texts = df["cleaned_headline"].tolist()
    print(f"Number of texts to process: {len(texts)}")
    
    if len(texts) == 0:
        print("ERROR: No valid texts found for vectorizer training!")
        return False
        
    # Sample check
    print("First 3 text samples:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i+1}. {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Create and fit the vectorizer with explicit parameters
    try:
        print("Creating and fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,         # Ignore terms that appear in fewer than 2 documents
            max_df=0.95,      # Ignore terms that appear in more than 95% of documents
            stop_words="english",
            lowercase=True,   # Convert all text to lowercase
            norm='l2',        # Apply L2 normalization
            use_idf=True,     # Enable inverse document frequency weighting
            smooth_idf=True,  # Add 1 to document frequencies to prevent division by zero
            sublinear_tf=True # Apply sublinear tf scaling (replace tf with 1 + log(tf))
        )
        
        # Explicitly fit the vectorizer
        print("Fitting vectorizer on training data...")
        X = vectorizer.fit_transform(texts)
        
        # Verify the vectorizer has been fitted
        if not hasattr(vectorizer, 'idf_'):
            print("ERROR: Vectorizer fitting failed - no idf_ attribute!")
            return False
            
        print(f"Vectorizer successfully fitted! Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"Transformed matrix shape: {X.shape}")
        
        # Test the vectorizer on a sample text
        test_text = "This is a test headline about women in politics"
        test_vector = vectorizer.transform([test_text])
        print(f"Test transformation successful: {test_vector.shape}")
        
        # Save the vectorizer
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        print("Vectorizer saved to tfidf_vectorizer.pkl")
        
        # Verify the saved vectorizer
        loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        if hasattr(loaded_vectorizer, 'idf_'):
            print("SUCCESS: Loaded vectorizer has idf_ attribute!")
            return True
        else:
            print("ERROR: Loaded vectorizer missing idf_ attribute!")
            return False
            
    except Exception as e:
        print(f"Error during vectorizer training: {e}")
        import traceback
        traceback.print_exc()
        return False

# Execute the function
if __name__ == "__main__":
    success = create_and_save_vectorizer()
    if success:
        print("\n✅ VECTORIZER SUCCESSFULLY TRAINED AND SAVED!")
    else:
        print("\n❌ VECTORIZER TRAINING FAILED!")