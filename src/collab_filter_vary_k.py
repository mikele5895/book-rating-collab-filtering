# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load and preprocess the ratings dataset
def load_data():
    # Read CSV with comma delimiter
    ratings = pd.read_csv('../data/Ratings.csv', sep=',', encoding='latin-1')
    
    # Clean column names
    ratings.columns = [col.strip().replace('\r', '').replace('\n', '') for col in ratings.columns]

    # Convert ratings to numeric and filter out missing or zero ratings
    ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')
    ratings = ratings.dropna(subset=['Book-Rating'])
    ratings = ratings[ratings['Book-Rating'] > 0]

    # Keep only users who rated at least 10 books
    user_counts = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 10].index)]

    # Keep only books that were rated at least 10 times
    book_counts = ratings['ISBN'].value_counts()
    ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 10].index)]

    return ratings

# Predict a user's rating for a specific book using item-item collaborative filtering
def predict_rating(user_id, isbn, matrix, similarity_matrix, k):
    if isbn not in similarity_matrix.columns or user_id not in matrix.index:
        return np.nan  # Cannot make prediction
    user_ratings = matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return np.nan
    sim_scores = similarity_matrix[isbn]
    
    # Find items rated by the user that also have similarity scores
    common_books = user_ratings.index.intersection(sim_scores.index)
    
    # Select top-k similar books
    top_k = sim_scores[common_books].sort_values(ascending=False)[:k]
    top_ratings = user_ratings[top_k.index]

    # Weighted average of top-k ratings
    if top_k.sum() == 0:
        return np.nan
    return np.dot(top_k, top_ratings) / top_k.sum()

# Evaluate the model using Mean Absolute Difference (MAD)
def evaluate(test_df, train_matrix, sim_matrix, k):
    actuals, predictions = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        uid, isbn, true_rating = row['User-ID'], row['ISBN'], row['Book-Rating']
        pred = predict_rating(uid, isbn, train_matrix, sim_matrix, k)
        if not np.isnan(pred):
            actuals.append(true_rating)
            predictions.append(pred)
    # Compute average absolute error
    mad = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
    return mad

# Main execution: load data, train model, and evaluate at different k values
if __name__ == "__main__":
    print("Loading data...")
    ratings = load_data()
    
    # Split dataset into 75% train and 25% test by individual (User-ID, ISBN) pairs
    train_df, test_df = train_test_split(ratings, test_size=0.25, random_state=42)

    print("Creating user-item matrix...")
    train_matrix = train_df.pivot(index='User-ID', columns='ISBN', values='Book-Rating')

    # Compute item-item similarity matrix using cosine similarity
    sim_matrix = cosine_similarity(train_matrix.T.fillna(0))
    sim_df = pd.DataFrame(sim_matrix, index=train_matrix.columns, columns=train_matrix.columns)

    print("Varying neighborhood size k...")
    for k in [5, 10, 15, 20, 50, 100]:
        print(f"Evaluating k={k}...")
        mad = evaluate(test_df, train_matrix, sim_df, k=k)
        print(f"  -> MAD: {mad:.4f}")
