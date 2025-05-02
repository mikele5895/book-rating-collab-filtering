# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load and filter the Ratings dataset
def load_data():
    ratings = pd.read_csv('../data/Ratings.csv', sep=',', encoding='latin-1')
    ratings.columns = [col.strip().replace('\r', '').replace('\n', '') for col in ratings.columns]
    
    # Convert ratings to numeric and drop invalid/missing values
    ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')
    ratings = ratings.dropna(subset=['Book-Rating'])
    ratings = ratings[ratings['Book-Rating'] > 0]

    # Filter for users and books with at least 10 ratings
    user_counts = ratings['User-ID'].value_counts()
    ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 10].index)]
    book_counts = ratings['ISBN'].value_counts()
    ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 10].index)]

    return ratings

# Predict rating for a user-book pair using item-item collaborative filtering
def predict_rating(user_id, isbn, matrix, similarity_matrix, k):
    if isbn not in similarity_matrix.columns or user_id not in matrix.index:
        return np.nan
    user_ratings = matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return np.nan
    sim_scores = similarity_matrix[isbn]
    
    # Identify books rated by user that are similar to target book
    common_books = user_ratings.index.intersection(sim_scores.index)
    top_k = sim_scores[common_books].sort_values(ascending=False)[:k]
    top_ratings = user_ratings[top_k.index]

    # Compute weighted average of ratings
    if top_k.sum() == 0:
        return np.nan
    return np.dot(top_k, top_ratings) / top_k.sum()

# Evaluate predictions using Mean Absolute Difference (MAD)
def evaluate(test_df, train_matrix, sim_matrix, k):
    actuals, predictions = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        uid, isbn, true_rating = row['User-ID'], row['ISBN'], row['Book-Rating']
        pred = predict_rating(uid, isbn, train_matrix, sim_matrix, k)
        if not np.isnan(pred):
            actuals.append(true_rating)
            predictions.append(pred)
    mad = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
    return mad

# Main block: vary training sample ratios from 60% to 90% and evaluate MAD with fixed k=10
if __name__ == "__main__":
    print("Loading data...")
    ratings = load_data()

    print("Varying sample ratios from 60% to 90%...")
    for ratio in range(60, 95, 5):
        print(f"\n== Train Ratio: {ratio}% ==")
        
        # Split into train/test using given ratio
        train_df, test_df = train_test_split(ratings, test_size=(100 - ratio)/100, random_state=42)

        # Create user-item matrix from training data
        train_matrix = train_df.pivot(index='User-ID', columns='ISBN', values='Book-Rating')

        # Compute item-item similarity matrix
        sim_matrix = cosine_similarity(train_matrix.T.fillna(0))
        sim_df = pd.DataFrame(sim_matrix, index=train_matrix.columns, columns=train_matrix.columns)

        # Evaluate and report MAD for current ratio
        mad = evaluate(test_df, train_matrix, sim_df, k=10)
        print(f"  -> MAD: {mad:.4f}")
