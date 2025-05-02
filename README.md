# DATA606 Project 2 – Book Rating Prediction Using Collaborative Filtering

**Author:** Hai Le  
**Student ID:** 121334831  
**Course:** DATA606  
**Project:** Collaborative Filtering for Book Rating Prediction  
**Approach:** Item-Item Collaborative Filtering  
**Environment:** Python 3.10+ with `venv` and preinstalled requirements  

---

## Project Overview

This project implements **Item-Item Collaborative Filtering** to predict how users would rate books they haven’t rated yet, based on the similarity between books and the user’s previous ratings. 

Predictions are evaluated using **Mean Absolute Difference (MAD)** on a held-out test set, following these key principles:
- Ratings are split randomly by **individual `(User-ID, ISBN)` pairs**, not by user or item
- Filtering ensures users and books with insufficient data are removed

---

## Dataset Description

We use the dataset consisting of:

- `Ratings.csv`:  
  Contains user ratings for books on a scale from 1 to 10 (0 indicates “not rated” and is ignored).
  
- `Users.csv` & `Books.csv`

**Data preprocessing includes:**
- Removing zero or invalid ratings
- Filtering out users with < 10 ratings and books with < 10 ratings (to reduce sparsity and improve similarity reliability)

---

## Algorithm and Implementation Details

1. **Data Cleaning and Filtering**  
   - Non-numeric or zero ratings are removed.
   - Only users with ≥10 ratings and books with ≥10 ratings are retained to ensure sufficient overlap.

2. **Train-Test Split**  
   - Data is randomly split by (User-ID, ISBN) pair.

3. **Matrix Construction**  
   - A user-item utility matrix is built with users as rows, books as columns, and ratings as values.

4. **Item-Item Similarity Computation**  
   - Cosine similarity is calculated between items (books) using the transposed matrix.

5. **Prediction Function**  
   - For each test pair, the model finds the `k` most similar books the user has rated and computes a weighted average of those ratings.

6. **Evaluation**  
   - Performance is measured using **Mean Absolute Difference (MAD)** between predicted and actual ratings.

---

## Dependencies and Libraries

The following Python libraries are used in this project:

- `pandas` – data loading and manipulation  
- `numpy` – numeric computations (e.g., vector operations)  
- `scikit-learn` – for `train_test_split` and cosine similarity  
- `tqdm` – to display progress bars for evaluations

All dependencies are listed in the `requirements.txt` and already installed in the included `venv`.

---

## How to Run (VSCode Setup)

> A preconfigured Python `venv` environment is included, but steps are provided below in case you need to reproduce it.

### 1. Open Project in VSCode

Navigate to the root of the project and open VSCode there.

### 2. (Optional) Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the Script

From the `src/` folder, run one of the scripts:

```bash
python collab_filter_vary_k.py         # Varies neighborhood size k
python collab_filter_vary_ratio.py     # Varies training ratio
```

---

## Evaluation: MAD (Mean Absolute Difference)

MAD measures the average difference between predicted and actual ratings. Lower is better. A **baseline MAD is ~2.0** — values well below that are strong.

### Performance vs Neighborhood Size (k)

| k       | MAD     |
|---------|---------|
| 5       | 1.2303  |
| 10      | 1.2221  |
| 15      | 1.2217  |
| 20      | 1.2219  |
| 50      | 1.2226  |
| 100     | 1.2226  |

**Interpretation:**  
As `k` increases, performance improves up to around **k=15–20**, then levels off. Typical in item-based collaborative filtering, where the top 10–20 most similar items capture most of the signal.

---

### Performance vs Training Ratio

| Train Ratio | MAD     |
|-------------|---------|
| 60%         | 1.2533  |
| 65%         | 1.2421  |
| 70%         | 1.2336  |
| 75%         | 1.2221  |
| 80%         | 1.2139  |
| 85%         | 1.2041  |
| 90%         | 1.2028  |

**Interpretation:**  
As the training size increases, MAD consistently decreases — indicating that the model benefits from more user history and book co-rating information.

---
## Summary

This project demonstrates a successful implementation of **Item-Item Collaborative Filtering** with MAD evaluations across different hyperparameters. The model shows consistent, reasonable predictive power and follows best practices in train/test splitting, filtering, and reproducibility.

---
