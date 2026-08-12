# Customer Behavior Prediction

## Project Description

This project segments e-commerce customers into behavioral personas using unsupervised machine learning, based on RFM (Recency, Frequency, Monetary) features engineered from raw transaction data. Three clustering algorithms are implemented and compared — **K-Means**, **Agglomerative Hierarchical Clustering**, and **Gaussian Mixture Models (GMM)** — evaluated using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index. The winning model is used to build a unified prediction pipeline and label each cluster with a human-readable persona (e.g. "Recently Active, Frequent, High-Value, Low-Return"). The project covers data loading, exploratory data analysis (EDA), preprocessing (RFM feature engineering and scaling), model development, hyperparameter tuning, and performance evaluation with PCA visualizations, dendrograms, elbow/BIC-AIC curves, and persona heatmaps/radar charts. A Gradio interface is also integrated for interactive, real-time customer segment prediction.

## How to Run the Code

1. Clone the repository:

```
git clone https://github.com/Gabriel-idkw/Customer-Behavior-Prediction.git
cd Customer-Behavior-Prediction
```

2. Install dependencies: It's recommended to create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r app/requirements.txt
```

3. Run the Jupyter/Colab Notebooks: Open the notebooks in Google Colab or a local Jupyter environment and execute all cells sequentially, in this order:

   - `01_data_preprocessing_eda.ipynb` — data loading, cleaning, EDA, and RFM feature engineering
   - `02_kmeans_clustering.ipynb` — K-Means clustering and tuning
   - `03_hierarchical_clustering.ipynb` — Agglomerative Hierarchical clustering and tuning
   - `04_gmm_clustering.ipynb` — Gaussian Mixture Model clustering and tuning
   - `05_model_comparison_pipeline.ipynb` — comparative evaluation, persona profiling, and export of the final unified pipeline

   Note: If running in Google Colab, grant Drive access if prompted, and make sure `data/raw_ecommerce_data.csv` is accessible to the notebook.

4. Run the Gradio App: Once Notebook 05 has been run and `models/final_unified_pipeline.pkl` has been generated, launch the interactive app:

```
cd app
pip install -r requirements.txt
python app.py
```

This opens a Gradio interface where you can enter a customer's RFM profile (Recency, Frequency, Monetary, Average Order Value, Return Rate) and get an instant persona prediction with a live radar chart comparison.

## Libraries Required

All necessary Python libraries are listed in `app/requirements.txt`. Install them using:

```
pip install -r app/requirements.txt
```

Key libraries used across the notebooks and app include `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`/`seaborn`, `plotly`, `joblib`, and `gradio`.

## Project Structure

```
├── 01_data_preprocessing_eda.ipynb     # Data loading, EDA, RFM feature engineering
├── 02_kmeans_clustering.ipynb          # K-Means clustering
├── 03_hierarchical_clustering.ipynb    # Agglomerative Hierarchical clustering
├── 04_gmm_clustering.ipynb             # Gaussian Mixture Model clustering
├── 05_model_comparison_pipeline.ipynb  # Model comparison & persona profiling
├── app/
│   ├── app.py                          # Gradio app for interactive predictions
│   ├── rfm_pipeline.py                 # RFM feature transformer used by the pipeline
│   └── requirements.txt
├── data/                               # Raw/preprocessed data and generated plots
├── metrics/                            # JSON results per algorithm + final comparison
└── models/                             # Saved model artifacts (.pkl)
```

## Algorithm Implementation Summary

* **K-Means Clustering**: `sklearn.cluster.KMeans`, tuned to `k=4` (`init='k-means++'`, `n_init=10`)
* **Agglomerative Hierarchical Clustering**: tuned to `k=4` (`linkage='average'`, `metric='manhattan'`) — **winning algorithm**, selected via Silhouette Score
* **Gaussian Mixture Model (GMM)**: `sklearn.mixture.GaussianMixture`, tuned to `n_components=4` (`covariance_type='tied'`)

Each algorithm was tuned and evaluated independently, then compared in Notebook 05 using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index to select the final production pipeline.
