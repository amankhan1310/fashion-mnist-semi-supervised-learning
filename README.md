# Fashion-MNIST Semi-Supervised Learning

A practical comparison of **Semi-Supervised Learning and Supervised Learning** using the Fashion-MNIST dataset.

This project investigates how machine learning models perform when only a small portion of the training data is labelled and the remaining data is unlabelled. It compares Label Spreading, Logistic Regression and Random Forest using a consistent experimental setup.

## Project Overview

In real-world machine learning applications, obtaining labelled data can be expensive and time-consuming. However, large amounts of unlabelled data are often available.

This project explores whether unlabelled data can improve classification performance through semi-supervised learning.

Using Fashion-MNIST, the experiment trains models with only **10% labelled training data and 90% unlabelled training data**.

## Dataset

The project uses the official Fashion-MNIST dataset.

| Dataset split | Images |
|---|---:|
| Total images | 70,000 |
| Training images | 60,000 |
| Test images | 10,000 |
| Labelled training images (10%) | 6,000 |
| Unlabelled training images (90%) | 54,000 |

The dataset contains 10 clothing categories, with 28 × 28 grayscale images.

### Classes

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Methodology

### 1. Data Preprocessing
- Loaded the official Fashion-MNIST dataset.
- Flattened each image into 784 features.
- Normalized pixel values to the range 0–1.
- Used a fixed random seed of 42.
- Kept the official test set separate from training and model tuning.

### 2. Labelled and Unlabelled Data
The training dataset was divided into:

- 10% labelled data: 6,000 images.
- 90% unlabelled data: 54,000 images.

Unlabelled samples were assigned a label of `-1` for Label Spreading.

### 3. Dimensionality Reduction
Principal Component Analysis (PCA) reduced the input dimensions from 784 to 50.

- Variance retained: 86.3%.
- PCA was fitted only on training images.
- PCA features were used by Label Spreading.
- Logistic Regression and Random Forest used the original 784 pixel features.

### 4. Machine Learning Models

**Label Spreading — Semi-Supervised Learning**
- Uses labelled and unlabelled samples.
- Builds a similarity graph between samples.
- Propagates label information through the graph.
- Parameters: `n_neighbors=20`, `alpha=0.8`, `max_iter=1000`.

**Logistic Regression — Supervised Learning**
- Learns classification boundaries using labelled samples.
- Parameters: `C=0.1`, `solver=lbfgs`, `max_iter=2000`.

**Random Forest — Supervised Learning**
- Uses an ensemble of decision trees.
- Parameters: `n_estimators=300`, `max_depth=20`, `max_features=sqrt`.

## Experimental Results

The following results were obtained on the official 10,000-image test set.

| Metric | Label Spreading | Logistic Regression | Random Forest |
|---|---:|---:|---:|
| Accuracy | 81.90% | 83.07% | **84.32%** |
| Macro Precision | 0.8213 | 0.8293 | **0.8425** |
| Macro Recall | 0.8190 | 0.8307 | **0.8432** |
| Macro F1-score | 0.8187 | 0.8296 | **0.8413** |
| 5-Fold CV Accuracy | 0.8173 | 0.8360 | **0.8443** |

### Key Findings

- Random Forest achieved the highest test accuracy of 84.32%.
- Logistic Regression achieved 83.07%.
- Label Spreading achieved 81.90%.
- Label Spreading inferred the hidden labels of the 54,000 unlabelled training samples with 82.60% accuracy.
- Random Forest performed best in this experimental setup, even though it used only the 6,000 labelled training samples.

**Important:** These results show that semi-supervised learning is not guaranteed to outperform supervised learning. Performance depends on the dataset, model and experimental setup.

## Label Percentage Experiment

To investigate how labelled data availability affects performance, additional experiments were conducted using:

- 5% labelled data
- 10% labelled data
- 20% labelled data
- 30% labelled data
- 50% labelled data

Each percentage was tested using three random seeds:

`42`, `123`, `2026`

This produced 15 experimental runs.

## Streamlit Dashboard

The project includes an interactive, read-only Streamlit dashboard for exploring the saved experimental results.

### Dashboard Features

1. Dataset overview
2. Model comparison
3. Confusion matrices
4. Per-class performance
5. Training and prediction timing
6. Labelled-data percentage comparison
7. Hyperparameter tuning results
8. Live image prediction

The dashboard loads the saved models and results. It does not retrain models or modify the experimental results.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Structure

```text
fashion-mnist-semi-supervised-learning/
│
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
│
├── fmnist/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   └── figures.py
│
├── models/
│   ├── label_spreading.joblib
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── results/
│   ├── results.json
│   ├── test_predictions.npz
│   └── figures/
│
└── logs/
    └── training log
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/amankhan1310/fashion-mnist-semi-supervised-learning.git

cd fashion-mnist-semi-supervised-learning
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline

```bash
python train.py
```

The training pipeline downloads the dataset when needed, trains the models and saves the generated results and model artifacts.

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

**Note:** The repository includes saved model artifacts and experimental results. The dashboard can be explored using these saved artifacts without running the training pipeline again.

## Reproducibility

- Main experiment random seed: 42.
- Five-fold stratified cross-validation.
- Official Fashion-MNIST test set kept separate.
- Saved model artifacts and predictions are included in the repository.
- Additional labelled-data experiments use three different random seeds.

## Author

**Aman Khan**

B.Tech – Computer Science and Engineering (AI & Analytics)

MIT ADT University, Pune

GitHub: [@amankhan1310](https://github.com/amankhan1310)

## License

Add a license to this repository if you intend to permit others to reuse, modify or distribute the project. No license is specified here yet.
