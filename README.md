# Fake News Prediction using Machine Learning

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Frameworks](https://img.shields.io/badge/Libraries-Scikit--learn%2C%20Pandas%2C%20NLTK-orange?style=for-the-badge)

A machine learning project to classify news articles as "Real" or "Fake" based on their headlines and authors. This project leverages Natural Language Processing (NLP) techniques to build a robust predictive model.



## 📜 Overview

In the age of digital information, the rapid spread of fake news has become a significant societal problem. This project aims to address this issue by creating a machine learning model capable of automatically detecting misinformation. By training on a labeled dataset of news articles, the model learns to identify linguistic patterns that distinguish real news from fake news.

---

## 🎯 Problem Statement

The goal of this project is to develop an efficient and accurate binary classification model. Given the title and author of a news article, the model should predict whether the news is reliable (Real) or unreliable (Fake).

---

## 💾 Dataset

The model is trained on a dataset named `train.csv`, which contains over 20,000 news articles. Each record in the dataset includes the following features:

* **id:** A unique identifier for the article.
* **title:** The headline of the news article.
* **author:** The author of the article.
* **text:** The main body of the article (not used in this model).
* **label:** The target variable, where `0` represents a reliable article and `1` represents an unreliable article.

---

## ⚙️ Project Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading & Initial Analysis:** The dataset is loaded using Pandas for initial exploration.
2.  **Data Preprocessing:** Missing values are handled, and the relevant features (`title` and `author`) are combined into a single `content` feature for analysis.
3.  **Text Cleaning (NLP):**
    * The text is cleaned by removing punctuation and numbers.
    * All text is converted to lowercase.
    * **Stopword Removal:** Common English words (e.g., "the", "a", "is") that do not add significant meaning are removed using NLTK's stopwords list.
    * **Stemming:** Words are reduced to their root form (e.g., "studies", "studying" -> "studi") using the Porter Stemmer to treat different forms of a word as a single entity.
4.  **Feature Extraction:** The cleaned text data is converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer. This technique helps in highlighting words that are more important to a specific article.
5.  **Model Training:** The dataset is split into training (80%) and testing (20%) sets. A **Logistic Regression** model is then trained on the vectorized training data.
6.  **Model Evaluation:** The model's performance is evaluated on both the training and testing sets to check for accuracy and prevent overfitting.

---

## 🛠️ Technologies Used

* **Programming Language:** Python 3
* **Libraries:**
    * **Pandas:** For data manipulation and loading CSV files.
    * **NumPy:** For numerical operations.
    * **NLTK (Natural Language Toolkit):** For text preprocessing, including stopword removal and stemming.
    * **Scikit-learn:** For building the machine learning model (`TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `accuracy_score`).
    * **re (Regular Expressions):** For cleaning and pattern matching in text.

---

## 🚀 How to Run this Project

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install numpy pandas nltk scikit-learn jupyter
    ```

4.  **Download NLTK Data:**
    Run the following commands in a Python interpreter to download the stopwords:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

5.  **Run the Jupyter Notebook:**
    Start Jupyter and open the `Fake_News_Prediction.ipynb` file.
    ```bash
    jupyter notebook
    ```
    Execute the cells in the notebook sequentially to see the entire process and results.

---

## 📊 Results

The Logistic Regression model achieved an impressive accuracy, demonstrating its effectiveness in distinguishing between real and fake news.

* **Training Accuracy:** ~99.9%
* **Testing Accuracy:** **~98.8%**

The small difference between the training and testing accuracy indicates that the model generalizes well to new, unseen data without significant overfitting.

---

## 💡 Future Improvements

* **Use Full Article Text:** Incorporate the `text` column of the dataset to provide the model with more context.
* **Try Advanced Models:** Experiment with more complex algorithms like Naive Bayes, Support Vector Machines (SVM), or deep learning models (LSTMs, BERT) to potentially improve accuracy further.
* **Hyperparameter Tuning:** Optimize the model's parameters using techniques like GridSearchCV to find the best configuration.
* **Deploy as an API:** Create a REST API using a framework like Flask or FastAPI to make the model accessible as a web service.
