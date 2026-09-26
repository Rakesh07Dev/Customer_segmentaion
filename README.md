# 🎯 Customer Segmentation using K-Means

A machine learning project that groups customers into meaningful segments using **K-Means Clustering**.

The project uses customer **age, annual income, and spending score** to discover groups of customers with similar characteristics.

## 🚀 Live Demo

👉 **[Open Customer Segmentation App](https://customersegmentaiongit-bzo44znfni6zvoxw9ox9eu.streamlit.app/)**

The application is built with **Streamlit** and can be used directly in a web browser.

---

## 🧠 Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Exploration
       ↓
Feature Selection
       ↓
Feature Scaling
       ↓
Elbow Method
       ↓
K-Means Clustering
       ↓
Cluster Analysis
       ↓
Customer Visualization
       ↓
Streamlit Dashboard
```

---

## 📊 Dataset

The project uses the **Mall Customers** dataset containing:

* Customer ID
* Genre
* Age
* Annual Income
* Spending Score

For clustering, the model uses:

* Age
* Annual Income
* Spending Score

`CustomerID` is not used because it is only an identifier.

---

## 🤖 Machine Learning Model

The project uses **K-Means Clustering**, an unsupervised machine learning algorithm.

K-Means groups customers according to their similarity based on the selected features.

The project also uses the **Elbow Method** to analyze different values of `K` and determine a suitable number of clusters.

### Feature Scaling

Before applying K-Means, the numerical features are standardized using `StandardScaler`.

This is important because K-Means is a distance-based algorithm.

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

---

## 📁 Project Structure

```text
customer-segmentation/
│
├── app.py
├── train_model.py
├── Mall_Customers.csv
├── customer_segmentation_model.pkl
├── clustered_customers.csv
├── elbow.png
├── clusters.png
└── requirements.txt
```

---

## 📈 Features

* Customer data exploration
* Feature scaling using StandardScaler
* K-Means clustering
* Elbow Method
* Cluster analysis
* Customer segment visualization
* New customer segment prediction
* Interactive Streamlit dashboard
* Saved trained ML model

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Rakesh07Dev/customer-segmentation
```

### 2. Enter the project folder

```bash
cd customer-segmentation
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

### 5. Start Streamlit

```bash
python -m streamlit run app.py
```

The dashboard will open at:

```text
http://localhost:8501
```

---

## 🎯 Customer Segmentation

The model identifies groups of customers based on their characteristics.

Examples of possible segments include:

* High income / high spending
* High income / low spending
* Low income / high spending
* Low income / low spending
* Moderate income / moderate spending

The actual characteristics are calculated from the dataset rather than manually assigned.

---

## 📚 What I Learned

This project helped me understand:

* Unsupervised Learning
* Clustering
* K-Means
* Centroids
* Distance-based machine learning
* Feature Scaling
* StandardScaler
* Elbow Method
* Inertia
* Cluster interpretation
* Model persistence
* Streamlit dashboards

---

## 👨‍💻 Author

**Rakesh Singh**

GitHub:
https://github.com/Rakesh07Dev/customer-segmentation

Project Repository:
https://github.com/Rakesh07Dev/customer-segmentation
