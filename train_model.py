import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Mall_Customers.csv")

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]


# =========================
# SCALE FEATURES
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================
# ELBOW METHOD
# =========================

inertia = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)

plt.savefig("elbow.png")
plt.close()


# =========================
# FINAL K-MEANS MODEL
# =========================

K = 5

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# =========================
# CLUSTER ANALYSIS
# =========================

summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\n===== CUSTOMER SEGMENTS =====")
print(summary)

print("\n===== CLUSTER SIZES =====")
print(df["Cluster"].value_counts().sort_index())


# =========================
# SAVE MODEL
# =========================

package = {
    "model": kmeans,
    "scaler": scaler,
    "features": features,
    "summary": summary
}

joblib.dump(
    package,
    "customer_segmentation_model.pkl"
)


# =========================
# SAVE CLUSTERED DATA
# =========================

df.to_csv(
    "clustered_customers.csv",
    index=False
)


# =========================
# CLUSTER VISUALIZATION
# =========================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=70
)

# Plot centroids
centers_scaled = kmeans.cluster_centers_

centers_original = scaler.inverse_transform(
    centers_scaled
)

plt.scatter(
    centers_original[:, 1],
    centers_original[:, 2],
    marker="X",
    s=250,
    edgecolors="black"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.grid(True)

plt.savefig("clusters.png")
plt.close()

print("\nModel saved successfully!")
print("Files created:")
print("customer_segmentation_model.pkl")
print("clustered_customers.csv")
print("elbow.png")
print("clusters.png")