import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide"
)


# =========================
# LOAD MODEL
# =========================

package = joblib.load(
    "customer_segmentation_model.pkl"
)

model = package["model"]
scaler = package["scaler"]
features = package["features"]


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "clustered_customers.csv"
)


# =========================
# TITLE
# =========================

st.title("🎯 Customer Segmentation Dashboard")

st.write(
    "K-Means based customer segmentation "
    "using Age, Annual Income and Spending Score."
)


# =========================
# METRICS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "👥 Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "🎯 Number of Clusters",
        df["Cluster"].nunique()
    )

with col3:
    st.metric(
        "💰 Average Income",
        f"${df['Annual Income (k$)'].mean():.1f}k"
    )


st.divider()


# =========================
# CLUSTER SUMMARY
# =========================

st.subheader("📊 Cluster Summary")

summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

st.dataframe(
    summary,
    use_container_width=True
)


# =========================
# CLUSTER DISTRIBUTION
# =========================

st.subheader("👥 Customers per Cluster")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)


# =========================
# VISUALIZATION
# =========================

st.subheader(
    "💰 Income vs Spending Score"
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

scatter = ax.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=70
)

ax.set_xlabel(
    "Annual Income (k$)"
)

ax.set_ylabel(
    "Spending Score (1-100)"
)

ax.set_title(
    "Customer Clusters"
)

ax.grid(True)

st.pyplot(fig)


# =========================
# CUSTOMER PREDICTION
# =========================

st.divider()

st.subheader(
    "🔮 Predict Customer Segment"
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

with col2:
    income = st.number_input(
        "Annual Income (k$)",
        min_value=1,
        max_value=300,
        value=60
    )

with col3:
    spending = st.number_input(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )


if st.button(
    "🎯 Predict Customer Segment"
):

    customer = pd.DataFrame(
        [[
            age,
            income,
            spending
        ]],
        columns=features
    )

    customer_scaled = scaler.transform(
        customer
    )

    cluster = model.predict(
        customer_scaled
    )[0]

    st.success(
        f"Customer belongs to **Cluster {cluster}**"
    )

    st.write(
        "### Cluster Characteristics"
    )

    st.dataframe(
        summary.loc[[cluster]],
        use_container_width=True
    )


# =========================
# ELBOW METHOD
# =========================

st.divider()

st.subheader(
    "📉 Elbow Method"
)

st.image(
    "elbow.png",
    use_container_width=True
)


# =========================
# PROJECT INFO
# =========================

st.divider()

st.subheader(
    "🧠 How this project works"
)

st.markdown("""
1. Load customer data.
2. Select useful numerical features.
3. Standardize the features.
4. Use the Elbow Method to study different K values.
5. Train K-Means clustering.
6. Assign every customer to a cluster.
7. Analyze the characteristics of each cluster.
8. Use the trained model to predict the segment of a new customer.
""")