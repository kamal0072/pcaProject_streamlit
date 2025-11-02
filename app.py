# import pickeled file
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.express as px

# ----------------------------
# Title and Description
# ----------------------------
st.markdown("<p > This app lets you **perform Principal Component Analysis (PCA)** on any dataset.  You can upload your own CSV file or use the built-in **Iris dataset**.  It shows how PCA reduces dimensions while retaining most variance.</p>", unsafe_allow_html=True)


# ----------------------------
# Upload or Use Default Dataset
# ----------------------------

upoaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
# print(upoaded_file)
if upoaded_file is not None:
    df = pd.read_csv(upoaded_file)
    st.success("File uploaded successfully")
else:
    st.info("Use the default Iris dataset")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

st.write("### 🧾 Dataset Preview")
st.dataframe(df.head())


# ----------------------------
# Select Features
# ----------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
selected_features = st.multiselect(
    "Select Features for PCA", numeric_cols, default=numeric_cols[:-1])

if len(selected_features) < 2:
    st.warning('Please select at least two features for PCA.')
    st.stop()

X = df[selected_features]


# ----------------------------
# Standardize the Data
# ----------------------------
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

# ----------------------------
# PCA Parameters
# ----------------------------
n_components = st.slider("Select Number of Principal Components",
                         min_value=2, max_value=min(len(selected_features), 10), value=2)

# ----------------------------
# Apply PCA
# ----------------------------
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scale)


# ----------------------------
# Explained Variance
# ----------------------------
st.write("### 📊 Explained Variance Ratio")
explained_var = np.round(pca.explained_variance_ratio_*100, 2)
st.bar_chart(pd.DataFrame(explained_var, columns=["% Variance Explained"]))

# ----------------------------
# Combine Results
# ----------------------------
pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
if "taget" in df.columns:
    pca_df["Target"] = df["target"].astype(str)

# ----------------------------
# Visualization
# ----------------------------
st.write("### 🎨 PCA Visualization")

if n_components > 3:
    fig = px.scatter_3d(
        pca_df, x="PC1", y="PC2", z="PC3",
        color=pca_df["Target"] if "Target" in pca_df.columns else None,
        title="3D PCA Plot",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    fig = px.scatter(
        pca_df, x="PC1", y="PC2",
        color=pca_df["Target"] if "Target" in pca_df.columns else None,
        title="2D PCA Plot",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Display Download Option
# ----------------------------
st.write("### 📥 Download PCA Results")
csv = pca_df.to_csv(index=False).encode('utf-8')
if st.download_button(
    label="Download CSV",
    data=csv,
    file_name="pca_transformed_data.csv",
    mime="text/csv",
):
    st.success("✅ Download successful!")