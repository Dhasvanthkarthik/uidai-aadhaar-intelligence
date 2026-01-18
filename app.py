import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ================= PAGE CONFIG =================
st.set_page_config(page_title="UIDAI Aadhaar Analytics", layout="wide")
st.title("🇮🇳 UIDAI – Aadhaar Intelligence Dashboard")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    enrol = pd.read_csv("data/enrolment.csv")
    demo = pd.read_csv("data/demographic_updates.csv")
    bio = pd.read_csv("data/biometric_updates.csv")
    return enrol, demo, bio

enrol_df, demo_df, bio_df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("Controls")

dataset = st.sidebar.selectbox(
    "Dataset",
    ["Enrolment", "Demographic Updates", "Biometric Updates"]
)

if dataset == "Enrolment":
    df = enrol_df.copy()
    age_cols = ["age_0_5", "age_5_17", "age_18_greater"]
elif dataset == "Demographic Updates":
    df = demo_df.copy()
    age_cols = ["demo_age_5_17", "demo_age_17_"]
else:
    df = bio_df.copy()
    age_cols = ["bio_age_5_17", "bio_age_17_"]

# ================= DATE FIX =================
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])

state = st.sidebar.selectbox("State", sorted(df["state"].unique()))
df = df[df["state"] == state]

district = st.sidebar.selectbox(
    "District (Optional)",
    ["All"] + sorted(df["district"].unique())
)

if district != "All":
    df = df[df["district"] == district]

age_col = st.sidebar.selectbox("Age Group", age_cols)

# ================= TREND =================
st.subheader("📈 Trend Analysis")

trend = df.groupby("date")[age_col].sum().reset_index().sort_values("date")
st.line_chart(trend.set_index("date")[age_col])

# ================= OPTIMIZED FORECAST =================
st.subheader("🔮 Optimized Forecast (Next 6 Months)")

trend["rolling"] = trend[age_col].rolling(3).mean().fillna(trend[age_col])
ts = trend.set_index("date")["rolling"]

if len(ts) >= 18:
    model = SARIMAX(
        ts,
        order=(1,1,0),
        seasonal_order=(1,0,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    forecast = result.forecast(6)

    forecast_df = pd.DataFrame({
        "date": pd.date_range(ts.index[-1], periods=7, freq="M")[1:],
        "Forecast": forecast.values
    })

    st.line_chart(pd.concat([
        ts,
        forecast_df.set_index("date")["Forecast"]
    ], axis=1))
else:
    st.info("Not enough data for forecasting")

# ================= ANOMALY =================
st.subheader("🚨 Anomaly Detection")

iso = IsolationForest(contamination=0.1, random_state=42)
trend["Anomaly"] = iso.fit_predict(trend[[age_col]])
anomalies = trend[trend["Anomaly"] == -1]

st.dataframe(anomalies)

# ================= ANOMALY EXPLANATION =================
st.subheader("🧠 Anomaly Explanation")

if not anomalies.empty:
    for _, row in anomalies.iterrows():
        ratio = row[age_col] / trend[age_col].mean()
        if ratio > 1.5:
            st.warning(f"Spike on {row['date'].date()} → Possible migration or update drive")
        else:
            st.warning(f"Drop on {row['date'].date()} → Possible system or access issue")
else:
    st.success("No significant anomalies detected")

# ================= SEVERITY =================
st.subheader("🚦 Severity Indicator")

latest = trend[age_col].iloc[-1]
avg = trend[age_col].mean()

if latest > avg * 1.3:
    st.error("🔴 HIGH SEVERITY: Sudden surge detected")
elif latest < avg * 0.7:
    st.warning("🟡 MEDIUM SEVERITY: Activity dip detected")
else:
    st.success("🟢 LOW SEVERITY: Normal activity")

# ================= CLUSTERING =================
st.subheader("🗺️ District Segmentation")

cluster_df = df.groupby("district")[age_col].sum().reset_index()

if len(cluster_df) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_df["Cluster"] = kmeans.fit_predict(cluster_df[[age_col]])
    st.dataframe(cluster_df)
else:
    st.info("Not enough districts for clustering")

# ================= LIFECYCLE VIEW =================
st.subheader("🔄 Aadhaar Lifecycle View")

if dataset == "Enrolment":
    lifecycle = df[["age_0_5", "age_5_17", "age_18_greater"]].sum()
    st.bar_chart(lifecycle)

# ================= COMPARISON MODE =================
st.subheader("📊 Comparison Mode")

compare_state = st.selectbox(
    "Compare With State",
    [s for s in enrol_df["state"].unique() if s != state]
)

compare_df = enrol_df[enrol_df["state"] == compare_state]
compare_df["date"] = pd.to_datetime(compare_df["date"], dayfirst=True, errors="coerce")
compare_trend = compare_df.groupby("date")["age_18_greater"].sum()

st.line_chart(pd.concat([
    trend.set_index("date")[age_col],
    compare_trend
], axis=1))

# ================= INDIA MAP =================
st.subheader("🗺️ India Aadhaar Activity Map")

map_df = enrol_df.groupby("state")["age_18_greater"].sum().reset_index()
map_df.columns = ["state", "value"]

fig = px.choropleth(
    map_df,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/india_states.geojson",
    featureidkey="properties.ST_NM",
    locations="state",
    color="value",
    color_continuous_scale="YlOrRd",
    title="Aadhaar Activity Intensity by State"
)
fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, use_container_width=True)

# ================= DOWNLOAD =================
st.subheader("📥 Download Insight Report")

report = trend.copy()
report["Severity"] = np.where(
    report[age_col] > avg * 1.3, "High",
    np.where(report[age_col] < avg * 0.7, "Medium", "Normal")
)

st.download_button(
    "Download CSV Report",
    report.to_csv(index=False),
    "uidai_insights.csv",
    "text/csv"
)

# ================= FOOTER =================
st.markdown("---")
st.caption("UIDAI Hackathon Prototype | Predictive & Explainable Analytics")
