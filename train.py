import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/earthquake.csv"
df = pd.read_csv(url)

print("✅ Dataset Loaded")
print("Columns:", df.columns.tolist())

# Define categories
categories = [
    "0 Deaths",
    "1-50 Deaths",
    "51-100 Deaths",
    "101-1000 Deaths",
    ">1001 Deaths"
]

# Restructure dataset
data = []

for _, row in df.iterrows():
    for cat in categories:
        lat_col = f"{cat}, lat"
        lon_col = f"{cat}, lon"

        lat = row.get(lat_col)
        lon = row.get(lon_col)

        if pd.notna(lat) and pd.notna(lon):
            data.append({
                "lat": lat,
                "lon": lon,
                "label": cat
            })

# Convert to DataFrame
new_df = pd.DataFrame(data)

print("✅ Restructured Data Size:", new_df.shape)

# -------------------------------
# 🔥 DATA CLEANING (IMPORTANT)
# -------------------------------

# Convert to numeric safely
new_df["lat"] = pd.to_numeric(new_df["lat"], errors="coerce")
new_df["lon"] = pd.to_numeric(new_df["lon"], errors="coerce")

# Drop invalid rows
new_df = new_df.dropna()

# Reset index
new_df = new_df.reset_index(drop=True)

print("✅ Cleaned Data Size:", new_df.shape)

# -------------------------------
# 🔐 LABEL ENCODING
# -------------------------------
le = LabelEncoder()
new_df["label"] = le.fit_transform(new_df["label"])

# Save encoder (VERY IMPORTANT for API)
joblib.dump(le, "label_encoder.pkl")

# -------------------------------
# 🎯 FEATURES & TARGET
# -------------------------------
X = new_df[["lat", "lon"]]
y = new_df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🤖 MODEL TRAINING
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# -------------------------------
# 💾 SAVE MODEL
# -------------------------------
joblib.dump(model, "earthquake_model.pkl")

print("✅ Model saved as earthquake_model.pkl")
print("✅ Encoder saved as label_encoder.pkl")
