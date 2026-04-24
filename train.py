import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------------
# 📥 LOAD CLEAN DATA (FROM GITHUB)
# -------------------------------
url = "https://raw.githubusercontent.com/singhnirbhai/mlops-project/refs/heads/main/earthquake_final.csv"
df = pd.read_csv(url, on_bad_lines='skip', engine='python')

print("✅ Clean Dataset Loaded")
print(df.head())

# -------------------------------
# 🧹 DATA CLEANING
# -------------------------------
df = df.dropna()

df["Year"] = df["Year"].astype(int)
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

df = df.dropna()

# -------------------------------
# 🔐 ENCODERS
# -------------------------------

# Encode Country
country_encoder = LabelEncoder()
df["Country_encoded"] = country_encoder.fit_transform(df["Country"])

# Encode Target (Death Category)
label_encoder = LabelEncoder()
df["Death_Category"] = label_encoder.fit_transform(df["Death_Category"])

# Save encoders
joblib.dump(country_encoder, "country_encoder.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# -------------------------------
# 🎯 FEATURES & TARGET
# -------------------------------
X = df[["Country_encoded", "Year", "Latitude", "Longitude"]]
y = df["Death_Category"]

# -------------------------------
# ✂️ SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🤖 MODEL TRAINING
# -------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 📊 EVALUATION
# -------------------------------
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# -------------------------------
# 💾 SAVE MODEL
# -------------------------------
joblib.dump(model, "earthquake_model.pkl")

print("✅ Model saved as earthquake_model.pkl")
print("✅ Encoders saved")
