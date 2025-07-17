import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Supply Chain Demand Predictor", layout="wide")
st.title("Supply Chain Optimization - Predictive Demand Analysis")

# Load dataset
data = pd.read_csv("supply_chain_data.csv")
st.subheader("Raw Data Preview")
st.dataframe(data.head())

# Drop missing target rows
data.dropna(subset=["Number of products sold"], inplace=True)

# Select features & target
target = "Number of products sold"
features = [
    "Product type", "Price", "Stock levels", "Lead times", "Order quantities",
    "Location", "Production volumes", "Manufacturing lead time", "Manufacturing costs",
    "Defect rates", "Transportation modes", "Costs"
]

# Encode categorical columns
label_encoders = {}
categorical = data[features].select_dtypes(include="object").columns
for col in categorical:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "demand_model.pkl")

# Evaluate
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

st.markdown("### 📊 Model Performance")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Feature Importance
importance = pd.Series(model.feature_importances_, index=features).sort_values()
fig1, ax1 = plt.subplots()
importance.plot(kind="barh", ax=ax1)
ax1.set_title("Feature Importance")
st.pyplot(fig1)

# Prediction Form
st.markdown("### 🔮 Predict Demand")
user_input = {}
for col in features:
    if col in categorical:
        options = label_encoders[col].classes_
        val = st.selectbox(f"{col}", options)
        user_input[col] = label_encoders[col].transform([val])[0]
    else:
        user_input[col] = st.number_input(f"{col}", value=float(data[col].median()))

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Number of Products Sold: {prediction:.0f}")
