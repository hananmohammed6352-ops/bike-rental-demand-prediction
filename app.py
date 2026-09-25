import streamlit as st
import pandas as pd
import gzip
import pickle

st.title("Bike Rental Demand Prediction")

# Load trained model
with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature columns
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


st.subheader("Enter Bike Rental Details")


# -----------------------------
# User Inputs
# -----------------------------

season = st.selectbox(
    "Season",
    ["Spring", "Summer", "Fall", "Winter"]
)

year = st.selectbox(
    "Year",
    [2011, 2012]
)

month = st.selectbox(
    "Month",
    list(range(1, 13))
)

hour = st.selectbox(
    "Hour",
    list(range(24))
)

holiday = st.selectbox(
    "Holiday",
    ["No", "Yes"]
)

weekday = st.selectbox(
    "Weekday",
    [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday"
    ]
)

working_day = st.selectbox(
    "Working Day",
    ["No", "Yes"]
)

weather = st.selectbox(
    "Weather Situation",
    ["Clear", "Mist", "Light Rain/Snow", "Heavy Rain/Snow"]
)


# These are the normalized values used by your dataset
temp = st.number_input(
    "Temperature (normalized)",
    min_value=0.02,
    max_value=1.00,
    value=0.50
)

hum = st.number_input(
    "Humidity (normalized)",
    min_value=0.08,
    max_value=1.00,
    value=0.64
)

windspeed = st.number_input(
    "Windspeed (normalized)",
    min_value=0.00,
    max_value=0.8507,
    value=0.19
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Bike Rental Demand"):

    # Convert user-friendly values
    # back to the values used by the model

    if year == 2011:
        yr = 0
    else:
        yr = 1

    if holiday == "No":
        holiday_value = 0
    else:
        holiday_value = 1

    if working_day == "No":
        workingday_value = 0
    else:
        workingday_value = 1

    weekday_values = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6
    }

    weekday_value = weekday_values[weekday]


    # Create original input structure
    input_data = pd.DataFrame({
        "season": [season],
        "yr": [yr],
        "mnth": [month],
        "hr": [hour],
        "holiday": [holiday_value],
        "weekday": [weekday_value],
        "workingday": [workingday_value],
        "weathersit": [weather],
        "temp": [temp],
        "hum": [hum],
        "windspeed": [windspeed]
    })


    # Same categorical features used during training
    categorical_features = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit"
    ]

    numerical_features = [
        "temp",
        "hum",
        "windspeed"
    ]


    # Give categorical columns their original categories
    # so get_dummies behaves exactly like training

    input_data["season"] = pd.Categorical(
        input_data["season"],
        categories=["Fall", "Spring", "Summer", "Winter"]
    )

    input_data["yr"] = pd.Categorical(
        input_data["yr"],
        categories=[0, 1]
    )

    input_data["mnth"] = pd.Categorical(
        input_data["mnth"],
        categories=list(range(1, 13))
    )

    input_data["hr"] = pd.Categorical(
        input_data["hr"],
        categories=list(range(24))
    )

    input_data["holiday"] = pd.Categorical(
        input_data["holiday"],
        categories=[0, 1]
    )

    input_data["weekday"] = pd.Categorical(
        input_data["weekday"],
        categories=list(range(7))
    )

    input_data["workingday"] = pd.Categorical(
        input_data["workingday"],
        categories=[0, 1]
    )

    input_data["weathersit"] = pd.Categorical(
        input_data["weathersit"],
        categories=[
            "Clear",
            "Heavy Rain/Snow",
            "Light Rain/Snow",
            "Mist"
        ]
    )


    # One-hot encoding
    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_features,
        drop_first=True
    )


    # Make sure columns are exactly the same
    # as the training data
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Scale numerical features
    input_encoded[numerical_features] = scaler.transform(
        input_encoded[numerical_features]
    )


    # Make prediction
    prediction = model.predict(input_encoded)[0]


    st.success(
        f"Predicted Bike Rental Demand: {round(prediction)} bikes"
    )
