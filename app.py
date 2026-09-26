import streamlit as st
import pandas as pd
import gzip
import pickle
from datetime import datetime
import time

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="BikeCast",
    page_icon="🚲",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #e8f7ff 0%,
        #eefcf7 45%,
        #f3f0ff 100%
    );
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Title */

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #176b87;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #527784;
    margin-bottom: 30px;
}

/* Section Cards */

.section-card {
    background: rgba(255, 255, 255, 0.65);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(120, 190, 200, 0.35);
    box-shadow: 0 8px 25px rgba(60, 120, 140, 0.10);
    margin-bottom: 20px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #176b87;
    margin-bottom: 15px;
}

/* Labels */

label {
    font-weight: 600 !important;
    color: #315b68 !important;
}

/* Input boxes */

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.75);
    border-radius: 10px;
}

div[data-testid="stNumberInput"] input {
    background-color: rgba(255,255,255,0.75);
    border-radius: 10px;
}

/* Button */

.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button {
    background: linear-gradient(
        90deg,
        #35a9c7,
        #3bbf9a
    );
    color: white;
    font-size: 18px;
    font-weight: 700;
    padding: 12px 35px;
    border-radius: 30px;
    border: none;
    box-shadow: 0 6px 18px rgba(50, 160, 160, 0.25);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(50, 160, 160, 0.35);
}

/* Result */

.result-card {
    background: linear-gradient(
        135deg,
        #dff8f4,
        #e1f4ff
    );
    padding: 30px;
    border-radius: 25px;
    text-align: center;
    border: 1px solid #a8ded7;
    box-shadow: 0 10px 30px rgba(50, 140, 150, 0.15);
    margin-top: 25px;
}

.result-title {
    font-size: 20px;
    font-weight: 600;
    color: #3b6873;
}

.result-number {
    font-size: 52px;
    font-weight: 800;
    color: #167c83;
    margin: 5px 0;
}

.result-text {
    font-size: 17px;
    color: #527784;
}

/* Information cards */

.info-card {
    background: rgba(255,255,255,0.55);
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(120,190,200,0.25);
}

.info-title {
    font-size: 14px;
    color: #66828b;
}

.info-value {
    font-size: 20px;
    font-weight: 700;
    color: #246b78;
}

/* Footer */

.footer {
    text-align: center;
    color: #6c8992;
    font-size: 14px;
    margin-top: 35px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD MODEL, SCALER AND FEATURES
# ---------------------------------------------------

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown("""
<div class="main-title">🚲 BikeCast</div>

<div class="subtitle">
    Bike + Forecast | Smart Rental Demand Prediction
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.markdown("""
<div class="section-card">
<div class="section-title">🌤️ Weather & Conditions</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    season = st.selectbox(
        "Season",
        ["Spring", "Summer", "Fall", "Winter"]
    )

    weather = st.selectbox(
        "Weather Situation",
        [
            "Clear",
            "Mist",
            "Light Rain/Snow",
            "Heavy Rain/Snow"
        ]
    )

    temp = st.number_input(
        "Temperature (normalized)",
        min_value=0.02,
        max_value=1.00,
        value=0.50,
        step=0.01
    )


with col2:

    hum = st.number_input(
        "Humidity (normalized)",
        min_value=0.08,
        max_value=1.00,
        value=0.64,
        step=0.01
    )

    windspeed = st.number_input(
        "Windspeed (normalized)",
        min_value=0.00,
        max_value=0.8507,
        value=0.19,
        step=0.01
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# TIME & CALENDAR SECTION
# ---------------------------------------------------

st.markdown("""
<div class="section-card">
<div class="section-title">🕒 Time & Calendar</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    hour_options = {
        "12:00 AM": 0,
        "1:00 AM": 1,
        "2:00 AM": 2,
        "3:00 AM": 3,
        "4:00 AM": 4,
        "5:00 AM": 5,
        "6:00 AM": 6,
        "7:00 AM": 7,
        "8:00 AM": 8,
        "9:00 AM": 9,
        "10:00 AM": 10,
        "11:00 AM": 11,
        "12:00 PM": 12,
        "1:00 PM": 13,
        "2:00 PM": 14,
        "3:00 PM": 15,
        "4:00 PM": 16,
        "5:00 PM": 17,
        "6:00 PM": 18,
        "7:00 PM": 19,
        "8:00 PM": 20,
        "9:00 PM": 21,
        "10:00 PM": 22,
        "11:00 PM": 23
    }

    selected_time = st.selectbox(
        "Time of Day",
        list(hour_options.keys())
    )

    hour = hour_options[selected_time]


with col2:

    weekday_options = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6
    }

    selected_weekday = st.selectbox(
        "Day of the Week",
        list(weekday_options.keys())
    )

    weekday = weekday_options[selected_weekday]


col1, col2 = st.columns(2)

with col1:

    holiday = st.selectbox(
        "Is it a Holiday?",
        ["No", "Yes"]
    )


with col2:

    working_day = st.selectbox(
        "Is it a Working Day?",
        ["No", "Yes"]
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🚲 Predict Bike Rental Demand")


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_button:

    with st.spinner("🔮 BikeCast is analyzing the conditions..."):

        time.sleep(1)

        # Automatically use current year/month
        yr = 1
        mnth = datetime.now().month

        # Convert Yes/No to 0/1

        holiday_value = 0 if holiday == "No" else 1

        workingday_value = 0 if working_day == "No" else 1

        # ---------------------------------------------------
        # CREATE INPUT DATA
        # ---------------------------------------------------

        input_data = pd.DataFrame({
            "season": [season],
            "yr": [yr],
            "mnth": [mnth],
            "hr": [hour],
            "holiday": [holiday_value],
            "weekday": [weekday],
            "workingday": [workingday_value],
            "weathersit": [weather],
            "temp": [temp],
            "hum": [hum],
            "windspeed": [windspeed]
        })

        # ---------------------------------------------------
        # CATEGORICAL FEATURES
        # ---------------------------------------------------

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

        # ---------------------------------------------------
        # SET CATEGORY RANGES
        # ---------------------------------------------------

        input_data["season"] = pd.Categorical(
            input_data["season"],
            categories=[
                "Fall",
                "Spring",
                "Summer",
                "Winter"
            ]
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

        # ---------------------------------------------------
        # ONE-HOT ENCODING
        # ---------------------------------------------------

        input_encoded = pd.get_dummies(
            input_data,
            columns=categorical_features,
            drop_first=True
        )

        # Make sure the input has exactly the same
        # 52 features as the training data

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ---------------------------------------------------
        # SCALE NUMERICAL FEATURES
        # ---------------------------------------------------

        input_encoded[numerical_features] = scaler.transform(
            input_encoded[numerical_features]
        )

        # ---------------------------------------------------
        # MAKE PREDICTION
        # ---------------------------------------------------

        prediction = model.predict(input_encoded)[0]

        prediction = round(prediction)

        # ---------------------------------------------------
        # RESULT
        # ---------------------------------------------------

        st.balloons()

        st.markdown(f"""
        <div class="result-card">

            <div class="result-title">
                🚲 BikeCast Prediction
            </div>

            <div class="result-number">
                {prediction}
            </div>

            <div class="result-text">
                Estimated bike rentals for the selected conditions
            </div>

        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # SUMMARY CARDS
        # ---------------------------------------------------

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(f"""
            <div class="info-card">

                <div class="info-title">
                    🌤️ Season
                </div>

                <div class="info-value">
                    {season}
                </div>

            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="info-card">

                <div class="info-title">
                    🕒 Time
                </div>

                <div class="info-value">
                    {selected_time}
                </div>

            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class="info-card">

                <div class="info-title">
                    ☁️ Weather
                </div>

                <div class="info-value">
                    {weather}
                </div>

            </div>
            """, unsafe_allow_html=True)

        with col4:

            st.markdown(f"""
            <div class="info-card">

                <div class="info-title">
                    📅 Day
                </div>

                <div class="info-value">
                    {selected_weekday}
                </div>

            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<div class="footer">

🚲 <b>BikeCast</b> — Bike + Forecast<br>
Powered by Machine Learning & Random Forest Regression

</div>
""", unsafe_allow_html=True)
