import streamlit as st
import pandas as pd
import gzip
import pickle
from datetime import datetime
import time


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)


# ==========================================
# CUSTOM LIGHT THEME
# ==========================================

st.markdown("""
<style>

/* Main page */
.stApp {
    background: linear-gradient(
        135deg,
        #f4fbff 0%,
        #f7fcf9 50%,
        #eef8ff 100%
    );
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* Main title */
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #164e63;
    margin-bottom: 5px;
}


/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #52717d;
    margin-bottom: 35px;
}


/* Section cards */
.section-card {
    background: rgba(255, 255, 255, 0.92);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid #dceff5;
    box-shadow: 0 8px 25px rgba(60, 130, 150, 0.08);
    margin-bottom: 20px;
}


/* Section headings */
.section-title {
    color: #155e75;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 20px;
}


/* Input labels */
label {
    color: #365a63 !important;
    font-weight: 600 !important;
}


/* Prediction button */
.stButton > button {
    background: linear-gradient(
        90deg,
        #38bdf8,
        #2dd4bf
    );
    color: white;
    border: none;
    border-radius: 14px;
    padding: 14px 25px;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
    box-shadow: 0 6px 15px rgba(45, 212, 191, 0.25);
}


/* Button hover */
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(45, 212, 191, 0.35);
}


/* Result card */
.result-card {
    background: linear-gradient(
        135deg,
        #ecfeff,
        #ecfdf5
    );
    border: 2px solid #99f6e4;
    border-radius: 24px;
    padding: 35px;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 12px 30px rgba(20, 184, 166, 0.12);
}


/* Result title */
.result-title {
    color: #115e59;
    font-size: 24px;
    font-weight: 700;
}


/* Result number */
.result-number {
    color: #0f766e;
    font-size: 55px;
    font-weight: 800;
    margin: 10px 0;
}


/* Result description */
.result-text {
    color: #52717d;
    font-size: 17px;
}


/* Small info cards */
.info-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    border: 1px solid #dceff5;
    box-shadow: 0 5px 15px rgba(60, 130, 150, 0.06);
}


/* Footer */
.footer {
    text-align: center;
    color: #72909a;
    margin-top: 40px;
    font-size: 14px;
}


/* Divider */
hr {
    border: none;
    border-top: 1px solid #d8edf2;
    margin: 30px 0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="main-title">🚲 Bike Rental Demand Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered prediction of expected bike rental demand'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# LOAD MODEL
# ==========================================

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)


# ==========================================
# LOAD SCALER
# ==========================================

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ==========================================
# LOAD FEATURE COLUMNS
# ==========================================

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# ==========================================
# TWO COLUMN LAYOUT
# ==========================================

left_column, right_column = st.columns(2, gap="large")


# ==========================================
# WEATHER SECTION
# ==========================================

with left_column:

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🌦️ Weather Conditions</div>',
        unsafe_allow_html=True
    )

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

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================
# TIME SECTION
# ==========================================

with right_column:

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🕒 Time & Calendar</div>',
        unsafe_allow_html=True
    )


    # Time options

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


    # Weekday

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


    # Holiday

    holiday = st.selectbox(
        "Is it a Holiday?",
        ["No", "Yes"]
    )


    # Working day

    working_day = st.selectbox(
        "Is it a Working Day?",
        ["No", "Yes"]
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)

button_left, button_center, button_right = st.columns(
    [1, 2, 1]
)

with button_center:

    predict_button = st.button(
        "🚲  Predict Bike Rental Demand",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    # Loading animation

    with st.spinner(
        "🔍 Analyzing weather, time and calendar conditions..."
    ):

        time.sleep(1)


        # ----------------------------------
        # Fixed year
        # ----------------------------------

        yr = 1


        # ----------------------------------
        # Current month
        # ----------------------------------

        mnth = datetime.now().month


        # ----------------------------------
        # Holiday conversion
        # ----------------------------------

        if holiday == "No":
            holiday_value = 0
        else:
            holiday_value = 1


        # ----------------------------------
        # Working day conversion
        # ----------------------------------

        if working_day == "No":
            workingday_value = 0
        else:
            workingday_value = 1


        # ----------------------------------
        # Create input
        # ----------------------------------

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


        # ----------------------------------
        # Feature lists
        # ----------------------------------

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


        # ----------------------------------
        # Set categories
        # ----------------------------------

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


        # ----------------------------------
        # One-hot encoding
        # ----------------------------------

        input_encoded = pd.get_dummies(

            input_data,

            columns=categorical_features,

            drop_first=True

        )


        # ----------------------------------
        # Match training columns
        # ----------------------------------

        input_encoded = input_encoded.reindex(

            columns=feature_columns,

            fill_value=0

        )


        # ----------------------------------
        # Scale numerical values
        # ----------------------------------

        input_encoded[numerical_features] = scaler.transform(

            input_encoded[numerical_features]

        )


        # ----------------------------------
        # Prediction
        # ----------------------------------

        prediction = model.predict(

            input_encoded

        )[0]


        prediction = round(prediction)


    # ======================================
    # RESULT
    # ======================================

    st.balloons()


    st.markdown(

        '<div class="result-card">',

        unsafe_allow_html=True

    )


    st.markdown(

        '<div class="result-title">'
        '🎯 Predicted Bike Rental Demand'
        '</div>',

        unsafe_allow_html=True

    )


    st.markdown(

        f'<div class="result-number">'
        f'{prediction} 🚲'
        f'</div>',

        unsafe_allow_html=True

    )


    st.markdown(

        '<div class="result-text">'
        'bikes are expected to be rented under these conditions.'
        '</div>',

        unsafe_allow_html=True

    )


    st.markdown(

        '</div>',

        unsafe_allow_html=True

    )


    # ======================================
    # SUMMARY
    # ======================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📊 Prediction Summary"
    )


    summary1, summary2, summary3 = st.columns(3)


    with summary1:

        st.markdown(
            f"""
            <div class="info-card">
                <b>🌤️ Season</b>
                <br><br>
                {season}
            </div>
            """,
            unsafe_allow_html=True
        )


    with summary2:

        st.markdown(
            f"""
            <div class="info-card">
                <b>🕒 Time</b>
                <br><br>
                {selected_time}
            </div>
            """,
            unsafe_allow_html=True
        )


    with summary3:

        st.markdown(
            f"""
            <div class="info-card">
                <b>☁️ Weather</b>
                <br><br>
                {weather}
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        Built with Python • Streamlit • Random Forest Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
