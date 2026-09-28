import streamlit as st 
import pandas as pd 
import numpy as np 
from datetime import datetime 
import time 
import joblib 
 
# --------------------------------------------------- 
# PAGE SETTINGS 
# --------------------------------------------------- 
 
st.set_page_config( 
    page_title="BikeCast", 
    page_icon="🚲", 
    layout="wide" 
) 
 
# --------------------------------------------------- 
# CUSTOM CSS - DARK MAROON THEME 
# --------------------------------------------------- 
 
st.markdown(""" 
<style> 
 
.stApp { 
    background: 
        radial-gradient( 
            circle at top left, 
            #54243a 0%, 
            #321622 40%, 
            #1e1017 100% 
        ); 
    color: #f8eee8; 
} 
 
.block-container { 
    max-width: 1200px; 
    padding-top: 2rem; 
    padding-bottom: 2rem; 
} 
 
.main-title { 
    text-align: center; 
    font-size: 52px; 
    font-weight: 800; 
    color: #ffe9d6; 
    margin-bottom: 4px; 
    letter-spacing: 1px; 
} 
 
.subtitle { 
    text-align: center; 
    font-size: 19px; 
    color: #e9b9aa; 
    margin-bottom: 35px; 
} 
 
.section-box { 
    background: rgba(76, 31, 49, 0.88); 
    padding: 26px; 
    border-radius: 22px; 
    border: 1px solid rgba(231, 174, 151, 0.22); 
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.28); 
    margin-bottom: 25px; 
} 
 
.section-heading { 
    font-size: 24px; 
    font-weight: 700; 
    color: #ffd9c4; 
    margin-bottom: 20px; 
} 
 
label { 
    color: #f6ddd3 !important; 
    font-weight: 600 !important; 
} 
 
div[data-baseweb="select"] > div { 
    background-color: #321923; 
    border: 1px solid #714052; 
    border-radius: 10px; 
    color: #fff2ea; 
} 
 
div[data-baseweb="select"] span { 
    color: #fff2ea !important; 
} 
 
div[data-testid="stNumberInput"] input { 
    background-color: #321923; 
    color: #fff2ea; 
    border: 1px solid #714052; 
    border-radius: 10px; 
} 
 
.stButton { 
    display: flex; 
    justify-content: center; 
} 
 
.stButton > button { 
    background: linear-gradient( 
        90deg, 
        #a94f68, 
        #c56a72 
    ); 
    color: #fff7f2; 
    font-size: 18px; 
    font-weight: 700; 
    padding: 13px 38px; 
    border-radius: 30px; 
    border: 1px solid #d88b87; 
    box-shadow: 
        0 8px 25px rgba(180, 70, 100, 0.30); 
    transition: 0.3s; 
} 
 
.stButton > button:hover { 
    background: linear-gradient( 
        90deg, 
        #c56a72, 
        #d98580 
    ); 
    color: white; 
    transform: translateY(-2px); 
} 
 
.result-box { 
    background: 
        linear-gradient( 
            135deg, 
            #55243a, 
            #3b1b2b 
        ); 
    padding: 32px; 
    border-radius: 25px; 
    text-align: center; 
    border: 1px solid #9a5968; 
    box-shadow: 
        0 12px 35px rgba(0, 0, 0, 0.35); 
    margin-top: 25px; 
} 
 
.result-heading { 
    font-size: 22px; 
    font-weight: 600; 
    color: #f5cbb9; 
} 
 
.result-number { 
    font-size: 58px; 
    font-weight: 800; 
    color: #ffe3cf; 
    margin: 5px 0; 
} 
 
.result-description { 
    font-size: 17px; 
    color: #ddb2a5; 
} 
 
.summary-box { 
    background: rgba(76, 31, 49, 0.85); 
    padding: 19px; 
    border-radius: 18px; 
    text-align: center; 
    border: 1px solid rgba(218, 139, 130, 0.22); 
    box-shadow: 
        0 8px 20px rgba(0, 0, 0, 0.20); 
} 
 
.summary-title { 
    font-size: 14px; 
    color: #d7aaa0; 
} 
 
.summary-value { 
    font-size: 18px; 
    font-weight: 700; 
    color: #ffe1cf; 
    margin-top: 5px; 
} 
 
.footer { 
    text-align: center; 
    color: #b9918c; 
    font-size: 14px; 
    margin-top: 40px; 
    padding-bottom: 10px; 
} 
 
</style> 
""", unsafe_allow_html=True) 
 
# --------------------------------------------------- 
# LOAD MODEL AND SCALER 
# --------------------------------------------------- 
 
model = joblib.load("model.pkl.gz") 
 
scaler = joblib.load("scaler.pkl") 
 
# Get the exact 52 features used during training 
feature_columns = [ 
    'temp', 'hum', 'windspeed', 
    'season_Spring', 'season_Summer', 'season_Winter', 
    'yr_1', 
    'mnth_2', 'mnth_3', 'mnth_4', 'mnth_5', 'mnth_6', 
    'mnth_7', 'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12', 
    'hr_1', 'hr_2', 'hr_3', 'hr_4', 'hr_5', 'hr_6', 
    'hr_7', 'hr_8', 'hr_9', 'hr_10', 'hr_11', 'hr_12', 
    'hr_13', 'hr_14', 'hr_15', 'hr_16', 'hr_17', 'hr_18', 
    'hr_19', 'hr_20', 'hr_21', 'hr_22', 'hr_23', 
    'holiday_1', 
    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 
    'weekday_5', 'weekday_6', 
    'workingday_1', 
    'weathersit_Heavy Rain/Snow', 
    'weathersit_Light Rain/Snow', 
    'weathersit_Mist' 
] 
 
# --------------------------------------------------- 
# TITLE 
# --------------------------------------------------- 
 
st.markdown( 
    '<div class="main-title">🚲 BikeCast</div>', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="subtitle">' 
    'Bike + Forecast | Smart Rental Demand Prediction' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
# --------------------------------------------------- 
# WEATHER SECTION 
# --------------------------------------------------- 
 
st.markdown( 
    '<div class="section-box">', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="section-heading">🌤️ Weather & Conditions</div>', 
    unsafe_allow_html=True 
) 
 
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
 
st.markdown( 
    '</div>', 
    unsafe_allow_html=True 
) 
 
# --------------------------------------------------- 
# TIME & CALENDAR 
# --------------------------------------------------- 
 
st.markdown( 
    '<div class="section-box">', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="section-heading">🕒 Time & Calendar</div>', 
    unsafe_allow_html=True 
) 
 
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
 
weekday_options = { 
    "Sunday": 0, 
    "Monday": 1, 
    "Tuesday": 2, 
    "Wednesday": 3, 
    "Thursday": 4, 
    "Friday": 5, 
    "Saturday": 6 
} 
 
col1, col2 = st.columns(2) 
 
with col1: 
 
    selected_time = st.selectbox( 
        "Time of Day", 
        list(hour_options.keys()) 
    ) 
 
    hour = hour_options[selected_time] 
 
with col2: 
 
    selected_weekday = st.selectbox( 
        "Day of the Week", 
        list(weekday_options.keys()) 
    ) 
 
    weekday = weekday_options[selected_weekday] 
 
st.markdown( 
    '</div>', 
    unsafe_allow_html=True 
) 
 
# --------------------------------------------------- 
# PREDICTION BUTTON 
# --------------------------------------------------- 
 
st.markdown("<br>", unsafe_allow_html=True) 
 
if st.button("🚲 Predict Bike Rental Demand"): 
 
    with st.spinner("BikeCast is analyzing the conditions..."): 
 
        time.sleep(1) 
 
        # Current year and month 
        yr = 1 
        mnth = datetime.now().month 
 
        # Holiday and working day are automatically determined 
        holiday_value = 0 
 
        if weekday in [1, 2, 3, 4, 5]: 
            workingday_value = 1 
        else: 
            workingday_value = 0 
 
        # --------------------------------------------------- 
        # INPUT DATA 
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
        # FEATURES 
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
        # CATEGORY SETTINGS 
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
        # ONE HOT ENCODING 
        # --------------------------------------------------- 
 
        input_encoded = pd.get_dummies( 
            input_data, 
            columns=categorical_features, 
            drop_first=True 
        ) 
 
        # Make sure input has exactly the same 52 columns 
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
        # PREDICTION 
        # --------------------------------------------------- 
 
        prediction = model.predict(input_encoded)[0] 
 
        prediction = round(prediction) 
 
    # --------------------------------------------------- 
    # RESULT 
    # --------------------------------------------------- 
 
    st.markdown( 
        '<div class="result-box">', 
        unsafe_allow_html=True 
    ) 
 
    st.markdown( 
        '<div class="result-heading">' 
        '🚲 BikeCast Prediction' 
        '</div>', 
        unsafe_allow_html=True 
    ) 
 
    st.markdown( 
        f'<div class="result-number">{prediction}</div>', 
        unsafe_allow_html=True 
    ) 
 
    st.markdown( 
        '<div class="result-description">' 
        'Estimated bike rentals for the selected conditions' 
        '</div>', 
        unsafe_allow_html=True 
    ) 
 
    st.markdown( 
        '</div>', 
        unsafe_allow_html=True 
    ) 
 
    # --------------------------------------------------- 
    # SUMMARY 
    # --------------------------------------------------- 
 
    st.markdown("<br>", unsafe_allow_html=True) 
 
    col1, col2, col3, col4 = st.columns(4) 
 
    with col1: 
 
        st.markdown( 
            f''' 
            <div class="summary-box"> 
                <div class="summary-title">🌤️ Season</div> 
                <div class="summary-value">{season}</div> 
            </div> 
            ''', 
            unsafe_allow_html=True 
        ) 
 
    with col2: 
 
        st.markdown( 
            f''' 
            <div class="summary-box"> 
                <div class="summary-title">🕒 Time</div> 
                <div class="summary-value">{selected_time}</div> 
            </div> 
            ''', 
            unsafe_allow_html=True 
        ) 
 
    with col3: 
 
        st.markdown( 
            f''' 
            <div class="summary-box"> 
                <div class="summary-title">☁️ Weather</div> 
                <div class="summary-value">{weather}</div> 
            </div> 
            ''', 
            unsafe_allow_html=True 
        ) 
 
    with col4:
 
        st.markdown( 
            f''' 
            <div class="summary-box"> 
                <div class="summary-title">📅 Day</div> 
                <div class="summary-value">{selected_weekday}</div> 
            </div> 
            ''', 
            unsafe_allow_html=True 
        ) 
 
# --------------------------------------------------- 
# FOOTER 
# --------------------------------------------------- 
 
st.markdown( 
    '<div class="footer">' 
    '🚲 <b>BikeCast</b> — Bike + Forecast<br>' 
    'Powered by Machine Learning & Random Forest Regression' 
    '</div>', 
    unsafe_allow_html=True 
)
