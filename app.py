import pickle

import numpy as np
import pandas as pd
import streamlit as st

file_path = "./models/xgb_reg.pkl"
# xgb_model_loaded = pickle.load(open(file_path, "rb"))

with open(file_path, "rb") as f:
    model = pickle.load(f)

features = [
    "property_tax_rate",
    "year_built",
    "num_of_appliances",
    "num_of_patio_and_porch_features",
    "lot_size_sq_ft",
    "living_area_sq_ft",
    "avg_school_rating",
    "home_type_Apartment",
    "home_type_Condo",
    "home_type_Mobile / Manufactured",
    "home_type_MultiFamily",
    "home_type_Multiple Occupancy",
    "home_type_Other",
    "home_type_Residential",
    "home_type_Single Family",
    "home_type_Townhouse",
    "home_type_Vacant Land",
    "city_austin",
    "city_del valle",
    "city_driftwood",
    "city_manchaca",
    "city_manor",
    "city_pflugerville",
    "num_of_stories_1",
    "num_of_stories_2",
    "num_of_stories_3",
    "num_of_stories_4",
    "num_of_bathrooms_1.0",
    "num_of_bathrooms_1.5",
    "num_of_bathrooms_1.7",
    "num_of_bathrooms_1.75",
    "num_of_bathrooms_2.0",
    "num_of_bathrooms_2.5",
    "num_of_bathrooms_2.7",
    "num_of_bathrooms_2.75",
    "num_of_bathrooms_3.0",
    "num_of_bathrooms_3.5",
    "num_of_bathrooms_4.0",
    "num_of_bathrooms_4.5",
    "num_of_bathrooms_5.0",
    "num_of_bathrooms_6.0",
    "num_of_bathrooms_7.0",
    "num_of_bathrooms_8.0",
    "num_of_bedrooms_1",
    "num_of_bedrooms_2",
    "num_of_bedrooms_3",
    "num_of_bedrooms_4",
    "num_of_bedrooms_5",
    "num_of_bedrooms_6",
    "num_of_bedrooms_7",
    "num_of_bedrooms_8",
    "num_of_bedrooms_10",
    "num_of_security_features_0",
    "num_of_security_features_1",
    "num_of_security_features_2",
    "num_of_security_features_3",
    "num_of_security_features_4",
    "num_of_security_features_5",
    "num_of_security_features_6",
    "garage_spaces_0",
    "garage_spaces_1",
    "garage_spaces_2",
    "garage_spaces_3",
    "garage_spaces_4",
    "garage_spaces_5",
    "garage_spaces_6",
    "garage_spaces_7",
    "garage_spaces_8",
    "garage_spaces_9",
    "garage_spaces_10",
    "garage_spaces_12",
    "num_of_high_schools_0",
    "num_of_high_schools_1",
    "num_of_high_schools_2",
    "latest_salemonth_1",
    "latest_salemonth_2",
    "latest_salemonth_3",
    "latest_salemonth_4",
    "latest_salemonth_5",
    "latest_salemonth_6",
    "latest_salemonth_7",
    "latest_salemonth_8",
    "latest_salemonth_9",
    "latest_salemonth_10",
    "latest_salemonth_11",
    "latest_salemonth_12",
]

df = pd.DataFrame(
    np.zeros((1, 86)),
    columns=features,
    dtype="int64",
)

# User Inputs
city = st.sidebar.selectbox(
    "Select Your Home Type",
    ("pflugerville", "del valle", "austin", "driftwood", "manor", "manchaca"),
)

zipcode = st.sidebar.number_input(
    "Enter Zipcode", min_value=73301, max_value=78799, step=1
)

df["property_tax_rate"] = st.sidebar.number_input("Enter Property Tax Rate")

num_of_garage_spaces = st.sidebar.slider("Number of garage spaces", max_value=12)

home_type = st.sidebar.selectbox(
    "Select Your Home Type",
    (
        "Single Family",
        "Residential",
        "Townhouse",
        "Condo",
        "Mobile / Manufactured",
        "Multiple Occupancy",
        "Other",
        "Apartment",
        "Vacant Land",
        "MultiFamily",
    ),
)

df["year_built"] = st.sidebar.number_input("Enter Year Built")

df["num_of_appliances"] = st.sidebar.number_input(
    "Enter Number Of Appliances", max_value=11
)

df["num_of_patio_and_porch_features"] = st.sidebar.number_input(
    "Enter Number Of Patio And Porch"
)

num_of_sec = st.sidebar.slider("Number of security features", max_value=6)

df["lot_size_sq_ft"] = st.sidebar.number_input("Enter Lot Size Sq Ft")

df["living_area_sq_ft"] = st.sidebar.number_input("Enter Living Area Sq Ft")

num_of_bed = st.sidebar.slider("Number of bedroom", max_value=10, min_value=1)

num_of_bath = st.sidebar.slider(
    "Number of bathroom", max_value=8.0, min_value=1.0, format="%.1f", step=0.5
)

num_of_stories = st.sidebar.slider(
    "How many stories in your building", max_value=4, min_value=1
)

df["avg_school_rating"] = st.sidebar.slider(
    "Enter School Rating", max_value=10, min_value=1
)

num_of_high_schools = st.sidebar.slider(
    "Number of high schools", max_value=2, min_value=0
)

latest_salemonth = st.sidebar.slider("Latest sale month", max_value=12, min_value=1)

features_col = [
    f"city_{city}",
    f"home_type_{home_type}",
    f"num_of_stories_{num_of_stories}",
    f"num_of_bathrooms_{num_of_bath}",
    f"num_of_bedrooms_{num_of_bed}",
    f"num_of_security_features_{num_of_sec}",
    f"garage_spaces_{num_of_garage_spaces}",
    f"num_of_high_schools_{num_of_high_schools}",
    f"latest_salemonth_{latest_salemonth}",
]

for col in features_col:
    df[col] = 1

st.write(f"Your house price: ${np.exp(model.predict(df)[0])}")
