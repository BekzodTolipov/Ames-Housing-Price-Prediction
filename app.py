import datetime
import pickle

import geopy.distance
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim


def cal_distance_miles(lat, long, coords_2):
    """Uses python geopy library to calculate distance between 2 coordinates

    Args:
        lat (float): latitude of given coordinate
        long (float): longitude of given coordinate

    Returns:
        _type_: distance between 2 coordinates
    """
    house_coords = (lat, long)
    return round(geopy.distance.geodesic(house_coords, coords_2).miles, 2)


def airport_distance(lat, long):
    austin_airport_coords = (30.1975, -97.6663058)
    return cal_distance_miles(lat, long, austin_airport_coords)


def downtown_distance(lat, long):
    austin_airport_coords = (30.266666, -97.733330)
    return cal_distance_miles(lat, long, austin_airport_coords)


# GRAPHS BELOW
austin_df = pd.read_csv("./data/austin_housing_data.csv")


feature_one = st.selectbox(
    "Select Feature 1",
    (x for x in austin_df.columns.tolist()),
)

max_val_one = round(austin_df[feature_one].max())
min_val_one = round(austin_df[feature_one].min())

house_p_min, house_p_max = st.slider(
    f"{feature_one}", min_val_one, max_val_one, (min_val_one, max_val_one), step=10_000
)

austin_df = austin_df[
    (austin_df[feature_one] > house_p_min) & (austin_df[feature_one] < house_p_max)
]

feature_two = st.selectbox(
    "Select Feature 2",
    (x for x in austin_df.columns.tolist()),
)

max_val_two = round(austin_df[feature_two].max())
min_val_two = round(austin_df[feature_two].min())

house_p_min, house_p_max = st.slider(
    f"{feature_two}", min_val_two, max_val_two, (min_val_two, max_val_two), step=10_000
)

austin_df = austin_df[
    (austin_df[feature_two] > house_p_min) & (austin_df[feature_two] < house_p_max)
]

graph = st.selectbox("Select Graph", ("Distribution", "Scatterplot", "Boxplot"))

if graph == "Distribution":
    fig = plt.figure()
    plt.hist(austin_df[feature_one])
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
elif graph == "Scatterplot":
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(
        austin_df[feature_one],
        austin_df[feature_two],
    )

    ax.set_xlabel(feature_one)
    ax.set_ylabel(feature_two)

    st.write(fig)
elif graph == "Boxplot":
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.boxplot(ax=ax, y=austin_df[feature_one], x=austin_df[feature_two])

    ax.set_ylabel(feature_one)
    ax.set_xlabel(feature_two)
    st.write(fig)


# INPUTS BELOW

file_path = "./models/stacked_reg.pkl"
with open(file_path, "rb") as f:
    model = pickle.load(f)

features = [
    "num_of_photos",
    "living_area_sq_ft",
    "avg_school_rating",
    "num_of_bathrooms",
    "num_of_bedrooms",
    "num_of_stories",
    "age",
    "airport_distance",
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
    "city_manchaca",
    "city_manor",
    "city_pflugerville",
]


df = pd.DataFrame(
    np.zeros((1, 23)),
    columns=features,
    dtype="int64",
)

# User Inputs
city = st.sidebar.selectbox(
    "Select City",
    ("pflugerville", "del valle", "austin", "driftwood", "manor", "manchaca"),
)

df[f"city_{city}"] = 1

zipcode = st.sidebar.number_input(
    "Enter Zipcode", min_value=73301, max_value=78799, step=1
)

geolocator = Nominatim(user_agent="myGeocoder")
location = geolocator.geocode({"postalcode": zipcode})

df["airport_distance"] = downtown_distance(location.raw["lat"], location.raw["lon"])

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

df[f"home_type_{home_type}"] = 1

df["age"] = datetime.date.today().year - st.sidebar.number_input(
    "Enter Year Built", min_value=1800
)

df["num_of_photos"] = st.sidebar.number_input("Enter Number Of Appliances", min_value=1)

df["living_area_sq_ft"] = st.sidebar.number_input("Enter Living Area Sq Ft")

df["num_of_bedrooms"] = st.sidebar.slider(
    "Number of bedroom", max_value=10, min_value=1
)

df["num_of_bathrooms"] = st.sidebar.number_input(
    "Number Of Bathrooms", min_value=0.0, max_value=10.0, step=0.1
)

df["num_of_stories"] = st.sidebar.slider(
    "How many stories in your building", max_value=35, min_value=1
)

df["avg_school_rating"] = st.sidebar.slider(
    "Enter School Rating", max_value=10, min_value=1
)

prediction = round(np.exp(model.predict(df)[0]), 2)
# st.write(f"Your house guesstimated price: ${}")

st.markdown(
    """
<style>
.big-font {
    font-size:36px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="big-font">Your house guesstimated price: ${prediction}</p>',
    unsafe_allow_html=True,
)
