import streamlit as st
import pandas as pd
import re # Regular expression library
import string
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Hotel Reservations.csv')
df = pd.DataFrame(data)
df = df.drop(columns=['Booking_ID'])

# Correcting the 'lower' lambda function
lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

# Applying the functions to the DataFrame
# Applying lowercase to all string columns in the DataFrame
for col in df.columns:
    if df[col].dtype == 'object':  # typically string columns are of 'object' type
        df[col] = df[col].map(lower)

encoded_df = df.copy()

encoded_df['type_of_meal_plan'] = encoded_df['type_of_meal_plan'].replace({'not selected': 0, 'meal plan 1': 1, 'meal plan 2': 2, 'meal plan 3': 3})
encoded_df['room_type_reserved'] = encoded_df['room_type_reserved'].replace({'room type 1': 0, 'room type 2': 1, 'room type 3': 2, 'room type 4': 3,
                                                             'room type 5': 4, 'room type 6': 5, 'room type 7': 6})
encoded_df['market_segment_type'] = encoded_df['market_segment_type'].replace({'offline': 0, 'online': 1, 'corporate': 2, 'aviation': 3, 'complementary': 4})

encoded_df['booking_status'] = encoded_df['booking_status'].replace({'canceled': 0, 'not canceled': 1})
X = encoded_df[['no_of_adults', 'no_of_children','no_of_weekend_nights','no_of_week_nights', 'type_of_meal_plan',
               'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
               'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
               'avg_price_per_room', 'no_of_special_requests']]

y = encoded_df['booking_status']

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define dictionaries for each category
meal_plan_mapping = {"Not Selected": 0, "Meal Plan 1": 1, "Meal Plan 2": 2, "Meal Plan 3": 3}
room_type_mapping = {"Room Type 1": 0, "Room Type 2": 1, "Room Type 3": 2, "Room Type 4": 3,
                     "Room Type 5": 4, "Room Type 6": 5, "Room Type 7": 6}
market_segment_mapping = {"Offline": 0, "Online": 1, "Corporate": 2, "Avitation": 3, "Complementary": 4}
st.title("Hotel Cancellation Prediction")
st.subheader("Booking Details")
no_of_adults = st.number_input(label="Number of Adults", step=1)
no_of_children = st.number_input(label="Number of Children", step=1)
no_of_weekend_nights = st.number_input(label="Number of Weekend nights", step=1)
no_of_week_nights = st.number_input(label="Number of Week Night", step=1)
lead_Time = st.number_input(label="Lead Time", min_value=1,max_value=300, step=1)
selected_meal_plan = st.selectbox("Select a meal plan:", list(meal_plan_mapping.keys()), index=0)
selected_room_type = st.selectbox("Select a room type:", list(room_type_mapping.keys()), index=0)
selected_market_segment = st.selectbox("Select a market segment:", list(market_segment_mapping.keys()), index=0)
parking_needed_value = int(st.checkbox("Parking Needed?"))

st.subheader("Arrival Information")
Arrival_Year = st.number_input(label="Arrival Year", step=1)
Arrival_Month = st.number_input(label="Arrival Month",min_value=1,max_value=12, step=1)
Arrival_Date = st.number_input(label="Arrival Date", min_value=1,max_value=31, step=1)

st.subheader("Booking History")
no_of_previous_cancellations = st.number_input(label="Number of Previous Cancellations", step=1)
no_of_previous_booking_not_canceled = st.number_input(label="Number of Previous Booking not Canceled", step=1)
repeated_guest_value = int(st.checkbox("Repeated Guest?"))

st.subheader("Special Requests")
no_of_special_request = st.number_input(label="Number of Special Requests", step=1)

st.subheader("Financial Information")
avg_price = st.number_input(label="Average Room Price",min_value=1,max_value=600, step=0.01)



# Get the values corresponding to the selected options
meal_plan_index = meal_plan_mapping[selected_meal_plan]
room_type_index = room_type_mapping[selected_room_type]
market_segment_index = market_segment_mapping[selected_market_segment]

if st.button("Click Me", type="primary"):
    data_test = [no_of_adults,no_of_children,no_of_weekend_nights,no_of_week_nights,meal_plan_index,parking_needed_value,room_type_index,lead_Time,Arrival_Year,Arrival_Month,Arrival_Date,market_segment_index,repeated_guest_value,no_of_previous_cancellations,no_of_previous_booking_not_canceled,avg_price,no_of_special_request]

    df_test = pd.DataFrame(data=[data_test])

    dtc = DecisionTreeClassifier(criterion = 'gini', max_depth= 11,random_state=42)

    # Train the model
    dtc.fit(X_train, y_train)

    # Take the model that was trained on the X_train_cv data and apply it to the X_test_cv
    prediction = dtc.predict(df_test)
    str_prediction = 'invalid data'
    if prediction == 0:
        str_prediction = 'Canceled'
    elif prediction == 1:
        str_prediction = 'Not Canceled'

    st.write(pd.DataFrame(df_test))
    st.write("Prediction result:", str_prediction)
