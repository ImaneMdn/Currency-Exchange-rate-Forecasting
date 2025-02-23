import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import joblib
from keras.models import load_model
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Styling the side bar
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #525252;
    }
</style>
""", unsafe_allow_html=True)

# Loading the saved models
def load_models():
    svr_model = joblib.load('./Best_SVR_Model.pkl')
    rf_model = joblib.load('./Best_RF_Model.pkl')
    mlp_model = load_model('./Best_MLP_Model.h5')
    cnn_model = load_model('./Best_CNN_Model.h5')
    return {
        'SVR': svr_model,
        'RF': rf_model,
        'MLP': mlp_model,
        'CNN': cnn_model
    }
    
models = load_models()

# Fetching the dataset
def fetch_data(currency, start, end):
    try:
        # Fetching data from Yahoo Finance API
        data = yf.download(currency+'=X', start, end)
        if isinstance(data.columns, pd.MultiIndex):
           data.columns = data.columns.droplevel(1)
           data.columns.name = None  
           data = data.rename_axis("Date").reset_index().set_index("Date")
        if not data.empty:
            st.success(f"Data fetched successfully from Yahoo Finance API!")
            return data
        else:
            st.error(f"No data found for {currency}. Please check the input or try a different currency pair.")
            return None
    except Exception as e:
        st.error("Error fetching data.")
        st.error(e)
        return None
    
# Defining a function to predict the test set    
def model_predict(model):
    predictions = model.predict(x_test)
    return scaler.inverse_transform(predictions.reshape(-1, 1))  

# Creating a DataFrame for original and predicted values    
def create_dataframe(y_test_original, predictions, index):
    return pd.DataFrame({
        'Original': y_test_original.flatten(),
        'Predicted': predictions.flatten()
    }, index=index)
    
# Function to plot the Model
def plot_model(data, model_name, forecast):
    st.subheader(f'Currency Exchange Rate Forecasting using {model_name}')
    fig = px.line(
        data,
        x=data.index,
        y=['Original', 'Predicted'],
        labels={'value': 'Price', 'index': 'Date'},
        template='plotly_dark', 
        width=900,
        height=600
    ) 
    fig.add_scatter(
            x=forecast.index,
            y=forecast['Forecast'],
            mode='lines',
            name='60-Day Forecast',
            line=dict(color='yellow', dash='solid')
        )

    st.plotly_chart(fig)
    
    # Calculating metrics
    mse = mean_squared_error(data['Original'], data['Predicted'])
    mae = mean_absolute_error(data['Original'], data['Predicted'])
    r2 = r2_score(data['Original'], data['Predicted'])

    # Displaying metrics
    st.write(f"***{model_name} Metrics:***")
    st.write(f"***Mean Squared Error:*** {mse:.6f}")
    st.write(f"***Mean Absolute Error:*** {mae:.6f}")
    st.write(f"***R² Score:*** {r2:.4f}")
    
    st.write("Note: Support Vector Regressor achieves the highest accuracy compared to all other models across the evaluation metrics on the GBP/USD data.")


st.sidebar.header("Data Input")
st.header('Currency Exchange Rate Forecasting Dashboard')

# Default value is set to 'GBPUSD'
currency = st.sidebar.text_input('Select the currency exchange pair (e.g., GBPUSD):', 'GBPUSD')

start = datetime(2014, 1, 1)
end = datetime(2025, 1, 1)

# Downloading historical data for the selected currency exchange rate pair
GBPUSD_Data = fetch_data(currency, start, end)
st.write("Start date:", start)
st.write("End date:", end - timedelta(days=1))

# Dropdown for model selection
model_options = ['SVR', 'RF', 'MLP', 'CNN']
selected_model = st.sidebar.selectbox('Select Forecasting Model:', model_options)

# Displaying Historical Data Table
st.subheader(f"{currency} Exchange Rate Data")
st.dataframe(GBPUSD_Data, height=400, width=900)

# Visualising the Exchange rate data
st.subheader(f'{currency} Exchange Rate Overview')
fig = px.line(GBPUSD_Data, GBPUSD_Data.index, y=GBPUSD_Data.columns, title = currency, template = 'plotly_dark', width = 900, height = 600) 
st.plotly_chart(fig)

# Plotting the 50, 100 and 150 day the moving averages
GBPUSD_Data['50-Day MA'] = GBPUSD_Data['Close'].rolling(window=50).mean()
GBPUSD_Data['100-Day MA'] = GBPUSD_Data['Close'].rolling(window=100).mean()
GBPUSD_Data['150-Day MA'] = GBPUSD_Data['Close'].rolling(window=150).mean()

st.subheader(f"{currency} Closing Price with Moving Averages")
fig = px.line(GBPUSD_Data, 
              x=GBPUSD_Data.index, 
              y=['50-Day MA', '100-Day MA', '150-Day MA'], 
              template='plotly_dark', 
              width=900, 
              height=600)

st.plotly_chart(fig)

# Splitting the data into training, validation, and testing sets 
train_data, temp_data = train_test_split(GBPUSD_Data[['Close']], test_size=0.3, shuffle=False)
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

# Scaling the data
scaler = joblib.load('scaler.pkl')
scaled_test_data = scaler.transform(test_data)

# Creating sequences from the the test data
sequence_length = 60
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Generating sequences for the test data
x_test, y_test = create_sequences(scaled_test_data, sequence_length)

y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
index = test_data.index[sequence_length:]

# Calling the function for each model
model_data = {}
for model_name, model in models.items():
    test_predictions = model_predict(model)
    model_data[model_name] = create_dataframe(y_test_original, test_predictions, index)

    
st.sidebar.markdown("---")

st.sidebar.subheader("Forecasting Method")
st.sidebar.markdown("""
- **SVR**: Support Vector Regressor.
- **RF**: Random Forest Regressor.
- **MLP**: Multi-layer Neural Network.
- **CNN**: Convolutional Neural Network.
""")


def add_forecast(data, model):
    # Forecasting the next 60 days
    forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 61)]
    forecast = pd.DataFrame(index=forecast_dates, columns=['Forecast'])

    # Using the last 60 days of data for forecasting
    last_60_days = data[['Close']].tail(60) 
    last_60_days_scaled = scaler.transform(last_60_days) 

    for i in range(60):
        # Preparing input for the forecast
        x_forecast = last_60_days_scaled[-60:].reshape(1, -1)
        # Making the prediction using the selected model
        y_forecast = model.predict(x_forecast)
        # Inverse transform to get the original scale of the prediction
        y_forecast_inverse = scaler.inverse_transform(y_forecast.reshape(-1, 1))
        # Storing the forecasted value in the DataFrame
        forecast.iloc[i] = y_forecast_inverse[0][0]

        # Appending the forecast to the last_60_days_scaled to use in the next iteration
        last_60_days_scaled = np.append(last_60_days_scaled, y_forecast)

    return forecast

if selected_model in models:
    # Adding forecast to the data using the selected model
    forecast = add_forecast(GBPUSD_Data, models[selected_model])
    # Plotting both the original model output and forecast
    plot_model(model_data[selected_model], selected_model, forecast)
    