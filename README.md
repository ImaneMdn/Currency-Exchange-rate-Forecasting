# Currency Exchange Rate Forecasting and Trading Algorithm
This project focuses on forecasting currency exchange rates using **machine learning** and **deep learning algorithms**. It also includes a **trading algorithm** to simulate buying and selling based on the forecasted data. 

### Key Features:
- **Machine Learning & Deep Learning Models**: Various models, including **SVR**, **Random Forest**, **ANN**, and others, are used to predict currency exchange rates.
- **Forecasting Model**: The project includes a **time series forecasting model**, which predicts future currency exchange rates based on historical data.
- **Trading Algorithm**: A **trading algorithm** is implemented that simulates trades (**buy/sell/hold**) based on the predicted exchange rates, aiming to make profit over time.
- **Streamlit App**: A **Streamlit app** is used to provide an interactive interface for users to visualize the forecasting results and simulate trading strategies.

### Installation
To install the necessary libraries for this project, please use the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
### Yahoo Finance API Update:
Please note that the Yahoo Finance API has been updated recently. As a result, when fetching the data, it will no longer be in a simple 2D format. Instead, the fetched data is now in a multi-index format. Be aware of this change when working with the data.

### Running the Streamlit App
To run the Streamlit app, use the following command:
```bash
streamlit run <path-to-your-project>/app.py
```
Replace <path-to-your-project> with the path to your project's directory.
