import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

import mysql.connector as connection

st.title('📈 Energy Consumption Automated Time Series Forecasting')

"""
This App uses time series forecasting models like ARIMA, SARIMA and Prophet to automatically generate future forecast values from an imported dataset. You'll be able to import your data from MySQL DB, visualize trends and features, analyze forecast performance, and finally download the created forecast ⚡️⚡️


"""
st.image(
            "data:image/jpeg;base64,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",
            width=400, # Manually Adjust the width of the image as per requirement
        )

# enter your server IP address
HOST = "173.249.22.93" 
# database name
DATABASE = "giga_fp"
# user data
USER = "root"
# user password
PASSWORD = "Gig@Sql0802$Adm/f"
# connect to MySQL server

try:
    mydb = connection.connect(host=HOST, database = DATABASE,user=USER, passwd=PASSWORD,use_pure=True)
    query = "Select * from Electricity_cons;"
    df = pd.read_sql(query,mydb)
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))

"""
### Fetch Data from Source
"""


if df is not None:
    data = df
    data['ds'] = pd.to_datetime(data['period'],errors='coerce') 
    data['y'] = df['total-consumption-btu']
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)

"""
### Select Forecast Horizon
Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?(months)',
min_value = 1, max_value = 60)

makes = df['sectorDescription'].drop_duplicates()

# option = st.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone'))
make_choice = st.selectbox('Select sector:', makes)

st.write('You selected:', make_choice)

data = df[df['sectorDescription'] == make_choice] 

if df is not None:
    m = Prophet()
    m.fit(data)


"""
### Visualize Forecast Data
The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input, freq='MS')
    """
    ds: the datestamp of the forecasted value 

    yhat: the forecasted value of our metric (in Statistics, yhat is a notation traditionally used to represent the predicted values of a value y)
    
    yhat_lower: the lower bound of our forecasts
    
    yhat_upper: the upper bound of our forecasts
    linkcode
    
    """
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)


    # Short-term forecast (Next 2 months)
    future_short = m.make_future_dataframe(periods=60)
    forecast_short = m.predict(future_short)
    forecast_short =  forecast_short[forecast_short['ds'] > max_date] 

    # Long-term forecast (3 years)
    future_long = m.make_future_dataframe(periods=36,freq='MS') # 3 years
    forecast_long = m.predict(future_long)
    forecast_long =  forecast_long[forecast_long['ds'] > max_date]

    # Convert back to original scale
#     forecast_short["yhat"] = np.exp(forecast_short["yhat"])
#     forecast_long["yhat"] = np.exp(forecast_long["yhat"])


    # Streamlit dashboard
    st.write("# Short-term Forecast (Next 2 months)")
    st.line_chart(forecast_short[["ds", "yhat"]].set_index("ds"))

    st.write("# Long-term Forecast (3 years)")
    st.line_chart(forecast_long[["ds", "yhat"]].set_index("ds"))

    
    st.write("# Actual vs Predicted")
    
    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)


"""
### Download the Forecast Data
The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)

