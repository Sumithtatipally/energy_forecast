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
This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😍 

"""


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
    
    A variation in values from the output presented is to be expected as Prophet relies on Markov chain Monte Carlo (MCMC) methods to generate its forecasts.

    MCMC is a stochastic process, so values will be slightly different each time.
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

