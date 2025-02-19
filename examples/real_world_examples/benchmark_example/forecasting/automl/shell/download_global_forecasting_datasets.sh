#!/bin/bash

mkdir -p ./data/global_forecasting

# Webpage URL format: https://zenodo.org/record/4659727

# Curated by Monash
echo australian_electricity_demand_dataset && curl https://zenodo.org/record/4659727/files/australian_electricity_demand_dataset.zip?download=1 --output ./data/global_forecasting/australian_electricity_demand_dataset.zip

# Curated by Monash
echo bitcoin_dataset_with_missing_values && curl https://zenodo.org/record/5121965/files/bitcoin_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/bitcoin_dataset_with_missing_values.zip
echo bitcoin_dataset_without_missing_values && curl https://zenodo.org/record/5122101/files/bitcoin_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/bitcoin_dataset_without_missing_values.zip

# From the book "Forecasting with exponential smoothing: the state space approach" by Hyndman, Koehler, Ord and Snyder (Springer, 2008): https://cran.r-project.org/web/packages/expsmooth/
echo car_parts_dataset_with_missing_values && curl https://zenodo.org/record/4656022/files/car_parts_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/car_parts_dataset_with_missing_values.zip
echo car_parts_dataset_without_missing_values && curl https://zenodo.org/record/4656021/files/car_parts_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/car_parts_dataset_without_missing_values.zip

# Computational Intelligence in Forecasting competition (2016): https://ieeexplore.ieee.org/document/8015455
echo cif_2016_dataset && curl https://zenodo.org/record/4656042/files/cif_2016_dataset.zip?download=1 --output ./data/global_forecasting/cif_2016_dataset.zip

# COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (): https://github.com/CSSEGISandData/COVID-19
echo covid_deaths_dataset && curl https://zenodo.org/record/4656009/files/covid_deaths_dataset.zip?download=1 --output ./data/global_forecasting/covid_deaths_dataset.zip
echo covid_mobility_dataset_without_missing_values && curl https://zenodo.org/record/4663809/files/covid_mobility_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/covid_mobility_dataset_without_missing_values.zip

# Dominick’s Finer Foods (1994): https://www.chicagobooth.edu/research/kilts/research-data/dominicks
echo dominick && curl https://zenodo.org/record/4654802/files/dominick_dataset.zip?download=1 --output ./data/global_forecasting/dominick_dataset.zip

# Half-hourly electricity demand For Victoria, Australia from R fpp2 package (2014): https://zenodo.org/record/4656069
echo elecdemand_dataset && curl https://zenodo.org/record/4656069/files/elecdemand_dataset.zip?download=1 --output ./data/global_forecasting/elecdemand_dataset.zip

# UCI: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
echo electricity_hourly_dataset && curl https://zenodo.org/record/4656140/files/electricity_hourly_dataset.zip?download=1 --output ./data/global_forecasting/electricity_hourly_dataset.zip
echo electricity_weekly_dataset && curl https://zenodo.org/record/4656141/files/electricity_weekly_dataset.zip?download=1 --output ./data/global_forecasting/electricity_weekly_dataset.zip

# FRED-MD (2016) by Michael W. McCracken & Serena Ng: https://www.tandfonline.com/doi/abs/10.1080/07350015.2015.1086655?journalCode=ubes20
echo fred_md_dataset && curl https://zenodo.org/record/4654833/files/fred_md_dataset.zip?download=1 --output ./data/global_forecasting/fred_md_dataset.zip

# From the book "Forecasting with exponential smoothing: the state space approach" by Hyndman, Koehler, Ord and Snyder (Springer, 2008): https://cran.r-project.org/web/packages/expsmooth/
echo hospital_dataset && curl https://zenodo.org/record/4656014/files/hospital_dataset.zip?download=1 --output ./data/global_forecasting/hospital_dataset.zip

echo kaggle_web_traffic_weekly_dataset && curl https://zenodo.org/record/4656664/files/kaggle_web_traffic_weekly_dataset.zip?download=1 --output ./data/global_forecasting/kaggle_web_traffic_weekly_dataset.zip
echo kaggle_web_traffic_dataset_with_missing_values && curl https://zenodo.org/record/4656080/files/kaggle_web_traffic_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/kaggle_web_traffic_dataset_with_missing_values.zip
echo kaggle_web_traffic_dataset_without_missing_values && curl https://zenodo.org/record/4656075/files/kaggle_web_traffic_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/kaggle_web_traffic_dataset_without_missing_values.zip

# KDD Cup 2018: https://www.kdd.org/kdd2018/kdd-cup
echo kdd_cup_2018_dataset_without_missing_values && curl https://zenodo.org/record/4656756/files/kdd_cup_2018_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/kdd_cup_2018_dataset_without_missing_values.zip

# Smart meters in London by JEAN-MICHEL D (2020): https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london
echo london_smart_meters_dataset_with_missing_values && curl https://zenodo.org/record/4656072/files/london_smart_meters_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/london_smart_meters_dataset_with_missing_values.zip
echo london_smart_meters_dataset_without_missing_values && curl https://zenodo.org/record/4656091/files/london_smart_meters_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/london_smart_meters_dataset_without_missing_values.zip

# M1 Competition: https://www.jstor.org/stable/2345077?origin=crossref
echo m1_monthly_dataset && curl https://zenodo.org/record/4656159/files/m1_monthly_dataset.zip?download=1 --output ./data/global_forecasting/m1_monthly_dataset.zip
echo m1_quarterly_dataset && curl https://zenodo.org/record/4656154/files/m1_quarterly_dataset.zip?download=1 --output ./data/global_forecasting/m1_quarterly_dataset.zip
echo m1_yearly_dataset && curl https://zenodo.org/record/4656193/files/m1_yearly_dataset.zip?download=1 --output ./data/global_forecasting/m1_yearly_dataset.zip

# M3 Competition: https://www.sciencedirect.com/science/article/abs/pii/S0169207000000571?via%3Dihub
echo m3_monthly_dataset && curl https://zenodo.org/record/4656298/files/m3_monthly_dataset.zip?download=1 --output ./data/global_forecasting/m3_monthly_dataset.zip
echo m3_other_dataset && curl https://zenodo.org/record/4656335/files/m3_other_dataset.zip?download=1 --output ./data/global_forecasting/m3_other_dataset.zip
echo m3_quarterly_dataset && curl https://zenodo.org/record/4656262/files/m3_quarterly_dataset.zip?download=1 --output ./data/global_forecasting/m3_quarterly_dataset.zip
echo m3_yearly_dataset && curl https://zenodo.org/record/4656222/files/m3_yearly_dataset.zip?download=1 --output ./data/global_forecasting/m3_yearly_dataset.zip

# M4 Competition: https://www.sciencedirect.com/science/article/pii/S0169207019301128?via%3Dihub
echo m4_daily_dataset && curl https://zenodo.org/record/4656548/files/m4_daily_dataset.zip?download=1 --output ./data/global_forecasting/m4_daily_dataset.zip
echo m4_hourly_dataset && curl https://zenodo.org/record/4656589/files/m4_hourly_dataset.zip?download=1 --output ./data/global_forecasting/m4_hourly_dataset.zip
echo m4_monthly_dataset && curl https://zenodo.org/record/4656480/files/m4_monthly_dataset.zip?download=1 --output ./data/global_forecasting/m4_monthly_dataset.zip
echo m4_quarterly_dataset && curl https://zenodo.org/record/4656410/files/m4_quarterly_dataset.zip?download=1 --output ./data/global_forecasting/m4_quarterly_dataset.zip
echo m4_weekly_dataset && curl https://zenodo.org/record/4656522/files/m4_weekly_dataset.zip?download=1 --output ./data/global_forecasting/m4_weekly_dataset.zip
echo m4_yearly_dataset && curl https://zenodo.org/record/4656379/files/m4_yearly_dataset.zip?download=1 --output ./data/global_forecasting/m4_yearly_dataset.zip

# NN5 competition 2012: https://www.sciencedirect.com/science/article/abs/pii/S0957417412000528?via%3Dihub and http://www.neural-forecasting-competition.com/NN5/
echo nn5_daily_dataset_with_missing_values && curl https://zenodo.org/record/4656110/files/nn5_daily_dataset_with_missing_values.zip?download=1.zip --output ./data/global_forecasting/nn5_daily_dataset_with_missing_values.zip
echo nn5_daily_dataset_without_missing_values && curl https://zenodo.org/record/4656117/files/nn5_daily_dataset_without_missing_values.zip?download=1.zip --output ./data/global_forecasting/nn5_daily_dataset_without_missing_values.zip
echo nn5_weekly_dataset && curl https://zenodo.org/record/4656125/files/nn5_weekly_dataset.zip?download=1 --output ./data/global_forecasting/nn5_weekly_dataset.zip

# OikoLab 2021: https://zenodo.org/record/5184708
echo oikolab_weather_dataset && curl https://zenodo.org/record/5184708/files/oikolab_weather_dataset.zip?download=1 --output ./data/global_forecasting/oikolab_weather_dataset.zip

# City of Melbourne, 2020
echo pedestrian_counts_dataset && curl https://zenodo.org/record/4656626/files/pedestrian_counts_dataset.zip?download=1 --output ./data/global_forecasting/pedestrian_counts_dataset.zip

# Curated by Monash
echo rideshare_dataset_with_missing_values && curl https://zenodo.org/record/5122114/files/rideshare_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/rideshare_dataset_with_missing_values.zip
echo rideshare_dataset_without_missing_values && curl https://zenodo.org/record/5122232/files/rideshare_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/rideshare_dataset_without_missing_values.zip

# McLeod and Gweon, 2013: http://www.jenvstat.org/v04/i11
echo saugeenday_dataset && curl https://zenodo.org/record/4656058/files/saugeenday_dataset.zip?download=1 --output ./data/global_forecasting/saugeenday_dataset.zip

# Curated by Monash
echo solar_4_seconds_dataset && curl https://zenodo.org/record/4656027/files/solar_4_seconds_dataset.zip?download=1 --output ./data/global_forecasting/solar_4_seconds_dataset.zip
echo solar_10_minutes_dataset && curl https://zenodo.org/record/4656144/files/solar_10_minutes_dataset.zip?download=1 --output ./data/global_forecasting/solar_10_minutes_dataset.zip
echo solar_weekly_dataset && curl https://zenodo.org/record/4656151/files/solar_weekly_dataset.zip?download=1 --output ./data/global_forecasting/solar_weekly_dataset.zip

# Sunspot, 2015: http://www.sidc.be/silso/newdataset
echo sunspot_dataset_with_missing_values && curl https://zenodo.org/record/4654773/files/sunspot_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/sunspot_dataset_with_missing_values.zip
echo sunspot_dataset_without_missing_values && curl https://zenodo.org/record/4654722/files/sunspot_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/sunspot_dataset_without_missing_values.zip

# Curated by Monash
echo temperature_rain_dataset_with_missing_values && curl https://zenodo.org/record/5129073/files/temperature_rain_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/temperature_rain_dataset_with_missing_values.zip
echo temperature_rain_dataset_without_missing_values && curl https://zenodo.org/record/5129091/files/temperature_rain_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/temperature_rain_dataset_without_missing_values.zip

# The tourism forecasting competition (2011): https://www.sciencedirect.com/science/article/abs/pii/S016920701000107X?via%3Dihub
echo tourism_monthly_dataset && curl https://zenodo.org/record/4656096/files/tourism_monthly_dataset.zip?download=1 --output ./data/global_forecasting/tourism_monthly_dataset.zip
echo tourism_quarterly_dataset && curl https://zenodo.org/record/4656093/files/tourism_quarterly_dataset.zip?download=1 --output ./data/global_forecasting/tourism_quarterly_dataset.zip
echo tourism_yearly_dataset && curl https://zenodo.org/record/4656103/files/tourism_yearly_dataset.zip?download=1 --output ./data/global_forecasting/tourism_yearly_dataset.zip

# Caltrans, 2020: https://pems.dot.ca.gov/
echo traffic_hourly_dataset && curl https://zenodo.org/record/4656132/files/traffic_hourly_dataset.zip?download=1 --output ./data/global_forecasting/traffic_hourly_dataset.zip
echo traffic_weekly_dataset && curl https://zenodo.org/record/4656135/files/traffic_weekly_dataset.zip?download=1 --output ./data/global_forecasting/traffic_weekly_dataset.zip

# Pruim et al., 2020: https://cran.r-project.org/web/packages/mosaicData/
echo us_births_dataset && curl https://zenodo.org/record/4656049/files/us_births_dataset.zip?download=1 --output ./data/global_forecasting/us_births_dataset.zip

# Uber 2015: https://github.com/fivethirtyeight/uber-tlc-foil-response
echo vehicle_trips_dataset_with_missing_values && curl https://zenodo.org/record/5122535/files/vehicle_trips_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/vehicle_trips_dataset_with_missing_values.zip
echo vehicle_trips_dataset_without_missing_values && curl https://zenodo.org/record/5122537/files/vehicle_trips_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/vehicle_trips_dataset_without_missing_values.zip

# Sparks et al., 2020: https://cran.r-project.org/web/packages/bomrang
echo weather_dataset && curl https://zenodo.org/record/4654822/files/weather_dataset.zip?download=1 --output ./data/global_forecasting/weather_dataset.zip

# Google, 2017: https://www.kaggle.com/c/web-traffic-time-series-forecasting
echo web_traffic_extended_dataset_with_missing_values && curl https://zenodo.org/record/7370977/files/web_traffic_extended_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/web_traffic_extended_dataset_with_missing_values.zip
echo web_traffic_extended_dataset_without_missing_values && curl https://zenodo.org/record/7371038/files/web_traffic_extended_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/web_traffic_extended_dataset_without_missing_values.zip

# Curated by Monash
echo wind_4_seconds_dataset && curl https://zenodo.org/record/4656032/files/wind_4_seconds_dataset.zip?download=1 --output ./data/global_forecasting/wind_4_seconds_dataset.zip
echo wind_farms_minutely_dataset_with_missing_values && curl https://zenodo.org/record/4654909/files/wind_farms_minutely_dataset_with_missing_values.zip?download=1 --output ./data/global_forecasting/wind_farms_minutely_dataset_with_missing_values.zip
echo wind_farms_minutely_dataset_without_missing_values && curl https://zenodo.org/record/4654858/files/wind_farms_minutely_dataset_without_missing_values.zip?download=1 --output ./data/global_forecasting/wind_farms_minutely_dataset_without_missing_values.zip
