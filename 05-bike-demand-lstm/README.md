# Bike Demand LSTM

## Goal
Learn temporal patterns in bike-rental demand with an LSTM.

## Dataset
**Official source:** UCI Bike Sharing, ID 275.

Official page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Exact archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

UCI provides hourly and daily rental counts with weather and seasonal variables. citeturn0search1

## Pipeline

Retrieve official data → order observations chronologically → scale the target → construct sliding windows → train LSTM → evaluate future portion → export model.

## Why sequence windows?

An LSTM needs ordered observations. A 24-step window gives the network a fixed history from which to predict the next value.

## Important limitation

This baseline intentionally focuses on temporal target behaviour. A production forecasting model should incorporate calendar/weather covariates and use strictly time-aware validation.

## Run

`pip install -r requirements.txt`

`python train.py`

## Next level

Add weather/season features, walk-forward validation, MAE/RMSE, baseline comparisons and forecast plots.