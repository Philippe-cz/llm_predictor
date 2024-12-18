from dotenv import load_dotenv
from prompts import system_prompt
from openai import OpenAI
import base64
import openai
import os
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from prompts import (
    intro_prompt,
    few_shot_prompt,
    final_prediction_prompt,
    system_prompt,
)
from helpers_functions import StockWizzard
import json
import ast
from helpers_functions import merge_df


load_dotenv()

secret_key = os.getenv("openai_general_key")
secret_key_alpha = os.getenv("secret_key_alpha")

models = ["gpt-4o-mini", "gpt-4o-2024-05-13"]
ticker = "ON"
indicators = ""
timeframe = "weekly"
model = models[1]

train_window = 7
target_window = 1

few_shot_train_set = 12


main_path = os.getcwd()


stock_predictor = StockWizzard(
    ticker=ticker,
    indicators=indicators,
    timeframe=timeframe,
    model=model,
    train_window=train_window,
    target_window=target_window,
    secret_key=secret_key,
    secret_key_alpha=secret_key_alpha,
    few_shot_train_set=few_shot_train_set,
    client=OpenAI(api_key=secret_key),
)

stock_predictor.create_folder_if_not_exists(main_path, ticker)

print(f"selected model {model}")

## Prepare environment
stock_predictor.clean_folders(os.path.join(main_path, f"chart_images/{ticker}"))
## Pull data
stock_data = stock_predictor.get_weekly_stock_data(ticker, secret_key_alpha)

# Description
# stock data are splitted to windows, last rows (top rows most recent ones) are target used for few shot examples as windows
# head is used as formation of last data for prediction

target_folder = f"chart_images/{ticker}/train"
target_final_folder = f"chart_images/{ticker}/final_chart"

results_df = pd.DataFrame()

for i in range(few_shot_train_set):
    # Get the target window slice (top rows for each iteration)
    target_data = stock_data.iloc[i : i + target_window]

    if target_window > 1:
        target_data = merge_df(target_data)
    # Get the train window slice (preceding rows) window
    train_data = stock_data.iloc[i + target_window :]

    # Create an iteration-specific folder inside the target_folder
    iteration_folder = f"{target_folder}/iteration_{i+1}"
    iteration_final_folder = f"{target_final_folder}/iteration_{i+1}"

    if not os.path.exists(iteration_folder):
        os.makedirs(iteration_folder)
    if not os.path.exists(iteration_final_folder):
        os.makedirs(iteration_final_folder)

    # Pass the iteration-specific folder as the target_folder argument in test_pipeline
    messages, response = stock_predictor.test_pipeline(
        train_data, iteration_folder, iteration_final_folder
    )

    # Display or process the train and target data
    print(f"Iteration {i+1}:")
    print("Target Data:")
    print(target_data)
    print("Train Data:")
    print(train_data)
    print("-" * 50)
    output = ast.literal_eval(json.dumps(json.loads(response)))
    predicted_high = output["High"]
    predicted_low = output["Low"]
    # Add predicted values to the DataFrame
    df = target_data.copy()

    results_df = pd.concat(
        [
            results_df,
            stock_predictor.calculate_profit_and_deviation(
                df, predicted_high, predicted_low, train_data.head(train_window)
            ),
        ]
    )

    print("b")


results_df["Profitable_trades"] = stock_predictor.calculate_profitable_percentage(
    results_df
)

print("calculating...")


def calculate_stuffs(df):

    return df


results_df.to_excel(f"{main_path}/{ticker}_performance_report.xlsx")
print("a")
