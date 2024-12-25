from dotenv import load_dotenv
from prompts import system_prompt
from datetime import datetime
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
from helpers_functions import merge_df, combined_score, calculate_rsi
import mlflow
import logging
import shutil

# https://www.mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
load_dotenv()

secret_key = os.getenv("openai_general_key")
secret_key_alpha = os.getenv("secret_key_alpha")
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5"


def calculate_ema(df, column_name, ema_period):
    """
    Calculate the Exponential Moving Average (EMA) for a specified column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column for which to calculate the EMA.
        ema_period (int): The period over which the EMA is calculated.

    Returns:
        pd.DataFrame: The DataFrame with an additional column containing the EMA.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Calculate EMA using pandas' built-in function for simplicity
    ema_column_name = f"{column_name}_EMA_{ema_period}"
    df[ema_column_name] = df[column_name].ewm(span=ema_period, adjust=False).mean()

    return df


class LLMPredictions(mlflow.pyfunc.PythonModel):
    def load_context(self, context: dict) -> None:
        self.logger = logging.getLogger("mlflow")
        self.secret_key = os.getenv("openai_general_key")
        self.secret_key_alpha = os.getenv("secret_key_alpha")

    def predict(self, context, model_input):
        return self.process_request(model_input)

    def process_request(self, event: dict) -> str:
        event = {k: str(v) for k, v in event.items()}
        results_df = pd.DataFrame()
        ttl_tokens = []
        retrieved_fingerprints = []
        client = OpenAI(api_key=self.secret_key)
        system_fingerprint = event.get("system_fingerprint")
        fewshots_bool = event.get("fewshots_bool")

        stock_predictor = StockWizzard(
            ticker=event.get("ticker"),
            indicators=event.get("indicators"),
            timeframe=event.get("timeframe"),
            model=event.get("model"),
            train_window=int(event.get("train_window")),
            target_window=int(event.get("target_window")),
            secret_key=self.secret_key,
            secret_key_alpha=self.secret_key_alpha,
            few_shot_train_set=event.get("few_shot_train_set"),
            client=OpenAI(api_key=secret_key),
        )
        main_path = event.get("main_path")
        specific_few_shots = event.get("specific_few_shots")

        folder_ticker = os.path.join(main_path, ticker)
        stock_predictor.create_folder_if_not_exists(main_path, ticker)
        folder_chart_images = os.path.join(folder_ticker, "chart_images")
        stock_predictor.create_folder_if_not_exists(folder_ticker, "chart_images")

        stock_data = stock_predictor.get_weekly_stock_data(
            ticker, self.secret_key_alpha
        )

        if event.get("indicators")[:3] == "EMA":
            # Apply indicators...
            print("it")
            stock_data = calculate_ema(stock_data, column_name="Close", ema_period=13)
            stock_data = calculate_ema(stock_data, column_name="Close", ema_period=8)

        rsi_bool = False
        if rsi_bool:
            stock_data["RSI"] = calculate_rsi(stock_data["Close"], 14)

        target_folder = folder_chart_images

        target_final_folder = os.path.join(folder_chart_images, folder_chart_images)

        if event.get("predict") == "True":
            print("predicting...")
            folder_fewshots_predicted = os.path.join(
                folder_ticker, "few_shots_prediction"
            )
            if os.path.exists(folder_fewshots_predicted):
                shutil.rmtree(
                    folder_fewshots_predicted
                )  # Delete everything inside chart_images
                os.makedirs(folder_fewshots_predicted)

            stock_data_window = stock_data.head(int(event.get("train_window")))

            stock_data_window.to_excel(
                os.path.join(folder_ticker, f"{event.get('ticker')}_prediction.xlsx")
            )

            messages, response, tokens, retrieved_fingerprint = (
                stock_predictor.test_pipeline(
                    stock_data,
                    folder_fewshots_predicted,
                    folder_ticker,
                    export_windows=False,
                    export_prompt=True,
                    table=event.get("table"),
                    only_table=event.get("only_table"),
                    system_fingerprint=system_fingerprint,
                    specific_few_shots=specific_few_shots,
                    fewshots_bool=fewshots_bool,
                )
            )
            ttl_tokens.append(tokens)
            output = ast.literal_eval(
                json.dumps(
                    json.loads(response.replace("```json\n", "").replace("\n```", ""))
                )
            )
            if "High" in output:
                predicted_high = output["High"]
            if "Low" in output:
                predicted_low = output["Low"]

            file_name = os.path.join(
                folder_ticker, f"{event.get('ticker')}_predicted.json"
            )

            # Open a file and write the JSON data
            with open(file_name, "w") as json_file:
                json.dump(output, json_file, indent=4)

        else:
            # clean all data from chart images
            stock_predictor.clean_folders(os.path.join(main_path, f"chart_images"))
            if os.path.exists(folder_chart_images):
                shutil.rmtree(
                    folder_chart_images
                )  # Delete everything inside chart_images
                os.makedirs(folder_chart_images)
            for i in range(int(event.get("test_set"))):
                # Get the target window slice (top rows for each iteration)
                target_data = stock_data.iloc[i : i + target_window]

                if target_window > 1:
                    target_data = merge_df(target_data)
                # Get the train window slice (preceding rows) window
                train_data = stock_data.iloc[i + target_window :]

                # Create an iteration-specific folder inside the target_folder

                iteration_folder = os.path.join(folder_chart_images, f"iteration_{i+1}")
                final_train_it_pred_folder = os.path.join(
                    iteration_folder, "train_prediction"
                )

                if not os.path.exists(iteration_folder):
                    os.makedirs(iteration_folder)
                if not os.path.exists(final_train_it_pred_folder):
                    os.makedirs(final_train_it_pred_folder)

                try:
                    messages, response, tokens, retrieved_fingerprint = (
                        stock_predictor.test_pipeline(
                            train_data,
                            iteration_folder,
                            final_train_it_pred_folder,
                            table=event.get("table"),
                            only_table=event.get("only_table"),
                            system_fingerprint=system_fingerprint,
                            specific_few_shots=specific_few_shots,
                            fewshots_bool=fewshots_bool,
                        )
                    )
                    ttl_tokens.append(tokens)
                    retrieved_fingerprints.append(retrieved_fingerprint)

                    output = ast.literal_eval(
                        json.dumps(
                            json.loads(
                                response.replace("```json\n", "").replace("\n```", "")
                            )
                        )
                    )

                    predicted_high = None
                    predicted_low = None

                    if "High" in output:
                        predicted_high = float(
                            output["High"].replace("$", "").replace("...", "0")
                        )
                    if "Low" in output:
                        predicted_low = float(
                            output["Low"].replace("$", "").replace("...", "0")
                        )
                except:
                    print("break this - exception happened !!!!!!!!!!!!")
                    messages, response, tokens, retrieved_fingerprint = (
                        stock_predictor.test_pipeline(
                            train_data,
                            iteration_folder,
                            final_train_it_pred_folder,
                            table=event.get("table"),
                            only_table=event.get("only_table"),
                            system_fingerprint=system_fingerprint,
                            specific_few_shots=specific_few_shots,
                            fewshots_bool=fewshots_bool,
                        )
                    )

                    output = ast.literal_eval(
                        json.dumps(
                            json.loads(
                                response.replace("```json\n", "").replace("\n```", "")
                            )
                        )
                    )

                    predicted_high = None
                    predicted_low = None

                    if "High" in output:
                        predicted_high = float(
                            output["High"].replace("$", "").replace("...", "0")
                        )
                    if "Low" in output:
                        predicted_low = float(
                            output["Low"].replace("$", "").replace("...", "0")
                        )

                    pass

                if predicted_high == "0" or predicted_low == "0":
                    print("break it...")

                # create json file
                file_name = os.path.join(
                    iteration_folder, f"{event.get('ticker')}_predicted.json"
                )

                # Open a file and write the JSON data
                with open(file_name, "w") as json_file:
                    json.dump(output, json_file, indent=4)

                    df = target_data.copy()

                    results_df = pd.concat(
                        [
                            results_df,
                            stock_predictor.calculate_profit_and_deviation(
                                df,
                                predicted_high,
                                predicted_low,
                                train_data.head(train_window),
                            ),
                        ]
                    )

                results_df["Profitable_trades_(%)"] = (
                    stock_predictor.calculate_profitable_percentage(results_df)
                )

            results_df["Total_win_trades"] = (
                (results_df["profit"] > 0).astype(int).sum()
            )

            results_df["cumulative_profit"] = (
                results_df["profit_(10000)"].astype(float).cumsum()
            )
            df_2 = stock_predictor.calculate_metrics(results_df)
            df_2["model"] = event.get("model")
            df_2["total_tokens"] = ttl_tokens
            df_2["system_fingerprint"] = retrieved_fingerprints

            profit_ratio = df_2["Profitable_trades_(%)"].head(1).to_list()[0] / 100

            df_2["combined_score"] = df_2.apply(
                lambda x: combined_score(
                    profit_ratio, -x["High_Mean_Negative_Deviation"], beta=1.0
                ),
                axis=1,
            )
            # TODO: track trades logic must be implemented on each consecutive row as each row is new trade.
            df_2 = stock_predictor.track_trades(df_2)
            timestamp = stock_predictor.get_unique_timestamp()
            with open(f"{folder_ticker}/settings_{timestamp}.json", "w") as json_file:
                json.dump(event, json_file, indent=4)

            df_2.to_excel(
                os.path.join(
                    folder_ticker,
                    f"{ticker}_performance_report_{timestamp}_table_{event.get('table')}_onlytable_{event.get('only_table')}.xlsx",
                )
            )


train_windows = [4, 7, 10, 14, 17]

fewshots_bool = True

models = [
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-mini",
    "o1-preview",
]

ticker = "ON"
indicators = "EMA13, EMA8"  #
timeframe = "weekly"
model = models[0]
train_window = 4
target_window = 1
few_shot_train_set = 10
specific_few_shots = False
test_set = 50

predict = False

table = True
only_table = False

system_fingerprint = None


event = {
    "predict": predict,
    "ticker": ticker,
    "model": model,
    "system_fingerprint": system_fingerprint,
    "test_set": test_set,
    "indicators": indicators,
    "train_window": train_window,
    "target_window": target_window,
    "few_shot_train_set": few_shot_train_set,
    "specific_few_shots": specific_few_shots,
    "timeframe": timeframe,
    "main_path": r"C:\Users\z0040jeb\Desktop\predictions",
    "table": table,
    "only_table": only_table,
    "fewshots_bool": fewshots_bool,
}

print(f"\nUsing model {model} \n")
stock_predictor = LLMPredictions()
stock_predictor.load_context({})  # Manually call load_context with an empty context
context = ""
result = stock_predictor.predict(context, event)
print(f"\nProcess finished, using model {model} \n")
result
