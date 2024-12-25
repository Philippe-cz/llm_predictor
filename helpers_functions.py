import base64
from PIL import Image
import io
import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
import json  # Ensure you have imported json if you're saving labels as JSON

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from prompts import (
    intro_prompt,
    few_shot_prompt,
    few_shot_prompt_table,
    few_shot_prompt_table_only,
    final_prediction_prompt,
    final_prediction_prompt_only_table,
    system_prompt,
)
import mplfinance as mpf
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import mlflow
import logging
import shutil
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

secret_key = os.getenv("openai_general_key")
secret_key_alpha = os.getenv("secret_key_alpha")


def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
        data (pd.Series): A pandas Series of stock prices (typically the closing prices).
        period (int): The look-back period for calculating RSI (default is 14).

    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    # Calculate the differences between consecutive prices
    delta = data.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate the exponential moving average (EMA) of gains and losses
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def combined_score(WTR: float, MND: float, beta: float = 1.0) -> float:
    """
    Calculate a combined metric that balances winning trades ratio (WTR)
    against mean negative deviation (MND).

    The formula used is:
        Combined Score = WTR / (1 + beta * MND)

    Parameters
    ----------
    WTR : float
        Winning Trades Ratio (0 <= WTR <= 1)
    MND : float
        Mean Negative Deviation. This should be a non-negative number
        representing how much predictions undershoot actual prices on average.
    beta : float, optional
        A penalty factor for how severely MND penalizes the score.
        Default is 1.0.

    Returns
    -------
    float
        The combined score. Higher is better. A value of 0 means no winning trades.
    """
    # Validate inputs
    if WTR < 0 or WTR > 1:
        raise ValueError("WTR should be between 0 and 1 (inclusive).")
    if MND < 0:
        raise ValueError("MND should be a non-negative number.")
    if beta < 0:
        raise ValueError("beta should be a non-negative number.")

    # Calculate the combined score
    score = WTR / (1 + beta * MND)
    return score


def merge_df(df):
    merged_row = pd.DataFrame(
        {
            "Date": [f"{df['Date'].iloc[0]} - {df['Date'].iloc[-1]}"],
            "Open": [df["Open"].iloc[0]],
            "High": [df["High"].max()],
            "Low": [df["Low"].min()],
            "Close": [df["Close"].iloc[-1]],
            "Volume": [df["Volume"].sum()],
        }
    )
    return merged_row


class StockWizzard:
    def __init__(
        self,
        ticker,
        indicators,
        timeframe,
        model,
        train_window,
        target_window,
        secret_key,
        secret_key_alpha,
        few_shot_train_set,
        client,
    ):
        # Constructor method to initialize the attributes
        self.ticker = ticker
        self.indicators = indicators
        self.timeframe = timeframe
        self.model = model
        self.train_window = train_window
        self.target_window = target_window
        self.secret_key = secret_key
        self.secret_key_alpha = secret_key_alpha
        self.few_shot_train_set = few_shot_train_set
        self.client = client

    def test_pipeline(
        self,
        stock_data,
        target_folder,
        target_final_folder,
        export_windows=True,
        export_prompt=True,
        table=False,
        only_table=False,
        system_fingerprint=None,
        specific_few_shots="False",
        fewshots_bool="False",
    ):
        """
        function will take arguments and build train set for few shots and make prediction

        stock_data : dataframe including entire stock data set , up until point of target window '
        target_final_date : last date of target window
        train_window : train window
        target_window : target window
        """
        print("target window : ", self.target_window)
        windowed_dataset = self.create_windows_with_labels(stock_data)
        if export_windows:
            labeled_windows = []
            # Iterate over each DataFrame in 'window' column and assign a window identifier
            for idx, window_df in enumerate(windowed_dataset["window"]):
                window_df = (
                    window_df.copy()
                )  # Make a copy to avoid modifying the original
                window_df["window_id"] = f"window_{idx}"  # Add an identifier column
                labeled_windows.append(window_df)

            # Concatenate all labeled DataFrames into a single DataFrame
            combined_dataframe = pd.concat(labeled_windows, ignore_index=True)
            combined_dataframe.to_excel(
                os.path.join(target_folder, f"{self.ticker}_train_windows.xlsx")
            )

        if self.target_window > 1:
            windowed_dataset["label"] = windowed_dataset["label"].apply(merge_df)

        target_final_date = stock_data["Date"].to_list()[0]

        ### TODO:To be tested
        if specific_few_shots == "True":
            current_date = pd.Timestamp(target_final_date)
            # Calculate 12 months back
            target_final_date = current_date - pd.DateOffset(months=12)
            # Adjust to the nearest Monday if not already a Monday
            if target_final_date.weekday() != 0:  # 0 means Monday
                target_final_date = target_final_date - pd.DateOffset(
                    days=target_final_date.weekday()
                )

        # prepare image charts for few shots
        self.build_dynamic_few_shots(
            windowed_dataset, target_final_date, target_folder, train=True
        )
        # load saved images and create mapping dictionaries
        # folder = "chart_images/train"  # Replace with your target folder
        images_data_dict, mapping_dict, csv_data_dict = (
            self.build_image_and_label_mappings(target_folder)
        )

        general_dict = {
            "TICKER": self.ticker,
            "INDICATORS": self.indicators,
            "TIMEFRAME": self.timeframe,
        }

        if fewshots_bool == "False":
            resulting_few_shots = None
        else:
            resulting_few_shots = self.construct_few_shots(
                few_shot_prompt,
                images_data_dict,
                mapping_dict,
                general_dict,
                csv_data_dict,
                table=table,
                only_table=only_table,
            )

        print("\n\nFINAL FORMATION....\n")
        stock_data_adjusted = stock_data.head(self.train_window)
        final_chart_dict = self.build_final_predictive_window_dict(
            stock_data_adjusted, target_final_folder
        )

        messages = self.build_messages(
            final_chart_dict,
            final_prediction_prompt,
            system_prompt,
            intro_prompt,
            resulting_few_shots,
            general_dict,
            only_table,
        )
        o_models = [
            "o1-mini-2024-09-12",
            "o1-preview-2024-09-12",
            "o1-mini",
            "o1-preview",
        ]

        if self.model in o_models:
            messages = [msg for msg in messages if msg.get("role") != "system"]

        if export_prompt:
            file_path = os.path.join(target_folder, "final_prompt.txt")

            # Open the file in write mode and write each item in the list to a new line
            with open(file_path, "w") as file:
                for line in messages:
                    file.write(
                        str(line) + "\n"
                    )  # Add a newline after each line of text

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0 if not self.model in o_models else 1,
            seed=123456,
            # system_fingerprint=system_fingerprint,
        )

        # print(response.choices[0].message.content)
        return (
            messages,
            response.choices[0].message.content,
            response.usage.total_tokens,
            response.system_fingerprint,
        )

    # Load the image and convert to base64
    def image_to_base64(self, image_path: str) -> str:
        # Open the image using PIL
        with Image.open(image_path) as img:
            # Create an in-memory byte stream for the image
            buffered = io.BytesIO()
            # Save the image to the stream in its original format
            img.save(buffered, format=img.format)
            # Get the byte data from the stream and encode it to base64
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64

    def get_b64_image(self, image_path: str) -> str:
        # Open the image file and encode it as a base64 string
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

        return images_b64

    def get_b64_images(self, images_path: str) -> dict:
        images_b64 = {}
        # Open the image file and encode it as a base64 string
        for image in os.listdir(images_path):
            images_b64[image] = image_to_base64(os.path.join(images_path, image))
        return images_b64

    def construct_few_shots(
        self,
        few_shot_prompt,
        images_data_dict,
        mapping_dict,
        general_dict,
        csv_data_dict,
        table=False,
        only_table=False,
    ):
        few_shots = []  # Initialize an empty list for few shots

        # Iterate over the images_data_dict
        for counter, (image_name, image_base64) in enumerate(images_data_dict.items()):
            high_label = mapping_dict[image_name][0]
            low_label = mapping_dict[image_name][1]

            if table == "True" and only_table == "False":

                # Format the few_shot_prompt with specific labels
                formatted_prompt = few_shot_prompt_table.format(
                    COUNTER=counter,
                    TICKER=general_dict["TICKER"],
                    TIMEFRAME=general_dict["TIMEFRAME"],
                    INDICATORS=general_dict["INDICATORS"],
                    high_label=high_label,
                    low_label=low_label,
                    stock_data=csv_data_dict[f"{counter}"].to_string(),
                )
            elif table == "False" and only_table == "False":
                # Format the few_shot_prompt with specific labels
                formatted_prompt = few_shot_prompt.format(
                    COUNTER=counter,
                    TICKER=general_dict["TICKER"],
                    TIMEFRAME=general_dict["TIMEFRAME"],
                    INDICATORS=general_dict["INDICATORS"],
                    high_label=high_label,
                    low_label=low_label,
                )

            elif only_table == "True":
                # Format the few_shot_prompt with specific labels
                formatted_prompt = few_shot_prompt_table_only.format(
                    COUNTER=counter,
                    TICKER=general_dict["TICKER"],
                    TIMEFRAME=general_dict["TIMEFRAME"],
                    INDICATORS=general_dict["INDICATORS"],
                    high_label=high_label,
                    low_label=low_label,
                    stock_data=csv_data_dict[f"{counter}"].to_string(),
                )

            if only_table == "False":
                # Constructing the message template
                template = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            else:
                # Constructing the message template WITHOUT IMAGE
                template = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt},
                    ],
                }

            # Append the template to the few_shots list
            few_shots.append(template)

        return few_shots

    def build_messages(
        self,
        final_chart_dict,
        final_prediction_prompt,
        system_prompt,
        intro_prompt,
        resulting_few_shots,
        general_dict,
        only_table,
    ):
        """
        Build a list of messages to send to the LLM API based on the system prompt, intro prompt, and few-shot examples.

        Parameters:
        - final_chart_dict(dict):base 64 image of final chart
        - final_prediction_prompt (str): main prompt at the final
        - system_prompt (str): The system-level instruction for the model.
        - intro_prompt (str): The introductory prompt to provide context.
        - resulting_few_shots (list): A list of few-shot examples with image analysis.
        - TICKER (str): The stock ticker symbol.
        - INDICATORS (str): Indicators related to the analysis.
        - TIMEFRAME (str): The timeframe for the analysis (e.g., weekly, daily).

        Returns:
        - list: A list of messages to be used in the API call.
        """

        # Initialize the message list with a system message if provided
        messages = []
        initial_intro = {
            "role": "user",
            "content": f"Analyze the {general_dict['TICKER']} stock with {general_dict['INDICATORS']} over the {general_dict['TIMEFRAME']} timeframe.",
        }
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Add a default system prompt if none is provided
            messages.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant providing technical analysis on stock charts.",
                }
            )

        # Add the introductory user message
        if intro_prompt:
            intro_message = intro_prompt.format(
                TICKER=general_dict["TICKER"],
                INDICATORS=general_dict["INDICATORS"],
                TIMEFRAME=general_dict["TIMEFRAME"],
            )
            messages.append({"role": "user", "content": intro_message})
        else:
            # Default intro message if none provided
            messages.append(initial_intro)

        # Add the few-shot examples to the messages
        if resulting_few_shots and isinstance(resulting_few_shots, list):
            messages.extend(resulting_few_shots)
        else:
            # Log a warning if few_shots is empty or malformed
            print("Warning: No valid few-shot examples provided.")

        if only_table == "False":
            abc = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": final_prediction_prompt.format(
                                TICKER=general_dict["TICKER"],
                                TIMEFRAME=general_dict["INDICATORS"],
                                INDICATORS=general_dict["TIMEFRAME"],
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{final_chart_dict['final_chart']}"
                            },
                        },
                    ],
                }
            ]

        else:
            abc = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": final_prediction_prompt_only_table.format(
                                TICKER=general_dict["TICKER"],
                                TIMEFRAME=general_dict["INDICATORS"],
                                INDICATORS=general_dict["TIMEFRAME"],
                            ),
                        },
                    ],
                }
            ]
        messages.extend(abc)
        return messages

    def process_windows(
        self,
        result_df,
        from_date,
        images_set,
        save_charts=False,
        output_folder="chart_images",
        train=True,
    ):
        # Ensure 'from_date' is a datetime object
        if isinstance(from_date, str):
            from_date = pd.to_datetime(from_date)

        # Find the index where the 'window' contains 'from_date'
        start_index = None
        for idx, window in result_df["window"].items():
            if from_date in pd.to_datetime(window["Date"]).values:
                start_index = int(idx)
                break

        if start_index is None:
            print(f"from_date {from_date.date()} not found in any window.")
            return

        # Create the output directory if saving charts
        if save_charts:
            if train:
                # output_folder = os.path.join(output_folder, "train")
                os.makedirs(output_folder, exist_ok=True)
            else:
                # output_folder = os.path.join(output_folder, "test")
                os.makedirs(output_folder, exist_ok=True)

        # Process the windows starting from 'start_index'
        for counter, i in enumerate(range(start_index, start_index + int(images_set))):
            if i >= len(result_df):
                print("Reached the end of the result_df.")
                break

            window = result_df.iloc[i]["window"]
            label = result_df.iloc[i]["label"]

            # Plot candlestick chart and get the figure object
            fig = self.plot_candlestick(window)

            # Save or show the plot
            if save_charts:
                # Construct the file paths
                image_filename = os.path.join(output_folder, f"chart_{counter}.png")
                label_filename = os.path.join(
                    output_folder, f"chart_{counter}_label.json"
                )

                window_json_path = os.path.join(
                    output_folder, f"formation_{counter}_window.csv"
                )
                window.to_csv(window_json_path)

                # Save the figure
                fig.savefig(image_filename, dpi=400)
                plt.close(fig)  # Close the figure to free memory

                # Save the label DataFrame as a JSON file
                label_dict = label.to_dict(orient="records")
                with open(label_filename, "w") as f:
                    json.dump(label_dict, f, default=str, indent=4)

                print(f"Saved chart and label for window {counter} to {output_folder}")
            else:
                plt.show()

            # Print window
            print(f"Window for window {i}:\n{window}\n")

            # Print labels
            print(f"Labels for window {i}:\n{label}\n")

    def plot_candlestick(self, window_df):
        # Prepare data
        window_df = window_df.copy()
        window_df["Date"] = pd.to_datetime(
            window_df["Date"]
        )  # Ensure Date column is datetime
        window_df.set_index("Date", inplace=True)  # Set Date as the index
        volume_flag = True if not "RSI" in window_df.columns else False
        # Create a candlestick chart using mplfinance
        fig, axlist = mpf.plot(
            window_df,
            type="candle",
            style="charles",
            title="Candlestick Chart",
            ylabel="Price ($)",
            volume=volume_flag,
            show_nontrading=True,
            returnfig=True,  # Return the figure object
            volume_alpha=1,
        )

        # Get the main candlestick and volume axes
        ax_candlestick = axlist[0]  # Main chart
        if volume_flag:
            ax_volume = axlist[2]  # Volume chart
            ax_volume.set_facecolor("black")

        # Plot moving averages if they exist in the DataFrame
        ma_columns = [col for col in window_df.columns if "EMA" in col or "SMA" in col]
        for ma_col in ma_columns:
            ax_candlestick.plot(
                window_df.index, window_df[ma_col], label=ma_col, linewidth=1.5
            )

        # Add legend for moving averages
        if ma_columns:
            ax_candlestick.legend(
                loc="upper center",
                bbox_to_anchor=(0.9, 1.15),  # Adjust coordinates for desired position
                ncol=1,  # Number of columns in the legend
                frameon=False,  # Optional: Remove the legend box border
            )

        # Set Y-axis ticks to 1 dollar increments on the candlestick chart
        ax_candlestick.yaxis.set_major_locator(MultipleLocator(1))
        ax_candlestick.yaxis.set_minor_locator(AutoMinorLocator(2))

        # Customize the x-axis to display Mondays correctly
        ax_candlestick.xaxis.set_major_locator(
            mdates.WeekdayLocator(byweekday=mdates.MONDAY)
        )
        ax_candlestick.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

        # Rotate the date labels for better readability
        plt.setp(ax_candlestick.get_xticklabels(), rotation=45, ha="right")

        # Plot RSI if "RSI" column exists
        if "RSI" in window_df.columns:
            # Create a new subplot for RSI
            ax_rsi = fig.add_axes([0.1, 0.1, 0.8, 0.2])  # Adjust position as needed
            ax_rsi.plot(
                window_df.index,
                window_df["RSI"],
                label="RSI",
                color="blue",
                linewidth=1.5,
            )
            ax_rsi.axhline(
                70, color="red", linestyle="--", linewidth=0.8, label="Overbought (70)"
            )
            ax_rsi.axhline(
                30, color="green", linestyle="--", linewidth=0.8, label="Oversold (30)"
            )
            ax_rsi.set_title("RSI Chart")
            ax_rsi.set_ylabel("RSI")
            ax_rsi.set_xlabel("Date")
            ax_rsi.legend(loc="upper left")

            # Customize the x-axis to display dates correctly
            ax_rsi.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.MONDAY)
            )
            ax_rsi.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

            # Rotate the date labels for better readability
            plt.setp(ax_rsi.get_xticklabels(), rotation=45, ha="right")

        return fig

    def create_windows_with_labels(self, df):
        # Ensure df is sorted in ascending order by date
        df = df.sort_values("Date").reset_index(drop=True)

        windows = []
        labels = []

        # Adjust the loop to correctly index windows and labels
        for i in range(len(df) - self.train_window - self.target_window + 1):
            window = df.iloc[i : i + self.train_window]
            label = df.iloc[
                i + self.train_window : i + self.train_window + self.target_window
            ]

            windows.append(window)
            labels.append(label)

        result_df = pd.DataFrame(
            {
                "window": windows,
                "label_number": list(range(len(windows))),
                "label": labels,
            }
        )

        return result_df.sort_values(by="label_number", ascending=False).reset_index(
            drop=True
        )

    def save_candlestick_chart(
        self, data, folder, save_chart=True, filename="chart.png"
    ):
        # Generate the candlestick chart using your existing function
        fig = self.plot_candlestick(data)

        if save_chart:
            # Create the directory if it doesn't exist
            os.makedirs(folder, exist_ok=True)

            # Construct the full file path
            file_path = os.path.join(folder, filename)

            # fig.set_size_inches(14, 10)

            # Save the figure
            fig.savefig(file_path, dpi=400)
            plt.close(fig)  # Close the figure to free memory

            print(f"Chart saved to {file_path}")
        else:
            # Display the chart
            plt.show()

    def load_image_and_convert_to_base64(self, image_path):
        final_chart_dict = {}

        # Read the image file in binary mode
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

            # Encode the image data in base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Store in the dictionary
            final_chart_dict["final_chart"] = image_base64

        return final_chart_dict

    def build_image_and_label_mappings(self, folder):
        images_data_dict = {}
        mapping_dict = {}

        # List all files in the target folder
        files = os.listdir(folder)
        csv_data_dict = dict()
        for filename in files:
            file_path = os.path.join(folder, filename)

            # Process image files
            if filename.endswith(".png"):
                # Get the base filename without extension
                base_filename = os.path.splitext(filename)[
                    0
                ]  # e.g., 'chart_0.png' -> 'chart_0'

                # Read and encode the image file in base64
                with open(file_path, "rb") as image_file:
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")

                    # Store in images_data_dict
                    images_data_dict[base_filename] = image_base64

            # Process label JSON files
            elif filename.endswith(".json"):
                # Get the base filename and remove '_label' suffix
                label_filename = os.path.splitext(filename)[
                    0
                ]  # e.g., 'chart_0_label.json' -> 'chart_0_label'
                if label_filename.endswith("_label"):
                    base_filename = label_filename[:-6]  # Remove '_label' suffix

                    # Read the JSON file
                    with open(file_path, "r") as json_file:
                        label_data = json.load(json_file)

                        # Ensure the JSON contains data
                        if label_data and isinstance(label_data, list):
                            first_item = label_data[
                                0
                            ]  # Assuming the first item contains the required data
                            high_value = first_item.get("High")
                            low_value = first_item.get("Low")

                            # Store in mapping_dict
                            mapping_dict[base_filename] = [high_value, low_value]

            # Process CSV files
            elif filename.endswith(".csv"):
                # Check if the filename follows the "formation_X_window.csv" pattern

                if "formation_" in filename and "_window" in filename:
                    # Extract the numeric part from the filename
                    base_filename = filename.split("_")[
                        1
                    ]  # e.g., 'formation_1_window.csv' -> '1'

                    # Read the CSV into a DataFrame
                    df = pd.read_csv(file_path).reset_index(drop=True)[
                        ["Date", "Open", "High", "Low", "Close", "Volume"]
                    ]
                    # Store the DataFrame in the csv_data_dict
                    csv_data_dict[base_filename] = df

        return images_data_dict, mapping_dict, csv_data_dict

    def get_weekly_stock_data(self, symbol, api_key):
        """
        Retrieves weekly adjusted stock data for the given symbol from Alpha Vantage,
        adjusts the dates to the previous Mondays, and returns a cleaned DataFrame.

        Parameters:
            symbol (str): The stock symbol to retrieve data for.
            api_key (str): Your Alpha Vantage API key.

        Returns:
            pandas.DataFrame: A DataFrame containing the stock data with dates adjusted to Mondays.
        """

        # Initialize TimeSeries with your API key
        ts = TimeSeries(key=api_key, output_format="pandas")

        # Get weekly adjusted data
        data, meta_data = ts.get_weekly_adjusted(symbol=symbol)

        # Rename columns
        columns_rename = {
            "date": "Date",
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "6. volume": "Volume",
        }
        stock_data = data.reset_index().rename(columns=columns_rename)[
            ["Date", "Open", "High", "Low", "Close", "Volume"]
        ]

        # Ensure 'Date' column is in datetime format
        stock_data["Date"] = pd.to_datetime(stock_data["Date"])

        # Adjust 'Date' to the previous Monday
        stock_data["Date"] = stock_data["Date"] - pd.to_timedelta(
            stock_data["Date"].dt.weekday, unit="D"
        )

        return stock_data

    def build_dataframe_from_folder(self, folder):
        """
        Builds a DataFrame from chart images and corresponding label JSON files in the specified folder.
        Adds 'High' and 'Low' columns extracted from the label JSON files.

        Parameters:
            folder (str): The path to the folder containing the chart images and label JSON files.

        Returns:
            pandas.DataFrame: A DataFrame with columns 'image_path', 'label_data', 'High', and 'Low'.
        """
        data = []

        # List all files in the folder
        files = os.listdir(folder)

        # Create a set of base filenames (without extension and without '_label' suffix)
        base_filenames = set()
        for filename in files:
            if filename.endswith(".png"):
                base_filename = os.path.splitext(filename)[0]  # Remove '.png'
                base_filenames.add(base_filename)

        # Loop over each base filename to find corresponding image and label files
        for base_filename in base_filenames:
            image_filename = base_filename + ".png"
            label_filename = base_filename + "_label.json"

            image_path = os.path.join(folder, image_filename)
            label_path = os.path.join(folder, label_filename)

            # Check if both the image and label files exist
            if os.path.exists(image_path) and os.path.exists(label_path):
                # Load the label data from the JSON file
                with open(label_path, "r") as f:
                    label_data = json.load(f)

                # Ensure the JSON contains data
                if label_data and isinstance(label_data, list):
                    first_item = label_data[
                        0
                    ]  # Assuming the first item contains the required data
                    high_value = first_item.get("High")
                    low_value = first_item.get("Low")
                    open_value = first_item.get("Open")

                    # Append the data to the list
                    data.append(
                        {
                            "image_path": image_path,
                            "label_data": label_data,  # Unpacked label data as dictionary
                            "Open": open_value,
                            "High": high_value,
                            "Low": low_value,
                        }
                    )
                else:
                    print(
                        f"Warning: Label data is empty or not a list in '{label_filename}'"
                    )
            else:
                print(f"Warning: Missing image or label file for '{base_filename}'")

        # Create a DataFrame from the data list
        df = pd.DataFrame(data)
        return df

    def clean_folders(self, main_folder):
        """
        Removes all files from the 'test' and 'train' subfolders within the specified main folder.

        Parameters:
            main_folder (str): The path to the main folder containing 'test' and 'train' subfolders.
        """
        # Paths to 'train' and 'test' subfolders
        train_folder = os.path.join(main_folder, "train")
        test_folder = os.path.join(main_folder, "test")
        final_chart_folder = os.path.join(main_folder, "final_chart")

        # List of subfolders to clean
        subfolders = [train_folder, test_folder, final_chart_folder]

        # Iterate over each subfolder
        for subfolder in subfolders:
            # Check if the subfolder exists
            if os.path.exists(subfolder):
                # List all files in the subfolder
                files = os.listdir(subfolder)
                # Iterate over each file
                for filename in files:
                    file_path = os.path.join(subfolder, filename)
                    # Check if it's a file (not a directory)
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Remove the file
                    elif os.path.isdir(file_path):
                        # If it's a directory, remove it and all its contents
                        import shutil

                        shutil.rmtree(file_path)
                print(f"All files removed from '{subfolder}'")
            else:
                print(f"Subfolder '{subfolder}' does not exist.")

    def add_weeks_to_date(self, date_str: str, weeks: int) -> tuple[str, int]:
        # Parse the input date in format "yyyy-mm-dd"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # Add the specified number of weeks (1 week = 7 days)
        final_date = date_obj + timedelta(weeks=weeks)

        # Adjust to the Monday of the week (0 = Monday, 6 = Sunday)
        final_monday = final_date - timedelta(days=final_date.weekday())

        # Return the final Monday date in "yyyy-mm-dd" format and the number of weeks added
        return final_monday.strftime("%Y-%m-%d"), weeks

    def subtract_weeks_from_date(self, date_str, weeks):
        # If the input is a Timestamp, convert it to a string in the format 'YYYY-MM-DD'
        if isinstance(date_str, pd.Timestamp):
            date_str = date_str.strftime("%Y-%m-%d")

        # Parse the input date in format "yyyy-mm-dd"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # Subtract the specified number of weeks (1 week = 7 days)
        final_date = date_obj - timedelta(weeks=weeks)

        # Return the final date as a string and weeks as is
        return final_date.strftime("%Y-%m-%d"), weeks

    def build_final_predictive_window_dict(self, stock_data, target_final_folder):
        main_path = os.getcwd()
        self.save_candlestick_chart(
            stock_data.head(self.train_window),
            target_final_folder,
            save_chart=True,
            filename="final_chart.png",
        )
        print(stock_data.head(self.train_window))
        final_image_path = f"{target_final_folder}/final_chart.png"

        final_chart_dict = self.load_image_and_convert_to_base64(final_image_path)

        return final_chart_dict

    def build_dynamic_few_shots(self, result, current_date, output_folder, train=True):
        """current date : it is a date which is target date that is to be predicted
        train_window : training window
        target_size : target window
        """
        ttl_window = self.train_window + self.target_window
        first_window_start_date = self.subtract_weeks_from_date(
            current_date, ttl_window - 1
        )[0]
        save_charts = True  # Set to True to save charts as images
        # output_folder = "chart_images"  # Folder where images will be saved
        # train = True

        self.process_windows(
            result,
            first_window_start_date,
            self.few_shot_train_set,
            save_charts=save_charts,
            output_folder=output_folder,
            train=train,
        )
        print("train data saved in train folder...")
        return first_window_start_date

    def calculate_profit_and_deviation(
        self, target_data, predicted_high, predicted_low, train_data
    ):
        """
        This function takes in a DataFrame (target_data), adds predicted high and low values,
        and calculates potential profit and deviation based on the logic provided.

        Parameters:
            target_data (pd.DataFrame): The input DataFrame containing stock data.
            predicted_high (float or pd.Series): The predicted high value(s).
            predicted_low (float or pd.Series): The predicted low value(s).

        Returns:
            pd.DataFrame: The updated DataFrame with the calculated potential profit and deviation.
        """
        # Make a copy of the target data
        df = target_data.copy()
        train_dict = train_data.to_dict()

        # Add predicted high and low values
        if predicted_high != None:
            df["Predicted_High"] = predicted_high
        if predicted_low != None:
            df["Predicted_Low"] = predicted_low

        # Convert relevant columns to numeric in case they're strings
        for col in ["High", "Low", "Predicted_High", "Predicted_Low"]:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except:
                continue

        if "Predicted_High" in df.columns:
            df["trade_result"] = df.apply(
                lambda row: (
                    1
                    if (row["Predicted_High"] <= row["High"])
                    and (row["Predicted_High"] > row["Open"])
                    else 0
                ),
                axis=1,
            )

            # Calculate potential profit
            df["prediction_deviation_High"] = df.apply(
                lambda row: (-row["Predicted_High"] + row["High"]),
                axis=1,
            )
            df["profit"] = df.apply(
                lambda x: (
                    -x["Open"] + x["Predicted_High"]
                    if (x["prediction_deviation_High"] >= 0)
                    and (-x["Open"] + x["Predicted_High"]) > 0
                    else 0
                ),
                axis=1,
            )
            df["profit_(10000)"] = df.apply(
                lambda x: x["profit"] * (round(10000 / x["Open"], 0)), axis=1
            )

        if "Predicted_Low" in df.columns:
            # Calculate deviation for predicted low
            df["prediction_deviation_Low"] = df.apply(
                lambda row: (+row["Predicted_Low"] - row["Low"]),
                axis=1,
            )

        df["ticker"] = self.ticker
        df["train_window"] = self.train_window
        df["target_window"] = self.target_window
        df["indicators"] = self.indicators
        df["timeframe"] = self.timeframe
        df["training_data"] = train_dict
        return df

    def calculate_profitable_percentage(self, df, profit_column="trade_result"):
        """
        Calculates the percentage of profitable trades in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the stock data.
        profit_column (str): The name of the column containing potential profit values.

        Returns:
        float: The percentage of profitable trades (greater than 0).
        """
        # Count profitable trades (where Potential_Profit > 0)
        profitable_trades = len(df[df[profit_column] > 0])

        # Count total trades
        total_trades = len(df)

        # Calculate the percentage of profitable trades
        if total_trades > 0:
            profitable_percentage = (profitable_trades / total_trades) * 100
        else:
            profitable_percentage = 0

        return np.round(profitable_percentage, 3)

    def create_folder_if_not_exists(self, main_path, ticker):
        """
        Checks if a folder exists at the given path, and if not, creates it.

        Parameters:
        main_path (str): The base directory path.
        ticker (str): The ticker symbol to be used in the folder name.

        Returns:
        str: The full path to the created or existing folder.
        """
        # Build the full path
        folder_path = os.path.join(main_path, ticker)

        # Check if the directory exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Directory created: {folder_path}")
        else:
            print(f"Directory already exists: {folder_path}")

        return folder_path

    def calculate_metrics(self, df):
        df_copy = df.copy()

        if "prediction_deviation_High" in df_copy.columns:
            # Filter for negative deviations
            negative_deviation_high = df_copy["prediction_deviation_High"][
                df_copy["prediction_deviation_High"] < 0
            ]

            # Calculate metrics
            mnd = negative_deviation_high.mean()
            msnd = (negative_deviation_high**2).mean()
            mand = negative_deviation_high.abs().mean()
            pup = len(negative_deviation_high) / len(df_copy)
            usi = pup * mand

            # Add metrics as new columns to the original DataFrame
            df_copy["High_Mean_Negative_Deviation"] = mnd
            # df_copy["High_Mean_Squared_Negative_Deviation"] = msnd
            # df_copy["High_Mean_Absolute_Negative_Deviation"] = mand
            # df_copy["High_Proportion_of_Undershoots"] = pup
            # df_copy["High_Undershoot_Severity_Index"] = usi

        if "prediction_deviation_Low" in df_copy.columns:
            # Filter for negative deviations
            negative_deviation_low = df_copy["prediction_deviation_Low"][
                df_copy["prediction_deviation_Low"] < 0
            ]

            # Calculate metrics
            mnd = negative_deviation_low.mean()
            msnd = (negative_deviation_low**2).mean()
            mand = negative_deviation_low.abs().mean()
            pup = len(negative_deviation_low) / len(df_copy)
            usi = pup * mand

            df_copy["Low_Mean_Negative_Deviation"] = mnd
            # df_copy["Low_Mean_Squared_Negative_Deviation"] = msnd
            # df_copy["Low_Mean_Absolute_Negative_Deviation"] = mand
            # df_copy["Low_Proportion_of_Undershoots"] = pup
            # df_copy["Low_Undershoot_Severity_Index"] = usi

        return df_copy

    def get_unique_timestamp(self):
        # Get the current datetime
        now = datetime.now()
        # Format it as DDMMYYYY_HHSS
        timestamp = now.strftime("%d%m%Y_%H%M")
        return timestamp

    def track_trades(self, df):
        trade_tracking_value = 0
        failed_trades = []

        # Initialize the new trade_tracking column
        df["trade_tracking"] = 0
        old_trade_closed = False

        for index, row in df.iloc[::-1].iterrows():
            current_open_price = row["Open"]
            current_high_price = row["High"]
            current_predicted_high = row["Predicted_High"]
            trade_result = row["trade_result"]
            if old_trade_closed:
                trade_tracking_value = 0
                old_trade_closed = False

            if trade_tracking_value == 0 and trade_result == 0:  # Failed trade is 0
                failed_trades.append(current_open_price)
                prev_open_price = current_open_price
                prev_predicted_high = current_predicted_high
                trade_tracking_value -= 1

            elif trade_tracking_value < 0 and current_high_price < prev_predicted_high:
                failed_trades.append(current_open_price)
                trade_tracking_value -= 1

            elif (
                trade_tracking_value < 0 and current_high_price >= prev_predicted_high
            ):  # Successful trade
                # Check if the high price surpasses the open prices of failed trades

                trade_tracking_value -= 1
                prev_predicted_high = 0
                prev_open_price = 0
                old_trade_closed = True

            # Store the tracking value in the new column
            df.at[index, "trade_tracking"] = trade_tracking_value

        return df
