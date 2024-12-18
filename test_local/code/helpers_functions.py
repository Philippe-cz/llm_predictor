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
    final_prediction_prompt,
    system_prompt,
)
import mplfinance as mpf
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np


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

    def test_pipeline(self, stock_data, target_folder, target_final_folder):
        """
        function will take arguments and build train set for few shots and make prediction

        stock_data : dataframe including entire stock data set , up until point of target window '
        target_final_date : last date of target window
        train_window : train window
        target_window : target window
        """
        print("target window : ", self.target_window)
        windowed_dataset = self.create_windows_with_labels(stock_data)

        if self.target_window > 1:

            windowed_dataset["label"] = windowed_dataset["label"].apply(merge_df)

        target_final_date = stock_data["Date"].to_list()[0]
        # Slice data and remove target window that is to be predicted
        stock_data_adjusted = stock_data.head(self.train_window)

        # prepare image charts for few shots
        self.build_dynamic_few_shots(
            windowed_dataset, target_final_date, target_folder, train=True
        )
        # load saved images and create mapping dictionaries
        # folder = "chart_images/train"  # Replace with your target folder
        images_data_dict, mapping_dict = self.build_image_and_label_mappings(
            target_folder
        )

        general_dict = {
            "TICKER": self.ticker,
            "INDICATORS": self.indicators,
            "TIMEFRAME": self.timeframe,
        }

        resulting_few_shots = self.construct_few_shots(
            few_shot_prompt, images_data_dict, mapping_dict, general_dict
        )

        print("\n\nFINAL FORMATION....\n")

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
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            seed=123456,
        )

        print(response.choices[0].message.content)
        return messages, response.choices[0].message.content

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
        self, few_shot_prompt, images_data_dict, mapping_dict, general_dict
    ):
        few_shots = []  # Initialize an empty list for few shots

        # Iterate over the images_data_dict
        for counter, (image_name, image_base64) in enumerate(images_data_dict.items()):
            high_label = mapping_dict[image_name][0]
            low_label = mapping_dict[image_name][1]

            # Format the few_shot_prompt with specific labels
            formatted_prompt = few_shot_prompt.format(
                COUNTER=counter,
                TICKER=general_dict["TICKER"],
                TIMEFRAME=general_dict["TIMEFRAME"],
                INDICATORS=general_dict["INDICATORS"],
                high_label=high_label,
                low_label=low_label,
            )

            # Constructing the message template
            template = {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{image_base64}"},
                    },
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

                # Save the figure
                fig.savefig(image_filename)
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

            # Uncomment the next line if you want to wait for user input before proceeding
            # input("Press Enter to continue to the next window...")

    # def plot_candlestick(self, window_df):
    #     # Prepare data
    #     window_df = window_df.copy()
    #     window_df["Date"] = pd.to_datetime(window_df["Date"])
    #     window_df.set_index("Date", inplace=True)

    #     # Plot using mplfinance and return the figure
    #     fig, axlist = mpf.plot(
    #         window_df,
    #         type="candle",
    #         style="charles",
    #         title="Candlestick Chart",
    #         ylabel="Price ($)",
    #         volume=True,
    #         show_nontrading=True,
    #         returnfig=True,  # Return the figure object
    #     )
    #     return fig

    def plot_candlestick(self, window_df):
        # Prepare data
        window_df = window_df.copy()
        window_df["Date"] = pd.to_datetime(
            window_df["Date"]
        )  # Ensure Date column is in datetime format
        window_df.set_index("Date", inplace=True)  # Set Date as the index

        # Create a candlestick chart using mplfinance
        fig, axlist = mpf.plot(
            window_df,
            type="candle",
            style="charles",
            title="Candlestick Chart",
            ylabel="Price ($)",
            volume=True,
            show_nontrading=True,
            returnfig=True,  # Return the figure object
        )

        # Get the main axis (first in axlist)
        ax = axlist[0]

        # Customize the x-axis to display Mondays correctly
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(byweekday=mdates.MONDAY)
        )  # Align ticks to Mondays
        ax.xaxis.set_major_formatter(
            DateFormatter("%Y-%m-%d")
        )  # Format dates as YYYY-MM-DD

        # Rotate the date labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

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

            # Save the figure
            fig.savefig(file_path)
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

        return images_data_dict, mapping_dict

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
        df["Predicted_High"] = predicted_high
        df["Predicted_Low"] = predicted_low

        # Convert relevant columns to numeric in case they're strings
        for col in ["High", "Low", "Predicted_High", "Predicted_Low"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calculate potential profit
        df["Potential_Profit"] = df.apply(
            lambda row: (
                row["Predicted_High"] - row["High"]
                if row["Predicted_High"] > row["High"]
                else 0
            ),
            axis=1,
        )

        # Calculate deviation for predicted low
        df["Deviation_Low"] = df.apply(
            lambda row: (
                row["Predicted_Low"] - row["Low"]
                if row["Predicted_Low"] > row["Low"]
                else 0
            ),
            axis=1,
        )
        df["ticker"] = self.ticker
        df["train_window"] = self.train_window
        df["target_window"] = self.target_window
        df["indicators"] = self.indicators
        df["timeframe"] = self.timeframe
        df["training_data"] = train_dict
        return df

    def calculate_profitable_percentage(self, df, profit_column="Potential_Profit"):
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
        folder_path = os.path.join(main_path, f"chart_images/{ticker}")

        # Check if the directory exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Directory created: {folder_path}")
        else:
            print(f"Directory already exists: {folder_path}")

        return folder_path
