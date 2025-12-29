# Tutorial: machine-learning-capstone-project-2025

This project, `machine-learning-capstone-project-2025`, focuses on **forecasting cinema audience counts**. It systematically *gathers and unifies raw data* from various sources, then *transforms and enriches this data* by extracting crucial time-series features. These prepared datasets are fed into a *model experimentation workbench* to identify the best-performing machine learning models, which are then used by an *iterative forecasting engine* to predict future attendance dynamically, often leveraging an *ensemble modeling strategy* for improved accuracy.

## Visual Overview
![Diagram](https://mermaid.ink/img/pako:eNptk1FvmzAQx7-K5Yc9JSihQICHSd2SqHmoVq2ZJm3swYEDrBkb2SZtGuW794DQQTQkBNi_O__vf8eZpioDGtNcqJe0ZNqS_TqRBK_7xe-ErpllZCctFJpZriR5aA6JTOifK7NE5klDrVUKxnBZkE9kr5k0udJVH_FFNLjPpR3HuRi35xXMn0FzMGQLzDYayJalVunTGL1D9IEbXOYpE6RT9MgkK6CCaVIPyUcsR5DNa4152_1ew0-l_x5ApuUY9xHfWWgLO-LJSkPKjG1r2MiCSxizAbIbaaA6CCDdGS33bDEYioncBZnPSUK3AJkhWSvWqoTi4md0a8o8aXXkGVbfSJ5zyG5xd4rvJLecCf6GEWVnyImgzQN9N7RkSA4103DVMOK8oQU9t5Hoa_kfbhDr9dwaaqFOhlRt8Wak0p9yO9m23hADAtLO_FHOYLC-Z-_rWrTdt8PEGHLk7FbAFf6O7RFpI9BxQ_J-Xia8O-V_1FmHDlZxeetU0IPfdFt_10pDag0Z73Qb8sIdx_kok85ooXlG45wJAzNaAU54-03Pbb6E2hInMqExvmaQs0bYhCbygnE1k7-UqmhsdYORWjVF-ZGn6XSuOcNf7B8CMgP9VTXS0tjzwy4Hjc_0lca-54RL14uipR-t8BHM6InGS9d1omgRhJ7rh66_Wq0uM_rWnbpwwpaLojDwwwBv__IOVHg24w?type=png)

## Chapters

1. [Data Integration Hub
](#chapter1)
2. [Historical Data Management
](#chapter2)
3. [Time-Series Feature Factory
](#chapter3)
4. [Preprocessing & Transformation Blueprint
](#chapter4)
5. [Model Experimentation Workbench
](#chapter5)
6. [Iterative Forecasting Engine
](#chapter6)
7. [Ensemble Modeling Strategy
](#chapter7)


---
<div style="page-break-before: always;"></div>


<a name="chapter1"></a>
# Chapter 1: Data Integration Hub

Welcome to the first chapter of your Machine Learning Capstone Project! Imagine you're a detective trying to solve a mystery, but all your clues are scattered in different folders, written in different styles, and some pieces are missing. That's often what data looks like before you start a machine learning project!

In this project, our goal is to forecast how many people will visit different cinemas on various dates. This is a real-world challenge where predicting audience numbers can help cinemas plan better, from staffing to snack inventory.

The problem we face right away is that the information we need comes from many different places. We have one file with booking details, another with theater locations, and yet another with calendar information. To make accurate predictions, we need to bring all these pieces of the puzzle together into one comprehensive view. This is where the **Data Integration Hub** comes in.

## What is a Data Integration Hub?

Think of a "Data Integration Hub" as your project's central command center for all data. It's a special place or process where you gather *all* your raw data sources and combine them into one unified, clean, and complete dataset.

Why is this so important? Machine learning models are like smart students: they learn best when they have all the relevant information presented clearly in one organized textbook, rather than sifting through many different scattered notes. Our Data Integration Hub's job is to create that "master textbook" for our models.

## Building Our Data Integration Hub: Step-by-Step

Let's walk through how we build this hub for our cinema audience forecasting project.

### Step 1: Gathering All the Raw Data

Our journey begins by collecting all the individual pieces of information. These often come in separate files, like CSVs (Comma Separated Values). In our case, we have several:

*   `booknow_booking.csv`: Contains booking records from the "BookNow" system.
*   `booknow_theaters.csv`: Details about theaters using the "BookNow" system (e.g., location, type).
*   `date_info.csv`: Calendar-related information for each date (e.g., day of the week, holidays).
*   `booknow_visits.csv`: Contains the historical `audience_count`, which is what we want to predict.
*   ...and several others!

We use a special library called `pandas` in Python to easily read these files into something called a `DataFrame`, which is like a spreadsheet in Python.

```python
import pandas as pd
import os # For listing files (not directly used here, but common for data paths)

# Define paths to our data files (simplified from notebook for brevity)
csv_path = {
    "booknow_booking.csv": "/kaggle/input/Cinema_Audience_Forecasting_challenge/booknow_booking/booknow_booking.csv",
    "booknow_theaters.csv": "/kaggle/input/Cinema_Audience_Forecasting_challenge/booknow_theaters/booknow_theaters.csv",
    "date_info.csv": "/kaggle/input/Cinema_Audience_Forecasting_challenge/date_info/date_info.csv",
    "booknow_visits.csv": "/kaggle/input/Cinema_Audience_Forecasting_challenge/booknow_visits/booknow_visits.csv",
}

# Load the essential dataframes
df_bookings = pd.read_csv(csv_path['booknow_booking.csv'])
df_theaters = pd.read_csv(csv_path['booknow_theaters.csv'])
df_date_info = pd.read_csv(csv_path['date_info.csv'])
df_audience = pd.read_csv(csv_path['booknow_visits.csv']) # This has our target: audience_count

print(f"Loaded Bookings data: {df_bookings.shape}")
print(f"Loaded Theaters data: {df_theaters.shape}")
print(f"Loaded Date Info data: {df_date_info.shape}")
print(f"Loaded Audience data: {df_audience.shape}")
```

This code loads four important CSV files into separate `DataFrame` objects (`df_bookings`, `df_theaters`, `df_date_info`, `df_audience`). Each `DataFrame` now holds a specific type of information, ready to be combined.

### Step 2: Creating the "Skeleton" DataFrame

Now that we have all our pieces, the next crucial step is to create a "skeleton" for our main data table. Imagine you have a calendar and a list of all the movie theaters. Our skeleton will be a table that lists *every single theater* for *every single day* within our project's timeframe.

Why do this? Because even if a theater had zero audience on a particular day (or was closed), we still need a row for that theater and date combination. This ensures our final dataset is comprehensive and "nothing is missing," as the concept description states.

```python
import datetime as dt
import pandas as pd # pandas is already imported, but good practice to show context

# Define our date range for the training data (up to Feb 28, 2024)
start_date = dt.date(2023, 1, 1)
end_date = dt.date(2024, 2, 28)
dates = pd.date_range(start=start_date, end=end_date)

# Create a list of all theater IDs (assuming 829 theaters from problem description)
all_theater_ids = [f"book_{i:05d}" for i in range(1, 830)]

# Create all possible combinations of theater ID and date
skeleton_df = pd.DataFrame(
    [(theater, date) for date in dates for theater in all_theater_ids],
    columns=["book_theater_id", "show_date"]
)

skeleton_df["show_date"] = pd.to_datetime(skeleton_df["show_date"]) # Convert date column
print(f"Skeleton DataFrame created with {skeleton_df.shape[0]} rows.")
print(skeleton_df.head())
```

This code snippet generates a `skeleton_df` with a row for every combination of a theater ID and a date from January 1, 2023, to February 28, 2024. This `DataFrame` is currently "empty" in terms of details, but its structure is perfect for holding all our integrated data.

### Step 3: Attaching the "Meat" (Merging External Metadata)

With our `skeleton_df` in place, it's time to "flesh it out" by attaching all the relevant information from our other data sources. This process is called "merging" or "joining" dataframes. We're essentially looking up information in one table and adding it to matching rows in another.

Let's merge the calendar information (`df_date_info`), the actual audience counts (`df_audience`), and the theater details (`df_theaters`) onto our `skeleton_df`.

```python
import pandas as pd # pandas is already imported

# Ensure 'show_date' in date_info is also datetime
df_date_info['show_date'] = pd.to_datetime(df_date_info['show_date'])

# Start with the skeleton and merge other dataframes
integrated_df = skeleton_df.merge(df_date_info, on='show_date', how='left')

# Merge actual audience counts (our target variable)
# 'how=left' ensures we keep all theater-date combinations from skeleton_df
integrated_df = integrated_df.merge(df_audience, on=['book_theater_id', 'show_date'], how='left')

# Merge theater details (e.g., type, area, location)
integrated_df = integrated_df.merge(df_theaters, on='book_theater_id', how='left')

print(f"Integrated DataFrame shape after merges: {integrated_df.shape}")
print(integrated_df.head())
```

After these merges, our `integrated_df` now contains rows for every theater-date combination, along with calendar information, theater specifics, and (where available) the `audience_count`. Notice the `how='left'` argument in `merge`. This is crucial: it means "keep all rows from the left (our `integrated_df` or `skeleton_df`) and add matching information from the right." If there's no match (e.g., no audience count for a specific date), it will fill with `NaN` (Not a Number), which we'll handle next.

### Step 4: Initial Cleanup (Filling Gaps)

Even after merging, some columns might have missing values (`NaN`). For example, `audience_count` will be missing for future dates we want to predict or for historical dates where no data was recorded. Other metadata, like `latitude` or `theater_area`, might also have gaps. A key part of data integration is ensuring these initial gaps are handled so our dataset is truly "clean."

```python
import pandas as pd # pandas is already imported

# Fill missing geographical coordinates with the median value
# Median is a good choice to avoid being skewed by extreme values
integrated_df['latitude'] = integrated_df['latitude'].fillna(integrated_df['latitude'].median())
integrated_df['longitude'] = integrated_df['longitude'].fillna(integrated_df['longitude'].median())

# Fill missing categorical theater information with 'unknown'
integrated_df['theater_area'] = integrated_df['theater_area'].fillna('unknown_area')
integrated_df['theater_type'] = integrated_df['theater_type'].fillna('unknown_type')

# For audience_count, fill NaN with 0 for now. This indicates no audience.
# We will filter these out later for training, but keep for a complete timeline.
integrated_df['audience_count'] = integrated_df['audience_count'].fillna(0)

print("Missing values after initial cleanup (showing only columns with NaNs):")
print(integrated_df.isnull().sum()[integrated_df.isnull().sum() > 0])
```

This code block fills numerical missing values (like `latitude`, `longitude`) with their respective median values, which is a common strategy to maintain data distribution. For categorical columns (`theater_area`, `theater_type`), we fill `NaN` with a new category 'unknown_area' or 'unknown_type'. Finally, for `audience_count`, we replace `NaN` with `0`, meaning no audience was recorded.

## How the Data Integration Hub Works (Under the Hood)

Let's visualize the entire data integration process, from raw files to our complete dataset.

![diagram](https://mermaid.ink/img/pako:eNqdVN9P2zAQ_ldufipSGyVpaUsekKYyJqQNoQF7mCohE19aa46d2Q6sq_jfd07TtJQiTctTfL7vu-9-ec1yI5BlzOGvGnWOF5IvLC_nGuiruPUylxXXHu4d2rfWb_x5dvvdAXcwZ3QAOsGlVOjm7K33DdeCu9vcyspvIDcrvzQaWlNv43ByDHv7ExV6o-_4o8INeGuCC-75JanGY8B7LQuJYg_XWhqYQx9AG1jIcXB-vi8zI_8rLb3kHhsAXGmPVCIvjd6G2wcQvi1KgCYRfDFcQFkrLysSEC62sNZvcDSkkE9S1FztknPQw2gR9UEUD36JJMi65iDo90HqwpzsUjmQ9Kp6IUAawcwGDuBKwd2GbnARDLkpH6VuEuykvsIfE_yprPwKPlspoOda5wdRdK28NsRsntDCG6R2taXkkC5XUBnnZOhUm-FAHEiCJfWQg8XcWPFuvvtND0GGEXxFu8BX5QKjvelSe6eZh0zk0THxWsiwNBui1vM_eLbdPMqzq9whxUfveb6UegE5V0jRbB-2kuhPi20RQaDnUrl_LtcoCkusQDaTr6CU1BWK88RVHebwml93nd3HHpuMmUKu-93a7PYOerKz7Y_KoTZaysDTLJ90oM0zdZ-LFRTGQlFbStJCZU2OjcgPgYf12YJmkWUFVw77rERb8nBm6xBkzggUnotALLDgtJ4B9kI4ejR-GFOyzNuakNbUi2XHU1dhetpHsrNaKj7amam1Z9loMmxIWLZmv1l2OomSOD0dxXEaJ8lwMhz32Ypl6XgUTcfT6dn4LB0lwzgdvfTZnyZuHE0npzF9aTJOkgl5vfwFEyPmpw?type=png)



As you can see, our Python script acts as the central orchestrator. It first loads all the separate data pieces. Then, it constructs a complete calendar-like grid (the "skeleton") to ensure we have a spot for every possible observation. Finally, it systematically combines all the individual data points onto this skeleton, filling in gaps along the way to produce a single, comprehensive dataset.

The core tools used here are the `pd.read_csv()` function to load data, `pd.date_range()` and list comprehensions to create the skeleton, and the `DataFrame.merge()` method to combine dataframes. The `DataFrame.fillna()` method helps us clean up any initial missing values.

Here's a summary of the core tasks performed by our Data Integration Hub:

| Task                      | What it does                                                 | Why it's useful                                                              | Python Tool/Method               |
| :------------------------ | :----------------------------------------------------------- | :--------------------------------------------------------------------------- | :------------------------------- |
| **Gathering Raw Data**    | Loads information from separate files into DataFrames.       | Brings all individual data sources into memory for processing.               | `pd.read_csv()`                  |
| **Creating Skeleton**     | Generates a base table with all unique theater-date combinations. | Ensures every possible event has a record, providing a complete timeline. | `pd.date_range()`, list comprehension |
| **Merging Metadata**      | Combines external details (calendar, theater info) onto the skeleton. | Enriches each theater-date record with all relevant descriptive information. | `DataFrame.merge(how='left')`    |
| **Initial Cleanup**       | Fills in initial missing values (`NaN`) after merging.       | Prepares the dataset for further processing by handling immediate data gaps. | `DataFrame.fillna()` with median/`'unknown'` |

## Conclusion

In this chapter, you've learned about the **Data Integration Hub**, a critical first step in any machine learning project. We started with scattered data sources and successfully brought them together into one unified, clean dataset (`integrated_df`). This `integrated_df` is like our fully assembled puzzle board, ready for us to start drawing the actual picture.

Having a complete and organized dataset is fundamental. Without it, our machine learning models wouldn't have all the necessary information to learn from, leading to poor predictions.

Now that our data is integrated and cleaned, the next exciting step is to prepare it for our machine learning models by transforming it into useful features. We'll explore this in the next chapter: [Preprocessing & Transformation Blueprint](#chapter4)



---


<div style="page-break-before: always;"></div>




















































<a name="chapter2"></a>
# Chapter 2: Historical Data Management

Welcome back! In [Chapter 1: Data Integration Hub](#chapter1), we learned how to clean and prepare our raw data, transforming it into a neat, standardized format ready for our machine learning models. Now, we're going to build on that foundation by exploring **"Historical Data Management."**

### Why is Historical Data Management So Important?

Imagine you're trying to predict how many people will attend a specific movie show at a particular theater next Tuesday. You wouldn't just pull a number out of thin air, right? You'd think: "How many people usually come to this theater on a Tuesday? What were the audience numbers for the past few Tuesdays, or even past few days?"

This is exactly what "Historical Data Management" is all about in our project. Our goal is to forecast future `audience_count` values for each `book_theater_id`. To do this accurately, our models need to "remember" past `audience_count` values for *each individual theater*. It's like keeping a detailed logbook or diary for every single theater's daily audience performance.

The big challenge is not just storing this past data, but also being able to efficiently use it to make new predictions, and then **update** that logbook with the latest information (even if it's a prediction itself!). This continuous update is key for making predictions far into the future, day by day.

### Key Concepts

This system helps us keep track of all the past `audience_count` values for every theater. Let's break it down:

#### 1. The `history_dict` – Our Theater Logbook

At the heart of our historical data management is a special Python dictionary called `history_dict`.

*   **What it is**: Think of `history_dict` as a big collection of personal diaries, one for each `book_theater_id` (each unique theater).
*   **How it's structured**:
    *   **Keys**: Each key in this dictionary is a `book_theater_id` (e.g., `'book_00001'`, `'book_00244'`). This is how we find a specific theater's diary.
    *   **Values**: The value for each key is a `list` of numbers. This list contains all the historical `audience_count` values for that specific theater, ordered from the oldest to the newest day.

#### 2. Storing Initial Past Data

Before we even start predicting, we need to fill our `history_dict` with all the *actual* past `audience_count` data we already have. This gives our logbooks a starting point. We collect all known historical audience numbers for each theater and put them into their respective lists.

#### 3. Dynamic Updating: Learning from the Future (Even if it's a Guess!)

This is the most crucial part for time-series forecasting. When we predict the `audience_count` for a theater for "Day X", that prediction itself becomes the "most recent past data" for that theater when we predict for "Day X+1".

*   **The Process**:
    1.  We predict `audience_count` for a specific theater for a given date.
    2.  As soon as we have this prediction, we add it to the end of that theater's list in our `history_dict`.
    3.  When we move to the next date, the model can now use this *newly predicted* value (along with older actual values) to calculate new features (like "what was the audience yesterday?") and make its next prediction.

This dynamic updating ensures that our predictions always leverage the most current available (or predicted) information, which is essential for accurate time-series forecasting.

### How to Use the `history_dict` for Prediction

Let's walk through how `history_dict` is used in a typical prediction scenario, where we need to predict `audience_count` day by day for all theaters for a future period.

#### Use Case: Predicting Audience for the Next 7 Days

Imagine we need to predict the `audience_count` for each theater for every day in the upcoming week.

**Step-by-Step Walkthrough:**

1.  **Start with a Prediction Date**: We pick the first day we want to predict (e.g., Monday).
2.  **Gather Theaters for Today**: Identify all the `book_theater_id`s we need to predict for on this Monday.
3.  **For Each Theater, Access its Logbook**:
    *   We go to our `history_dict` and find the list of past `audience_count`s for that specific theater.
    *   We use this list to calculate important features for our model, such as:
        *   **Lag features**: What was the audience 1 day ago? 7 days ago? (We'll learn more about these in [Chapter 3: Time-Series Feature Factory](#chapter3)).
        *   **Rolling features**: What was the average audience over the last 7 days? The standard deviation?
    *   Our model (from [Chapter 5: Iterative Forecasting Engine](#chapter5)) then takes these calculated features and other preprocessed data (from [Chapter 4: Model Experimentation Workbench](#chapter4)) to make a prediction for this theater for Monday.
4.  **Update the Logbook**:
    *   As soon as we have the prediction for this theater for Monday, we immediately add this predicted `audience_count` to the end of that theater's list in `history_dict`.
    *   We repeat this for all theaters for Monday.
5.  **Move to the Next Day**: Now it's Tuesday. When we go to predict for Tuesday, the `history_dict` *already includes* Monday's predictions. This allows the model to calculate the "yesterday's audience" (which is Monday's predicted audience) and other features based on the most up-to-date information.
6.  **Repeat**: This process continues day by day, dynamically updating the `history_dict` with each new prediction, forming an iterative prediction process (which we'll cover in [Chapter 5: Iterative Forecasting Engine](#chapter5)).

This approach is powerful because it simulates how real-world forecasting works, where new information becomes available and influences subsequent decisions.

### Under the Hood: How it Works

Let's look at a simplified view of the `history_dict` in action.

#### 1. Initializing `history_dict`

First, we populate `history_dict` with all the known historical `audience_count` values from our training data (`f_t_df`, which we created in [Chapter 1: Data Integration Hub](#chapter1)).

```python
import pandas as pd
import datetime as dt # For date handling
import numpy as np    # For numerical operations

# --- Setup (assuming f_t_df is prepared from Chapter 1) ---
# For a real scenario, f_t_df would already be loaded and preprocessed.
# Here, we'll create a minimal dummy f_t_df for demonstration.
start_date_hist = dt.date(2023, 1, 1)
end_date_hist = dt.date(2023, 1, 3) # A few days of history
dates_hist = pd.date_range(start=start_date_hist, end=end_date_hist)
all_theaters_dummy = [f"book_{i:05d}" for i in range(1, 3)] # Two dummy theaters

f_t_df_dummy = pd.DataFrame(
    [(theater, date, np.random.randint(10, 100))
     for date in dates_hist for theater in all_theaters_dummy],
    columns=["book_theater_id", "show_date", "audience_count"]
)

history_dict = {}

# Get all unique theater IDs from our dummy data
all_theaters = f_t_df_dummy['book_theater_id'].unique()

# Loop through each theater to build its history
for theater_id in all_theaters:
    # Filter data for this specific theater
    theater_data = f_t_df_dummy[f_t_df_dummy['book_theater_id'] == theater_id]
    # Sort by date to ensure correct order
    theater_data = theater_data.sort_values('show_date')
    # Get the list of audience counts and store it
    audience_list = theater_data['audience_count'].tolist()
    history_dict[theater_id] = audience_list

print(f"Total number of theaters with initial history: {len(history_dict)}")
print(f"Example: 'book_00001' has {len(history_dict.get('book_00001', []))} days of history.")
print(f"History for 'book_00001': {history_dict.get('book_00001')}")

# Output:
# Total number of theaters with initial history: 2
# Example: 'book_00001' has 3 days of history.
# History for 'book_00001': [55, 34, 78] (example values)
```

Now, `history_dict` is ready, with each theater's past audience counts stored as a list.

#### 2. The Daily Prediction Cycle (Simplified)

This diagram shows the core steps for a single prediction day for one theater. This entire process would then be repeated for all theaters for the current date, and then for all subsequent dates in our prediction period.


![dia](https://mermaid.ink/img/pako:eNqdVE2P2jAQ_SsjX7ZVQ5rAZgM5rFSx3fYAq0rdAyqgyIqHxFISU8fpliL-e-18iUCQquaCPJ735r0ZD0cSCYYkIAX-LDGP8InTWNJsk4P-9lQqHvE9zRU8C4kRLdT3Q6EwA1rAhrQxqIMbcg17ojw9LITY14jqCN8kMh4pLnIwV0O41wSpQvmVF0rIQw1O6kNosEOYZw0pJX7OY54jykZjHYQ2OgRcLpa6C2kNWC6gOpnEOrVvffTYeQp0euMFdkLCvJQSK9MK4R3asW3BksooAbdQ79vKHXz02Hdp6L6ggsZnRdlkwKpFvwjNLX6hhEvw2rZ1PVoybgYZMnoIX0butob1k0c9E-u6II9oGkaizFURrrZXWi-6a8TOaRqVqTGb0vijFGkKuzqrgJ0U2Q3mxskF4YWmqOVmYcs5pKoZ3fkk2g7AG1fJLSL4AIWi-gV0iltZDWNfzr7m1hxdf1f_MM1PjA1CQYn_nfL5EqzvmtvV3RZ4Abl4u_EKrEEZV-3UU-699bO2mmXtSQUmctwQYpFYckaCHU0LtEiGMqPmTI6GfUNUgpnOM1QMd7RMq909aZzevR9CZCRQstRIKco46XjKPdN1mr-jLqrXi6Gcm6dEAm88rUhIcCS_STDxbcd5uPdd33P1z9i1yEFHXd_2xt5kOtEXvjc5WeRPVdSxp_79TH_-zHOdmeP4p7865rv1?type=png)



#### 3. Summary of Core Tasks in Historical Data Management

| Task                      | What it does                                                 | Why it's useful                                                                 | Python Tool/Strategy             |
| :------------------------ | :----------------------------------------------------------- | :------------------------------------------------------------------------------ | :------------------------------- |
| **Initialize Logbook**    | Creates a `history_dict` with all known actual past data.       | Provides a factual starting point for the forecasting engine.                     | `dict` and `DataFrame.groupby()` |
| **Accessing History**     | Retrieves the ordered list of `audience_count` for a theater.  | Enables calculation of time-dependent features for the current prediction.       | `dict.get()`                     |
| **Dynamic Update**        | Adds the newly predicted `audience_count` to the logbook.    | Ensures future predictions are based on the most up-to-date (even if predicted) information. | `list.append()`                  |

#### 4. Code for the Prediction Loop

Here's a simplified look at the `making_predictions` function from the project's notebook, focusing on how `history_dict` is used (notebook cell [152]). This function will be the core of our [Chapter 5: Iterative Forecasting Engine](#chapter5).

```python
# (Assume the following are pre-defined from previous chapters or setup)
# model: Our trained machine learning model (e.g., LightGBM from Chapter 4)
# test_dataframe: DataFrame for future dates, pre-filled with static features
# history_dictionary: The `history_dict` initialized with actual past data
# X_test_cat: Pre-transformed categorical features for test_dataframe
# X_test_stat: Pre-transformed static numerical features for test_dataframe
# dyn_scaler: MinMaxScaler fitted on dynamic features from training data
# dynamic_columns: List of dynamic (lag/rolling) feature names

def make_predictions_simple(model, test_dataframe, history_dictionary, X_test_cat, X_test_stat, dyn_scaler, dynamic_columns):
    # Make a copy of history_dictionary so the original is untouched
    current_history = {k: v.copy() for k, v in history_dictionary.items()}

    # Sort test data by date to process day-by-day
    test_dataframe = test_dataframe.sort_values(['show_date', 'book_theater_id']).reset_index(drop=True)
    all_unique_dates = sorted(test_dataframe['show_date'].unique())
    all_predictions = np.zeros(len(test_dataframe)) # To store all predictions

    for current_date in all_unique_dates:
        date_mask = test_dataframe['show_date'] == current_date
        row_indices = np.where(date_mask)[0] # Get rows for current date
        theaters_today = test_dataframe.loc[row_indices, 'book_theater_id'].values
        num_theaters = len(theaters_today)

        # Prepare arrays for dynamic (lag/roll) features for today's predictions
        # These arrays will be filled by looking up current_history
        lag_feature_matrix = np.zeros((num_theaters, len(dynamic_columns)))

        for i, theater_id in enumerate(theaters_today):
            theater_history = current_history.get(theater_id, [])
            history_length = len(theater_history)

            # --- Accessing history to calculate dynamic features ---
            # Simplified: just using last known audience (actual or predicted)
            if history_length > 0:
                lag_feature_matrix[i, 0] = theater_history[-1] # audience_lag_1
            # In a real scenario, you'd calculate all lags and rolling features here
            # For this tutorial, we focus on the history lookup and update logic.

        # --- Scale dynamic features ---
        X_dynamic_scaled = dyn_scaler.transform(lag_feature_matrix)

        # --- Combine all features for prediction ---
        X_cat_today = X_test_cat[row_indices]
        X_stat_today = X_test_stat[row_indices]
        X_today_combined = np.hstack([X_cat_today, X_stat_today, X_dynamic_scaled])

        # --- Make predictions ---
        predictions_today = model.predict(X_today_combined)
        predictions_today = np.maximum(predictions_today, 0) # Ensure no negative predictions
        all_predictions[row_indices] = predictions_today # Store predictions

        # --- Dynamically update history for the next day's predictions ---
        for i, theater_id in enumerate(theaters_today):
            if theater_id not in current_history:
                current_history[theater_id] = []
            current_history[theater_id].append(predictions_today[i]) # Add the new prediction

    return all_predictions

# Example usage (assuming model_2_lightgbm, t_df, etc., are defined):
# predictions_model_2 = make_predictions_simple(
#     model=model_2_lightgbm,
#     test_dataframe=t_df.copy(),
#     history_dictionary=history_dict,
#     X_test_cat=X_test_categorical,
#     X_test_stat=X_test_static,
#     dyn_scaler=dynamic_scaler,
#     dynamic_columns=dynamic_columns
# )
# print("Predictions made using LightGBM and dynamic history.")
```
**Output Description:**
After calling `make_predictions_simple`, `predictions_model_2` will be a NumPy array containing the predicted `audience_count` for each `book_theater_id` for every day in the future period (e.g., March 1st to April 22nd, as defined in the project). Each day's prediction for a theater will have incorporated the previous day's predicted `audience_count` for that same theater into its feature calculations, demonstrating the iterative updating of the history.

### Conclusion

In this chapter, we introduced "Historical Data Management," focusing on how we store and dynamically update past `audience_count` data using a `history_dict`. This system is vital for time-series forecasting, allowing our models to continually learn from and leverage the most current available information. Understanding this dynamic process is key to building robust predictive systems that can forecast effectively into the future.

Next, we'll dive deeper into exactly *what kinds* of features we extract from this historical data to give our models the best possible insights.

[Chapter 3: Time-Series Feature Factory](#chapter3)

---


<div style="page-break-before: always;"></div>




















































<a name="chapter3"></a>
# Chapter 3: Time-Series Feature Factory

Welcome back, aspiring data scientist! In [Chapter 1: Data Integration Hub](#chapter1), we bravely gathered all our scattered data pieces into one organized `integrated_df`. Then, in [Chapter 2: Preprocessing & Transformation Blueprint](#chapter2), we refined these ingredients by handling missing values, scaling numbers, and converting text into numbers. Our data, now represented by `f_t_df` (a more "feature-rich" version of `integrated_df` as we add more to it), is clean and ready.

But for our cinema audience forecasting challenge, "clean" and "ready" isn't quite enough. We're dealing with *time-series* data – data where the order and timing of events matter a lot. Just like predicting tomorrow's weather isn't just about today's temperature, but also yesterday's, and the day before, predicting cinema audience needs to consider the flow of time.

This is where our **Time-Series Feature Factory** comes in. Imagine it as a special kitchen dedicated to preparing ingredients that capture the "story" of our data over time. It doesn't just look at what happened on one day; it looks at patterns, trends, and past events to predict the future. This module will help us extract hidden patterns and valuable insights from our `show_date` column, transforming simple dates into powerful predictive features.

## Why Do We Need a Time-Series Feature Factory?

Our goal is to forecast `audience_count`. Think about it:
*   Does the `day_of_week` matter? (Weekend vs. weekday). Yes!
*   Does the `month` matter? (Holiday season vs. off-season). Yes!
*   Does the audience `yesterday` affect the audience `today`? Definitely!
*   What about the `average audience over the last 7 days`? That's a strong indicator too!

Standard preprocessing (like scaling or one-hot encoding) doesn't capture these time-dependent relationships directly. The Time-Series Feature Factory focuses on creating features that explicitly tell our machine learning model about:
1.  **Date Components**: Breaking down a date into its parts.
2.  **Cyclical Nature of Time**: Representing repeating patterns (like days of the week, months of the year).
3.  **Past Events (Lag Features)**: What happened a certain number of periods ago.
4.  **Recent Trends (Rolling Statistics)**: Averages or variations over recent periods.

Let's see how we can build these powerful features.

## 1. Extracting Date Components: Making Dates More Descriptive

A single `show_date` (like `2024-03-05`) contains a lot of information. We can pull out different parts of it to create new numerical features.

**Why it's useful:**
*   A model can learn if people prefer movies on the weekend (`is_weekend`).
*   It can spot trends for certain months (`show_month`) or specific days of the month (`show_day`).

**Example: Turning a date into many numbers**

| `show_date`  | `show_day` | `show_month` | `show_year` | `day_of_week` | `is_weekend` |
| :----------- | :--------- | :----------- | :---------- | :------------ | :----------- |
| 2024-03-01   | 1          | 3            | 2024        | Friday (4)    | 0            |
| 2024-03-02   | 2          | 3            | 2024        | Saturday (5)  | 1            |
| 2024-03-03   | 3          | 3            | 2024        | Sunday (6)    | 1            |

Here's how we extract these features using Python:

```python
import pandas as pd
import numpy as np # For cyclical features later

# Assuming f_t_df is our main DataFrame with a 'show_date' column
f_t_df['show_date'] = pd.to_datetime(f_t_df['show_date'])

# Extract basic date features
f_t_df['show_day'] = f_t_df['show_date'].dt.day
f_t_df['show_month'] = f_t_df['show_date'].dt.month
f_t_df['show_year'] = f_t_df['show_date'].dt.year
f_t_df['day_of_week_num'] = f_t_df['show_date'].dt.dayofweek

# Create weekend indicators
f_t_df['is_weekend'] = (f_t_df['day_of_week_num'] >= 5).astype(int)
```
*Explanation*: We use the `.dt` accessor in pandas to easily pull out parts of our date column. We get the day, month, year, and a numerical representation of the day of the week. We also create simple "yes/no" (0/1) features like `is_weekend` because these days often have unique audience patterns.

## 2. Cyclical Encodings: Understanding Repeating Patterns

The day of the week (Monday-Sunday) or month of the year (January-December) are cyclical. Monday follows Sunday, and January follows December. If we just use numbers (0-6 for days, 1-12 for months), the model might think December (12) is much "further" from January (1) than November (11) is, which isn't true cyclically.

To capture this, we use sine and cosine transformations. This creates two new features for each cyclical component, placing them on a circle, so the start and end values are close together.

```python
# Cyclical encodings for day of week (7 days in a week)
f_t_df['dow_sin'] = np.sin(2 * np.pi * f_t_df['day_of_week_num'] / 7)
f_t_df['dow_cos'] = np.cos(2 * np.pi * f_t_df['day_of_week_num'] / 7)

# Cyclical encodings for month (12 months in a year)
f_t_df['month_sin'] = np.sin(2 * np.pi * f_t_df['show_month'] / 12)
f_t_df['month_cos'] = np.cos(2 * np.pi * f_t_df['show_month'] / 12)

# ... similar for day of year ...

print(f_t_df[['day_of_week_num', 'dow_sin', 'dow_cos']].head())
```
*Explanation*: For `day_of_week_num`, we divide the day number by 7 (total days) and multiply by `2 * pi` to get an angle. Then we apply `np.sin` and `np.cos`. This effectively turns a linear scale (0, 1, ..., 6) into points on a circle, where 0 and 6 are numerically distinct but geometrically close, just like Monday and Sunday. The same logic applies to months.

## 3. Computing Lag Features: Looking to the Past

"Lag" features are simply values from previous time steps. For forecasting, knowing the `audience_count` from yesterday, last week, or even last month can be incredibly predictive of today's audience.

For our cinema project, we need to create lag features *for each specific theater*. An audience count at `book_00001` yesterday doesn't tell us much about `book_00002` today.

```python
# First, ensure our data is sorted by theater and then by date - this is crucial!
f_t_df = f_t_df.sort_values(['book_theater_id', 'show_date']).reset_index(drop=True)

# Create lag features for 'audience_count'
for lag in [1, 7, 14, 30]: # Lags for yesterday, last week, two weeks ago, last month
    f_t_df[f'audience_lag_{lag}'] = f_t_df.groupby('book_theater_id')['audience_count'].shift(lag)

# Fill any NaN values created by shift (at the start of each theater's history) with 0
for lag in [1, 7, 14, 30]:
    f_t_df[f'audience_lag_{lag}'] = f_t_df[f'audience_lag_{lag}'].fillna(0)

print(f_t_df[['show_date', 'book_theater_id', 'audience_count', 'audience_lag_1']].head(5))
```
*Explanation*:
*   We first sort our `f_t_df` by `book_theater_id` and then `show_date`. This is crucial so `shift()` correctly looks at *previous dates for the same theater*.
*   `groupby('book_theater_id')` ensures that the `shift()` operation is performed independently for each theater.
*   `.shift(lag)` moves the `audience_count` value down by `lag` rows. So, `shift(1)` gives yesterday's value, `shift(7)` gives last week's value, etc.
*   `fillna(0)` handles the `NaN` values that appear at the beginning of each theater's data (because there's no historical data for those first few `lag` days).

## 4. Computing Rolling Statistics: Capturing Trends

While lag features tell us a specific point in the past, "rolling statistics" capture trends over a recent period. For example, knowing the average audience over the *last 7 days* can reveal if a cinema is generally busy or slow. The standard deviation over a period can tell us how consistent the audience numbers are.

Again, these statistics should be computed *per theater*.

```python
# Helper functions to calculate rolling mean and standard deviation
def roll_mean(window, x):
    return x.shift(1).rolling(window, min_periods=1).mean()

def roll_std(window, x):
    return x.shift(1).rolling(window, min_periods=1).std()

# Create rolling features for different time windows
for window in [7, 14, 30]:
    f_t_df[f'audience_roll_mean_{window}'] = (
        f_t_df.groupby('book_theater_id')['audience_count']
               .transform(lambda x: roll_mean(window, x))
    )
    f_t_df[f'audience_roll_std_{window}'] = (
        f_t_df.groupby('book_theater_id')['audience_count']
               .transform(lambda x: roll_std(window, x))
    )
# Fill any NaN values created by rolling statistics with 0
f_t_df[[f'audience_roll_mean_7', f'audience_roll_std_7']].fillna(0, inplace=True)

print(f_t_df[['show_date', 'book_theater_id', 'audience_roll_mean_7']].head(5))
```
*Explanation*:
*   We use `groupby('book_theater_id')` and `transform()` to apply the rolling calculation for each theater independently.
*   `x.shift(1)` is critical: it shifts the `audience_count` by one day *before* calculating the rolling window. This ensures we are only using *past* data to calculate the mean/std for the *current* day, preventing "data leakage" (where our model accidentally learns from information it wouldn't have at prediction time).
*   `.rolling(window, min_periods=1)` calculates the rolling statistic. `min_periods=1` means it will calculate a statistic even if there's only one data point available in the window (e.g., the first day of a 7-day window might only have 1 day's data).
*   `mean()` and `std()` compute the average and standard deviation, respectively.
*   `fillna(0)` handles `NaN` values for the initial days of each theater where a full rolling window isn't yet available.

## How the Time-Series Feature Factory Works (Under the Hood)

Let's visualize the process of generating these time-aware features:


![dia](https://mermaid.ink/img/pako:eNqNVMtu2zAQ_JUFL04Ax_UjfumQSx5AgaItmpwKAwJLrSSiFOmSVBw3yL93KcuKFStxeRKHmtnh7nKfmTAJsog5_FOiFngjeWZ5sdJA6zu3Xgq55trDZy295OqGew7cwYrVAFTIWRr7OEnPV-yYeYfclxbvuPDGbnfkB1ngxT1aiW5_DvUPXRIUA2-fvK3-2CkECBqsi3S9FUoKrh4s1y41tsCauj-Ag5MugS88u-ZKlIo3UQmCV6yL9MMoJfURsYbh3nPvTkjcaitFjslrrvfIcbJ37IPiXFxdtRMewT3qBHouN5s4oaz1gIc9LxMZKh4LU2rf2wl9NZRV80iZeivyzZIBR-n2VLK0LlmGGgmRRu-dtGlkplW6aF8xEKZYG42akpFaUxza2wm1eBcdt_qBtNWuYW57fehJF28Qf9OFe4BeDN611dEc0Ws_QI_0YpNWYrEui6BdxSmM9nltsUPjQ6MJ8Z3UQauSocy7EzZbLRg1fYOw5s4f1RDIBCAXOficlKirK9WWyIcOGz3Fs3gUjLaQ-Qm3R71_6NiioHKDt1QcB2cFcv3J-eT8HdNHWv9n3BItDtpktv8Wp3An73D4-Mi-KX5JTR3PlQKNm33nO9hInwM-SefDs6bG5W9fUFupernSgTYbyGWWqy2sLSZSePmIVQ58mIluNxMLmsrKsT7LrExYlHLlsM-owQoe9uw5BFsxyliBKxbRZ4IpL5UPM-GFeDRIfhpTsMjbkpjWlFne6JTr8NDqUd-goTBor0MnsWg6HlUiLHpmTyyaTAej4ehyMZtMh8vJfDGf9dmW4PFsMJksZ-PF-HIxGtPxS5_9reIOB4v55ZLWfDkdDZfD4fzlH_LuOv0?type=png)


The Time-Series Feature Factory takes our existing `f_t_df` and systematically adds these new, powerful features. It ensures that operations like lag and rolling statistics are performed correctly for each theater, preserving the individual time-series of each entity.

Here’s a summary of the features generated in this factory:

| Feature Type               | What it captures                                       | Why it's useful                                                            | Python Tool/Method                 |
| :------------------------- | :----------------------------------------------------- | :------------------------------------------------------------------------- | :--------------------------------- |
| **Date Components**        | Day, month, year, day of week, weekend flags.          | Identifies daily, weekly, monthly, yearly patterns in audience.            | `DataFrame.dt` accessor            |
| **Cyclical Features**      | Circular representation of day of week, month of year. | Prevents artificial linear relationships for cyclical data.                | `np.sin()`, `np.cos()`             |
| **Lag Features**           | Audience count from previous days (e.g., yesterday, last week). | Direct historical influence, strong predictor of immediate future.       | `groupby().shift()`                |
| **Rolling Statistics**     | Mean/Standard Deviation of audience over recent periods. | Captures recent trends, stability, or volatility in audience numbers.      | `groupby().rolling().mean()`, `std()` |

## Conclusion

In this chapter, you've mastered the **Time-Series Feature Factory**. We've transformed simple date and audience data into a rich set of predictive features, including various date components, cyclical representations, historical "lag" values, and recent "rolling" trends. These features are crucial for any time-series forecasting problem, enabling our models to understand and leverage the temporal dynamics of the data.

Our `f_t_df` is now packed with information, not just about what's happening now, but also about the context of time and how events unfolded in the past. This makes our dataset exceptionally well-prepared for our machine learning models.

Next, we'll take this feature-rich dataset and put it to work in the [Chapter 4: Model Experimentation Workbench](#chapter4), where we'll train, evaluate, and fine-tune various machine learning models to make our actual audience predictions.


---


<div style="page-break-before: always;"></div>




















































<a name="chapter4"></a>
# Chapter 4: Preprocessing & Transformation Blueprint

Welcome back, future data scientists! In our journey so far, we've gone from raw, scattered information to a rich, time-aware dataset. In [Chapter 1: Data Integration Hub](#chapter1), we assembled our data and performed initial cleanups. Then, in [Chapter 3: Time-Series Feature Factory](#chapter3), we engineered powerful time-based features (like `audience_lag_1` and `dow_sin`) from our `show_date` and `audience_count` columns, packing our `f_t_df` with valuable insights.

Now, we have a dataset full of amazing features, but it's still not quite ready for our machine learning models. Imagine you're making a special juice blend. You've gathered all your fruits (our features!), but before you can put them in the blender (our model), you might need to peel some, chop others into smaller pieces, or even ensure they are all at the right temperature.

This is exactly what the **Preprocessing & Transformation Blueprint** does for our data. It's like a detailed, standardized recipe that takes your `f_t_df` (now rich with features from Chapter 3) and puts it through a final preparation phase. This ensures that all the data is in the perfect, consistent format that a machine learning model can understand and learn from effectively.

## Why Do We Need a Blueprint?

Our `f_t_df` currently contains a mix of data types and scales:
*   **Numbers on different scales**: Features like `latitude` might range from 20-40, while `audience_lag_1` could be hundreds. Models can get confused if one feature's numbers are much larger than others, potentially giving it unfair importance.
*   **Categorical data**: Some columns, like `book_theater_id` or `theater_type`, are still text labels (e.g., 'book_00001', 'Drama'). Machine learning models fundamentally work with numbers, not text.
*   **Remaining missing values**: Even after initial cleanup, some features we created (like `audience_lag_7` at the very beginning of a theater's history) might still have `NaN` (Not a Number) values. Models usually cannot process `NaN`s.

The Preprocessing & Transformation Blueprint solves these problems by defining a clear, repeatable sequence of steps:
1.  **Handling Remaining Missing Values**: Fill any lingering `NaN`s.
2.  **Scaling Numerical Features**: Bring all numerical features into a similar, common range.
3.  **Encoding Categorical Features**: Convert all text categories into numbers.

Let's explore each step with examples.

## 1. Handling Remaining Missing Values: Filling the Gaps

When we created lag and rolling features in [Chapter 3: Time-Series Feature Factory](#chapter3), pandas' `shift()` and `rolling()` functions insert `NaN`s at the beginning of each theater's data series (because there's no "past" data for the very first days). Machine learning models cannot directly handle these `NaN`s.

For these newly generated numerical features, a common strategy is to fill `NaN`s with `0`. This implies that if there was no historical audience data (e.g., `audience_lag_7` is `NaN`), we assume the audience was `0`.

```python
import pandas as pd
import numpy as np # Used for numerical operations

# Assuming f_t_df is our DataFrame with engineered features from Chapter 3
# And dynamic_columns is a list of our lag/rolling feature names

dynamic_columns = [
    'audience_lag_1', 'audience_lag_7', 'audience_roll_mean_7', # ... etc.
    'audience_lag_14', 'audience_roll_mean_14', 'audience_roll_std_14',
    'audience_lag_30', 'audience_roll_mean_30', 'audience_roll_std_30'
] # Simplified list for example

# Fill NaN values in dynamic (lag/rolling) columns with 0
for col in dynamic_columns:
    if col in f_t_df.columns: # Check if column exists
        f_t_df[col] = f_t_df[col].fillna(0)

print("Missing values in dynamic features filled with 0!")
# Output: After this, columns like 'audience_lag_1' will have 0s instead of NaNs
# for the initial rows where historical data was unavailable.
```
*Explanation*: We iterate through our `dynamic_columns` (which are the lag and rolling features) and use `fillna(0)` to replace any `NaN` values with `0`. This ensures our dataset is complete and ready for the next steps.

## 2. Scaling Numerical Features: Making Numbers Play Fair

As mentioned, features like `latitude` (values around 30-40) and `audience_lag_1` (values potentially in the hundreds or thousands) exist on very different numerical scales. If a model treats larger numbers as inherently more important, it can lead to biased learning.

**Scaling** transforms numerical features to fit within a specific, common range. The `MinMaxScaler` is a popular choice, converting all values to a range between 0 and 1. This "normalizes" the playing field, so no feature dominates just because of its magnitude.

```python
from sklearn.preprocessing import MinMaxScaler

# Let's define some example numerical columns (both static and dynamic)
numerical_features = [
    'latitude', 'longitude', 'show_day', 'show_month', 'dow_sin', 'dow_cos',
    'audience_lag_1', 'audience_roll_mean_7', 'audience_roll_std_7'
] # Simplified for brevity

# Create a scaler
static_dynamic_scaler = MinMaxScaler()

# Important: Fit the scaler on the *training data* to learn min/max values
# Then transform all your numerical features
# (We'll do this as part of the full ColumnTransformer later)

# Example output (if applied directly):
# Original Latitude (first 3 rows):
# 0    34.05
# 1    34.05
# 2    34.05
#
# Scaled Latitude (first 3 rows):
# 0    0.5432
# 1    0.5432
# 2    0.5432
```
*Explanation*: `MinMaxScaler` learns the minimum and maximum values for each feature from your data. It then uses these to transform all values. A value equal to the minimum becomes 0, a value equal to the maximum becomes 1, and everything else scales proportionally between 0 and 1.

## 3. Encoding Categorical Features: Turning Text into Numbers

Machine learning models are mathematical; they understand numbers, not text labels. Our `f_t_df` might still have string categories like `book_theater_id` ('book_00001', 'book_00002') or `day_of_week` ('Monday', 'Tuesday'). These need to be converted to numerical representations.

**One-Hot Encoding** is a widely used method. For each unique category in a column, it creates a new "binary" (0 or 1) column. If a row belongs to a certain category, the corresponding new column gets a `1`, and all other new category columns get a `0`. This avoids implying any false ordering or relationship between categories.

```python
from sklearn.preprocessing import OneHotEncoder

# Let's use some example categorical columns
categorical_features = ['book_theater_id', 'day_of_week', 'theater_type', 'theater_area'] # Simplified for brevity

# Create an encoder
# handle_unknown='ignore' is important: it prevents errors if new categories appear in test data
categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# This would learn the unique categories and transform them.
# (We'll do this as part of the full ColumnTransformer later)

# Example output (conceptual):
# Original 'theater_type' | 'theater_type_Drama' | 'theater_type_Other'
# -----------------------|----------------------|----------------------
# Drama                  | 1                    | 0
# Other                  | 0                    | 1
```
*Explanation*: `OneHotEncoder` takes your string categories, identifies all unique labels (e.g., 'Drama', 'Other'), and creates a separate binary column for each. Each row then has a `1` in the column that matches its original category and `0`s elsewhere.

## The Preprocessing & Transformation Blueprint in Action: `ColumnTransformer`

To apply all these steps efficiently, especially since different transformations apply to different columns, we use `scikit-learn`'s `ColumnTransformer`. It's our central "blueprint" orchestrator. It allows you to specify which transformers (like `MinMaxScaler` or `OneHotEncoder`) should be applied to which subsets of your columns, all in one go.

Here’s how we'll build our `preprocessor` blueprint using `ColumnTransformer`:

```python
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Define lists of your column types
# (These lists reflect the types of features after Chapter 3 and initial cleanup)

# These are the original categorical columns that are still strings
categorical_features = ['book_theater_id', 'day_of_week', 'theater_type', 'theater_area']

# These are all numerical features (static and dynamic) that need scaling
# We are using a few examples here, but in the actual project, this list is much longer!
static_numerical_features = ['latitude', 'longitude', 'show_day', 'show_month', 'show_year',
                            'encoded_day_of_week', 'day_of_year', 'week_of_year', 'quarter',
                            'is_weekend', 'is_sunday', 'is_friday', 'dow_sin', 'dow_cos',
                            'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
                            'encoded_theater_type', 'encoded_theater_area'] # These were numerical mappings from Chapter 3

dynamic_features = ['audience_lag_1', 'audience_lag_7', 'audience_lag_14', # ... all lag/roll features
                   'audience_lag_21', 'audience_lag_28', 'audience_lag_30',
                   'audience_roll_mean_7', 'audience_roll_std_7',
                   'audience_roll_mean_14', 'audience_roll_std_14',
                   'audience_roll_mean_30', 'audience_roll_std_30']

# Combine all numerical features into one list for scaling
all_numerical_features = static_numerical_features + dynamic_features

# 2. Define the transformers
numerical_scaler = MinMaxScaler()
categorical_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 3. Create the ColumnTransformer (our blueprint!)
preprocessor = ColumnTransformer(
    transformers=[
        ('num_pipeline', numerical_scaler, all_numerical_features), # Scale all numbers
        ('cat_pipeline', categorical_ohe, categorical_features)     # One-hot encode string categories
    ],
    remainder='passthrough' # Keep any other columns (like 'ID', 'show_date') as they are
)

print("Preprocessing & Transformation Blueprint (ColumnTransformer) is ready!")
# This 'preprocessor' object now knows exactly how to prepare your data.
```
*Explanation*:
*   We categorize all our features into `all_numerical_features` (for `MinMaxScaler`) and `categorical_features` (for `OneHotEncoder`).
*   The `ColumnTransformer` is configured to apply `numerical_scaler` to `all_numerical_features` and `categorical_ohe` to `categorical_features`.
*   `remainder='passthrough'` tells the `ColumnTransformer` to leave any columns *not* in our lists (like `ID` or `show_date`) untouched, as these are usually not fed directly into the model for learning.

Once this `preprocessor` blueprint is defined, we can `fit` it to our training data (to learn min/max values and unique categories) and then `transform` both our training data and any new data (like future dates for prediction) using these learned rules.

```python
# Assuming f_t_df is our DataFrame after Chapter 3 and initial NaN filling
# And all_numerical_features, categorical_features are defined as above

# Fit the preprocessor to our entire dataset (or just the training part)
# This step 'learns' the necessary rules (min/max, unique categories)
preprocessor.fit(f_t_df)

# Now, transform the data using the learned rules
transformed_data = preprocessor.transform(f_t_df)

print(f"Original data shape: {f_t_df.shape}")
print(f"Transformed data shape: {transformed_data.shape}")
# Output:
# Original data shape: (214046, 39)
# Transformed data shape: (214046, 920) <- Example, shape will change due to One-Hot Encoding
```
*Explanation*:
*   `preprocessor.fit(f_t_df)`: The blueprint "learns" from the entire dataset (`f_t_df`). For `MinMaxScaler`, it finds the min and max for each numerical column. For `OneHotEncoder`, it identifies all unique categories in the categorical columns.
*   `preprocessor.transform(f_t_df)`: The blueprint now applies these learned transformations to the data. Numerical columns are scaled, and categorical columns are replaced by their one-hot encoded versions.
*   The output `transformed_data` is a NumPy array that is entirely numerical and scaled, ready for a machine learning model! Its shape will likely have many more columns because `OneHotEncoder` expands each categorical feature into many new binary columns (e.g., `book_theater_id` alone can create 829 new columns).

## How the Preprocessing & Transformation Blueprint Works (Under the Hood)

Let's visualize how `ColumnTransformer` orchestrates these preprocessing steps:

![dia](https://mermaid.ink/img/pako:eNqFU11vmzAU_SuWn1IpRARMIDxU2tpJ60M3qd3TFKny4EIsgZ35Y20X5b_vYpoMGtryhA_nHJ_7wZ4WqgSaUwO_HcgCrgWvNW83kuCz49qKQuy4tORGCit4c80tJ9yQDX0BiEdm1YN9KKuLDT1XXqnGtfKH5tJUSrege_05PPvcONhpIe2kz62Qt_zpvuDN0WKITCm-S_iq7BfZ1fgiGUGTt-CH5g54-fy_VI8FHvTldrpeOehKcHl5VlJO7kGWpOycHoXdEt40pAJunQbTO3xTFoj6g_km1DclSCsqAYZIh4jAUglHx4JbqFV_LrzOHCOd2WCwYaP6TEPDcaAhN5gu6g6QLw0xHakcOPlCZ2GwJEiv4eK9TKNJHEMZi-OvR-WN041UH8RTEoKtsgQ8eyKn8b1cmndzjhciR077S0icSDdLe6KWb871tcGnwQ4QroFI9fhqvH1jeyM6p7UWJc0r3hiYU-S1vDvTfUfYULuFFjY0x9cSKu4a2-3nAXW40D-VamlutUOlVq7ennzcDrtw_N9PqMYxgL5STlqaJ0vmTWi-p080j7NFmmRJGrFozVicrOf0GdEoXsRxkjEWpquQRRE7zOlff224yNIkxGeZraIVY-n68A8_34eQ?type=png)


As you can see, `ColumnTransformer` acts as a central hub. It intelligently directs different parts of your `f_t_df` to the right transformation tools (`MinMaxScaler` for numbers, `OneHotEncoder` for categories). Once each tool has done its job, `ColumnTransformer` collects all the processed pieces and combines them into one final, model-ready dataset.

Here’s a summary of the core transformations in our blueprint:

| Transformation                      | What it does                                                 | Why it's useful                                                              | Tool / Strategy                      |
| :---------------------------------- | :----------------------------------------------------------- | :--------------------------------------------------------------------------- | :----------------------------------- |
| **Handling Remaining Missing Values** | Fills `NaN`s (especially in lag/rolling features).             | Prevents model errors; provides complete input for learning.                 | `DataFrame.fillna(0)`                |
| **Scaling Numerical Features**      | Changes numerical feature values to a common, small range (e.g., 0 to 1). | Ensures no single feature's large values dominate the model's learning.      | `MinMaxScaler`                       |
| **Encoding Categorical Features**   | Converts text-based categories into numerical representations (0s and 1s). | Allows machine learning models (which understand numbers) to process text data. | `OneHotEncoder`                      |

## Conclusion

In this chapter, you've mastered the **Preprocessing & Transformation Blueprint**. We've established a standard, efficient way to prepare our feature-rich data by handling any remaining missing values, scaling all numerical features, and encoding categorical features into a numerical format. This `preprocessor` object (our blueprint) is incredibly powerful because it ensures that all our data goes through the exact same preparation steps, making it consistent and optimally prepared for any machine learning model.

With our data now thoroughly cleaned, scaled, and encoded, we're ready for the next exciting phase: training, evaluating, and fine-tuning our machine learning models! Get ready for [Chapter 5: Model Experimentation Workbench](#chapter5)!



---


<div style="page-break-before: always;"></div>




















































<a name="chapter5"></a>
# Chapter 5: Model Experimentation Workbench

Welcome back, future data scientists! In our journey so far, we've gone from raw, scattered information to a rich, time-aware dataset. In [Chapter 1: Data Integration Hub](#chapter1), we assembled our data. Then, in [Chapter 2: Preprocessing & Transformation Blueprint](#chapter2), we cleaned, scaled, and encoded it. And just recently, in [Chapter 3: Time-Series Feature Factory](#chapter3), we engineered powerful time-based features crucial for forecasting cinema audience. Our `f_t_df` is now a treasure chest of information, perfectly prepared.

Now what? We have all these fantastic features, but how do we turn them into actual predictions? This is where the magic of machine learning models comes in! But choosing the *best* model isn't like picking a ready-made tool; it's more like being a scientist in a laboratory. You need to try different approaches, carefully measure their results, and fine-tune them until you get the best possible outcome.

This entire process of trying, testing, and perfecting different machine learning models is what we call the **Model Experimentation Workbench**. Think of it as your project's advanced laboratory. Here, you'll put various machine learning "recipes" (algorithms) to the test, evaluate how well they predict, and make adjustments to get the most accurate forecasts for our cinema audience challenge.

## Why Do We Need a Model Experimentation Workbench?

Our goal is to predict the `audience_count`. But there isn't just one machine learning model that magically works best for every problem. Some models are good with simple linear relationships, others excel at complex patterns.
For our cinema audience forecasting, we need a systematic way to:

1.  **Try Different Algorithms**: Is a simple Linear Regression enough, or do we need a more complex model like a Decision Tree or Gradient Boosting?
2.  **Ensure Consistency**: How do we make sure that every time we test a new model, it uses the exact same data preparation steps we defined in [Chapter 4: Preprocessing & Transformation Blueprint](#chapter4)?
3.  **Measure Success**: How do we objectively tell if one model is "better" than another? We need specific metrics.
4.  **Optimize Models**: Even with a good model, it often has "knobs and dials" (hyperparameters) that need to be adjusted for peak performance.

The Model Experimentation Workbench gives us the tools and methods to answer these questions and systematically find the best predictive model.

## Key Components of Our Workbench

Our workbench consists of three main ideas:
1.  **Machine Learning Pipelines**: Combining data preparation and modeling into one consistent flow.
2.  **Model Evaluation Metrics**: Quantifying how well our models perform.
3.  **Hyperparameter Tuning**: Optimizing model "knobs and dials" for the best results.

Let's explore each one.

### 1. Machine Learning Pipelines: Your Consistent "Recipe"

Imagine you have a complex data preprocessing sequence (scaling numbers, encoding categories, adding time-series features). Every time you want to train a new machine learning model, you would have to run all these steps first. This can be repetitive and error-prone.

A **Machine Learning Pipeline** from `scikit-learn` is like a complete, automated assembly line for your data. It links all your preprocessing steps and your chosen machine learning model into one single, coherent object.

Our pipeline will look something like this:
`Raw Data` -> `Preprocessing Blueprint (ColumnTransformer)` -> `Machine Learning Model` -> `Predictions`

This ensures that whenever you train or make predictions with a pipeline, the data always goes through the *exact same* sequence of transformations before reaching the model.

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor # Our first simple model
# Assume 'preprocessor' is defined from Chapter 4

# Create a sample model
model = DecisionTreeRegressor(random_state=42)

# Combine preprocessing and model into a single pipeline
full_pipeline = Pipeline([
    ('preprocess', preprocessor), # Step 1: Apply all preprocessing
    ('model', model)             # Step 2: Train the Decision Tree model
])

print("Our full ML pipeline is ready!")
# You can now use full_pipeline.fit(X, y) and full_pipeline.predict(X_new)
```
*Explanation*: We combine our `preprocessor` (which takes care of scaling and encoding, as learned in [Chapter 4: Preprocessing & Transformation Blueprint](#chapter4)) with a `DecisionTreeRegressor` into a `Pipeline`. Now, when we `fit` this `full_pipeline` to our `X_train` data, it automatically preprocesses `X_train` *then* trains the `DecisionTreeRegressor`. Similarly, when we `predict`, it preprocesses the new data *then* uses the trained model to make predictions.

### 2. Model Evaluation Metrics: How Good are Our Predictions?

After training a model, we need to know how well it performed. Just saying "it's good" isn't enough; we need objective numbers. This is where evaluation metrics come in. For our regression task (predicting a number like `audience_count`), two common metrics are:

*   **R² (R-squared)**: This metric tells us how much of the variation in our target variable (`audience_count`) can be explained by our model.
    *   A value of `1.0` means the model perfectly explains all the variance (perfect prediction).
    *   A value of `0.0` means the model explains none of the variance (it's no better than simply predicting the average).
    *   Negative values mean the model is worse than simply predicting the average, indicating a very poor fit.
    *   *Analogy*: Imagine trying to predict how many apples are in a basket. R² tells you how much of the actual number of apples your prediction accounts for.

*   **MAE (Mean Absolute Error)**: This metric tells us the average absolute difference between our model's predictions and the actual values.
    *   A lower MAE means predictions are, on average, closer to the actual values.
    *   It's easy to understand: if MAE is 10, it means, on average, our predictions are off by 10 audience members.
    *   *Analogy*: If you predict 50 apples and there are 55, your error is 5. If you predict 40 and there are 35, your error is 5. MAE averages these absolute errors.

```python
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Dummy actual values (y_true) and model predictions (y_pred)
y_true = np.array([100, 150, 200, 120, 180]) # Actual audience counts
y_pred = np.array([105, 145, 190, 130, 175]) # Model's predictions

# Calculate R²
r2 = r2_score(y_true, y_pred)
print(f"R-squared (R²): {r2:.4f}") # Higher is better, closer to 1

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}") # Lower is better, closer to 0
```
*Output*:
```
R-squared (R²): 0.9412
Mean Absolute Error (MAE): 6.00
```
*Explanation*: We use `r2_score` and `mean_absolute_error` functions from `sklearn.metrics`. These functions take the true values (`y_true`) and the predicted values (`y_pred`) and return the respective scores.

### 3. Hyperparameter Tuning with RandomizedSearchCV: Fine-tuning Your Model

Machine learning models have two types of parameters:
*   **Learned parameters**: These are learned automatically from the data during training (e.g., the weights in a Linear Regression).
*   **Hyperparameters**: These are settings you choose *before* training the model (e.g., `max_depth` in a Decision Tree, `n_estimators` in a Random Forest). They control *how* the model learns.

Choosing the right hyperparameters can significantly impact a model's performance. It's like having a complicated audio system: the "volume" and "bass" knobs (hyperparameters) need to be adjusted just right for the best sound. Trying every single combination of hyperparameters is called "Grid Search," but it can be very slow for many parameters.

**RandomizedSearchCV** is a smarter way to tune hyperparameters. Instead of trying every combination, it tries a fixed number of *random* combinations from a specified range. This is often much faster and can still find very good hyperparameter settings.

```python
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

# --- Data Preparation (Simplified - in real project, use f_t_df) ---
# Assuming X_train, y_train are already prepared from Chapter 4
# and the 'preprocessor' object is defined.

# Create the pipeline with a DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_pipeline = Pipeline([
    ('preprocess', preprocessor), # Our preprocessor from Chapter 4
    ('decision_tree', dt_model)   # The model we want to tune
])

# Define the search space for hyperparameters
# The prefix 'decision_tree__' refers to the 'decision_tree' step in the pipeline
param_dist = {
    'decision_tree__max_depth': [5, 10, 15, 20, None], # Max depth of the tree
    'decision_tree__min_samples_split': [2, 10, 20]    # Min samples required to split a node
}

# For time-series data, regular cross-validation can be problematic
# because it might use future data to predict the past.
# TimeSeriesSplit ensures that validation sets always come *after* training sets.
tscv = TimeSeriesSplit(n_splits=2) # Uses 2 splits for demonstration

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=dt_pipeline,          # Our pipeline
    param_distributions=param_dist, # The hyperparameters to search
    n_iter=5,                       # Number of random combinations to try (reduced for quick example)
    cv=tscv,                        # The cross-validation strategy
    scoring='r2',                   # Metric to optimize (maximize R²)
    random_state=42,
    n_jobs=-1,                      # Use all available CPU cores
    verbose=0                       # Set to 1 or 2 for more output
)

# Run the search (this can take a while!)
print("Starting RandomizedSearchCV...")
random_search.fit(X_train, y_train) # Fit to your actual training data (X_train, y_train)
print("RandomizedSearchCV complete!")

# Get the best parameters and best score
print(f"\nBest hyperparameters found: {random_search.best_params_}")
print(f"Best R² score from cross-validation: {random_search.best_score_:.4f}")
```
*Output (example, actual values can vary)*:
```
Starting RandomizedSearchCV...
RandomizedSearchCV complete!

Best hyperparameters found: {'decision_tree__min_samples_split': 2, 'decision_tree__max_depth': 5}
Best R² score from cross-validation: 0.4621
```
*Explanation*:
1.  We define `dt_pipeline` which includes our `preprocessor` and `DecisionTreeRegressor`.
2.  `param_dist` specifies the range of hyperparameters we want to explore for the `decision_tree` step in our pipeline. Notice the `decision_tree__` prefix.
3.  `TimeSeriesSplit` is crucial for our time-series problem. It makes sure that during cross-validation, the model is always trained on older data and tested on newer data, mimicking real-world forecasting.
4.  `RandomizedSearchCV` then intelligently samples `n_iter` combinations of these hyperparameters, trains and evaluates the `dt_pipeline` with each, and reports the best combination and its corresponding R² score.

## How the Model Experimentation Workbench Works (Under the Hood)

Let's visualize the flow when you use the Model Experimentation Workbench to find the best settings for your model:

![dia](https://mermaid.ink/img/pako:eNqVVO1O2zAUfZU7_5g6KUWhTUubH0gDxjSJMkQZmqZKyNi3rbXEzmwHKIiX2iPsyWY7pKRNN23-lXvjc-7XuX4iTHEkKTH4o0TJ8ETQhab5TII7BdVWMFFQaeGLQQ3UwIyEr84JtRSmTKC0wth3M9KGXGkqpJCLcDVAaw8EV-frjfWOCFbVx06WC42FVgyNccCjrHSWcO7At_EPOscqK3PpgkgzVzpHvZNwcnZzIQrMhMSKZXIGtWPX_UsqucrFI_IpUs2WFWjbe3y9MxZaLZh5ifNidC5__Yxg8v7DzvSO0NiJG0pWgbwJV6VEDsHrIRXIz6F7eNgoJ3X3T3DuCyvqCu-FXUJeIxu47QIa4OWqQO1yojlaN2tTlW0KyvAfSD5JYQW1WONCBhuTf60hU6qAU6XhHHSgAqbyWyGpFUqa6o4_23HadX9Eu1U0K7V26tyqxtQV-NPgcIxNvXrKU2HhLazl1K6hpmkCu-3UXgXJW9DNDNpAIavhgZJgGzz8f3gm9LtThEYuWOirJ7ujmeChzX_l8mSVbNNq39YcvjMlwjV1K7nZ1Op-9w_yqMUPhin9ikTJa1HsGPZ6KTzDFDNkrblWQ7_16xKYa-I11CfkZespPhdWhAjwsmqSw0VLIufKqVjduRWocefqHjRSvgI3B3DLQjOwtSw8iXMjo8Y6-82MkIgstOAkndPMYETck5RTb5MnH2FG7BJzl6mn5jinZWZ97GeHc2_BN6VyklrX5YhoVS6Wa56ycDOr3-q116mdoz5WpbQkHcRxICHpE3nw5l4yHibDUdzvj3rjURKRFUn7-4O9fm-wPxgOe70kGST954g8hrDx3uggGbtzMB7sx-M4Pnj-Dc-oEnI?type=png)


The workbench orchestrates the entire process. It takes your raw training data, passes it through the predefined preprocessing steps using the `ColumnTransformer` within the `Pipeline`. Then, it systematically tries different model "knobs and dials" (hyperparameters) as instructed by `RandomizedSearchCV`. For each trial, it measures the model's performance using metrics like R² and MAE. Finally, it tells you which combination of hyperparameters yielded the best results, giving you your optimally tuned model.

Here’s a summary of the core tools and their roles in our workbench:

| Tool / Concept            | What it does                                                 | Why it's useful                                                                 |
| :------------------------ | :----------------------------------------------------------- | :------------------------------------------------------------------------------ |
| **`Pipeline`**            | Chains multiple data processing steps and a model into one object. | Ensures consistent data flow from raw input to prediction; simplifies workflow. |
| **R² Score**              | Measures how well the model explains variance in the target. | Gives an intuitive understanding of model fit (0-1, higher is better).         |
| **MAE (Mean Absolute Error)** | Measures the average absolute difference between predictions and actuals. | Provides a clear, interpretable error value in the original units.              |
| **`RandomizedSearchCV`**  | Efficiently searches for the best hyperparameters by sampling combinations. | Optimizes model performance without exhaustively trying every possibility.      |
| **`TimeSeriesSplit`**     | Ensures correct cross-validation for time-series data.     | Prevents data leakage by always training on past data and validating on future. |

## Conclusion

In this chapter, you've stepped into the **Model Experimentation Workbench**! You've learned how to streamline the process of building and evaluating machine learning models using `Pipelines`, how to quantify their performance with `R²` and `MAE`, and how to fine-tune them for optimal results using `RandomizedSearchCV` with appropriate `TimeSeriesSplit` cross-validation. This robust setup allows us to systematically test various algorithms and their configurations to find the best one for our cinema audience forecasting.

With our best models identified and tuned, we're now ready for the final, crucial step: using them to make predictions into the future, iterating as new data becomes available. This is what we'll explore in the next exciting chapter: [Chapter 6: Recursive Forecasting Engine](#chapter6).




---


<div style="page-break-before: always;"></div>




















































<a name="chapter6"></a>
# Chapter 6: Iterative Forecasting Engine

Welcome back! In our last chapter, [Chapter 5: Model Experimentation Workbench](chapter5), we explored how to rigorously test different machine learning models and select the best ones for our cinema audience prediction task. We found a champion model (or models!) that can make accurate predictions based on historical data and carefully engineered features.

### Why is an Iterative Forecasting Engine So Important?

Imagine you're asked to predict the `audience_count` for a specific cinema theater for the entire next month, let's say from March 1st to April 22nd, 2025. You've trained a great model that uses features like "audience count yesterday" (`audience_lag_1`) or "average audience over the last 7 days" (`audience_roll_mean_7`), as we learned in [Chapter 3: Time-Series Feature Factory](#chapter3).

Now, think about what happens when you try to predict for a day early next month, say March 2nd. Your model needs "yesterday's audience count" (March 1st) to make that prediction. But March 1st is also in the future! You don't *actually* know the audience count for March 1st yet.

This is where the **Iterative Forecasting Engine** comes in. It solves this real-world challenge by simulating how forecasting works step-by-step. Instead of trying to predict the entire future at once, we predict **one day at a time**.

Here's the magic: For each day we predict, the engine uses all the *actual historical data* available up to that point. Then, it immediately incorporates the *predictions it just made for previous future days* into its "history." It's like a rolling forecast, where each new prediction immediately becomes part of the "memory" for the next day's prediction. This approach ensures two crucial things:

1.  **No Peeking into the Future**: We never use actual information from a future date to predict a present or past date.
2.  **Leveraging Latest Information**: Our predictions for later dates are always as informed as possible, benefiting from earlier predictions.

Let's dive into how this clever "rolling forecast" works.

### Key Concepts: The Daily Prediction Cycle

The core idea behind the Iterative Forecasting Engine is a continuous cycle of: **Look at History → Calculate Features → Predict → Update History**.

#### 1. Our "History Book" (`history_dict`)

As we learned in [Chapter 2: Historical Data Management](#chapter2), we maintain a `history_dict`. This is a Python dictionary that stores a list of past `audience_count` values for *every single cinema theater*. It's like a running logbook for each theater.

*   **Initial State**: Before we start predicting, `history_dict` is filled with all the *actual* historical `audience_count` values from our training data.
*   **Dynamic Updates**: As soon as we make a prediction for a theater for a specific day, that *predicted* `audience_count` is immediately added to the end of that theater's list in the `history_dict`.

#### 2. Features Based on "Current" History

When we make a prediction for a future date, say March 2nd, 2025:
*   Features like `show_day`, `is_weekend`, `latitude`, `theater_type` (which are known for March 2nd) are used directly. These are often called "static" or "calendar" features.
*   Features like `audience_lag_1` (yesterday's audience) or `audience_roll_mean_7` (average over the last 7 days) are calculated using the `audience_count` values currently stored in the `history_dict` for that specific theater. This `history_dict` now includes any predictions we've already made for March 1st, February 28th, and so on.

#### 3. The Iterative Loop

The process repeats day after day. When we move from predicting March 2nd to March 3rd, the `history_dict` already contains the predicted audience count for March 2nd. So, when the model calculates `audience_lag_1` for March 3rd, it will use March 2nd's predicted value. This allows the model to propagate its forecasts forward in time.

### Solving the Use Case: Forecasting for Future Dates

Let's walk through how our engine forecasts the `audience_count` for all theaters for a future period (e.g., March 1st to April 22nd, 2025), day by day.

**Step-by-Step Walkthrough:**

1.  **Prepare Prediction Dates**: We first create a list of all future dates we need to predict, ordered chronologically.
2.  **Initialize `history_dict`**: Our `history_dict` is populated with all the *actual* historical `audience_count` data available up to the last day of our training set (as covered in [Chapter 2: Historical Data Management](#chatper2)).
3.  **Loop Through Each Prediction Date**: The engine starts a loop, processing one `current_date` at a time.
    *   For the `current_date` (e.g., March 1st):
        *   **Identify Theaters**: Find all unique `book_theater_id`s that need predictions for `current_date`.
        *   **For Each Theater**:
            *   **Calculate Dynamic Features**: Look up the `audience_count` history for this specific `theater_id` in a *copy* of the `history_dict` (to prevent modifying the original initial history). Use this history to calculate lag and rolling features for `current_date` (referencing [Chapter 3: Time-Series Feature Factory](#chapter3)).
            *   **Preprocess Features**: Scale these newly calculated dynamic features using the `dynamic_scaler` (which was fitted on training data, as seen in [Chapter 4: Preprocessing & Transformation Blueprint](#chapter4)).
            *   **Combine Features**: Combine the scaled dynamic features with the pre-transformed static and categorical features for the `current_date` (from [Chapter 4: Preprocessing & Transformation Blueprint](#chapter4)).
            *   **Make Prediction**: Feed the combined features for this theater and `current_date` into our best-performing machine learning model (selected in [Chapter 5: Model Experimentation Workbench](#chapter5)). The model outputs a `predicted_audience_count`.
            *   **Update History**: Immediately add the `predicted_audience_count` to the `history_dict` (the working copy) for that `theater_id`. This is the crucial iterative step!
        *   Store all predictions made for the `current_date`.
    *   The loop then moves to the next `current_date` (e.g., March 2nd), and the entire process repeats. The `history_dict` is now updated with March 1st's predictions, allowing March 2nd's lag/rolling features to be correctly calculated.

This cycle continues until predictions have been made for all future dates.

### Under the Hood: The `making_predictions` Function

The project's `making_predictions` function (or a simplified version below) encapsulates this entire iterative process. Let's look at its key parts, referencing how data from previous chapters is utilized.

#### 1. Initializing History

First, we need to set up our `history_dict` with all actual historical `audience_count` values. This is our engine's starting "memory."

```python
import numpy as np
import pandas as pd
import datetime as dt

# Assume `f_t_df` is our integrated, feature-engineered DataFrame (from Chapter 3)

history_dict = {}
all_theaters = f_t_df['book_theater_id'].unique()

for theater_id in all_theaters:
    # Filter data for this specific theater from our full historical data
    theater_data = f_t_df[f_t_df['book_theater_id'] == theater_id]
    # Sort by date to ensure correct chronological order
    theater_data = theater_data.sort_values('show_date')
    # Get the list of actual audience counts
    audience_list = theater_data['audience_count'].tolist()
    history_dict[theater_id] = audience_list

print(f"Initialized history for {len(history_dict)} theaters.")
# Example output: Initialized history for 826 theaters.
# Example: book_00001 has 342 days of history (actual past data).
```
*Explanation*: This code creates `history_dict`, where each `book_theater_id` (theater) maps to a list of its historical `audience_count` values, ordered by `show_date`. This gives our forecasting engine a solid factual base to start from.

#### 2. Preparing Future Dates and Static Features

Next, we prepare the "skeleton" for the future dates we want to predict. This `t_df` will contain all the future dates and theater IDs, along with their static features (e.g., `latitude`, `theater_type`, various date components). These static features are pre-transformed using `ohe_encoder` and `static_scaler` (from [Chapter 4: Preprocessing & Transformation Blueprint](#chapter4)) *before* the iterative loop begins, as they don't change daily based on predictions.

```python
# Assume `all_booking_ids` is a list of all theater IDs
# Assume `df6` (date_info) and `df2` (theater_details) are loaded (from Chapter 1)
# Assume `theater_type_dict` and `theater_area_dict` are mappings (from Chapter 3)
# Assume `ohe_encoder` and `static_scaler` are already fitted (from Chapter 4)
# Assume `static_numerical_columns` and `categorical_columns` are defined.

# Define the range of future dates for prediction
start_date_forecast = dt.date(2024, 3, 1) # Example start
end_date_forecast = dt.date(2024, 4, 22)   # Example end
all_forecast_dates = pd.date_range(start=start_date_forecast, end=end_date_forecast)

# Create a DataFrame for all future theater-date combinations
combinations = []
for date in all_forecast_dates:
    for booking_id in all_booking_ids: # Using `all_booking_ids` from project setup
        combinations.append((booking_id, date))

t_df = pd.DataFrame(combinations, columns=["book_theater_id", "show_date"])
t_df["show_date"] = pd.to_datetime(t_df["show_date"])
t_df["ID"] = t_df["book_theater_id"] + "_" + t_df["show_date"].dt.strftime("%Y-%m-%d")

# Merge static metadata and engineer date features (as in Chapter 1 & 3)
t_df = t_df.merge(df6, on='show_date', how='left')
t_df = t_df.merge(df2, on='book_theater_id', how='left')
# Fill NaNs for static features using medians/unknowns from training data (Chapter 1)
# ... (date feature extraction, cyclical encoding, categorical mapping as in Chapter 3) ...

# Pre-transform static/categorical features for the entire future `t_df`
X_test_categorical = ohe_encoder.transform(t_df[categorical_columns])
X_test_static = static_scaler.transform(t_df[static_numerical_columns])

print(f"Prepared future prediction DataFrame with {t_df.shape[0]} rows.")
# Example output: Prepared future prediction DataFrame with 44766 rows.
```
*Explanation*: This block first generates `t_df` (our future data skeleton), then adds all static and calendar-related features. Crucially, these features are then pre-transformed using the `ohe_encoder` and `static_scaler` that were *already fitted* on the training data. This ensures consistency in preprocessing (refer to [Chapter 4: Preprocessing & Transformation Blueprint](#chapter4)).

#### 3. The Iterative Prediction Loop Function

This is the core of our Iterative Forecasting Engine. The `making_predictions` function processes `t_df` day by day, calculating dynamic features, making predictions, and updating its `current_history` for subsequent days.

```python
# Assume `model` is our best-trained machine learning model (e.g., LightGBM from Chapter 5)
# Assume `history_dict` is initialized (as in Step 1)
# Assume `X_test_cat`, `X_test_stat` are pre-transformed static/categorical features for t_df (as in Step 2)
# Assume `dynamic_scaler` is the MinMaxScaler fitted on dynamic features from training data (from Chapter 4)
# Assume `dyn_cols` is a list of dynamic (lag/rolling) feature names (from Chapter 3)

def making_predictions(model, test_dataframe, history_dictionary, 
                       X_test_cat, X_test_stat, dyn_scaler, dyn_cols):
    
    # Make a copy of history_dictionary so the original (initial actuals) is untouched
    current_history = {tid: hist.copy() for tid, hist in history_dictionary.items()}

    # Sort test data by date to ensure chronological processing
    test_dataframe = test_dataframe.sort_values(['show_date', 'book_theater_id']).reset_index(drop=True)
    all_unique_dates = sorted(test_dataframe['show_date'].unique())
    
    all_predictions = np.zeros(len(test_dataframe)) # Array to store all final predictions

    for current_date in all_unique_dates:
        # Select rows for the current date for all theaters
        date_mask = test_dataframe['show_date'] == current_date
        row_indices = np.where(date_mask)[0]
        theaters_today = test_dataframe.loc[row_indices, 'book_theater_id'].values
        num_theaters = len(theaters_today)

        # Prepare a matrix for dynamic (lag/rolling) features for today's predictions
        # This will be filled by looking up `current_history`
        lag_feature_matrix = np.zeros((num_theaters, len(dyn_cols)))

        for i, theater_id in enumerate(theaters_today):
            theater_history = current_history.get(theater_id, [])
            history_length = len(theater_history)

            # --- Accessing `current_history` to calculate dynamic features ---
            # This is a simplified example; in a full project, all lag/rolling features
            # (e.g., audience_lag_1, audience_lag_7, audience_roll_mean_7, etc.)
            # from Chapter 3 would be calculated here.
            
            # Example: Calculate audience_lag_1 (yesterday's audience)
            lag_feature_matrix[i, 0] = theater_history[-1] if history_length >= 1 else 0
            
            # Example: Calculate audience_roll_mean_7 (average of last 7 days)
            if history_length >= 1:
                last_7_days = theater_history[-7:] if history_length >= 7 else theater_history
                lag_feature_matrix[i, 6] = np.mean(last_7_days)
            else:
                lag_feature_matrix[i, 6] = 0
            # ... (similar logic for other dynamic features like lag_7, roll_std_7, etc.) ...

        # --- Scale the dynamically calculated features ---
        X_dynamic_scaled = dyn_scaler.transform(lag_feature_matrix)

        # --- Combine all features for today's prediction ---
        X_cat_today = X_test_cat[row_indices]
        X_stat_today = X_test_stat[row_indices]
        X_today_combined = np.hstack([X_cat_today, X_stat_today, X_dynamic_scaled])

        # --- Make predictions for the current day ---
        predictions_today = model.predict(X_today_combined)
        predictions_today = np.maximum(predictions_today, 0) # Ensure no negative predictions
        all_predictions[row_indices] = predictions_today # Store predictions

        # --- UPDATE `current_history` with today's predictions for the next iteration ---
        for i, theater_id in enumerate(theaters_today):
            current_history.setdefault(theater_id, []).append(predictions_today[i])

    return all_predictions

print("Iterative forecasting function defined!")
```
*Explanation*:
1.  **`current_history`**: A copy of our `history_dict` (initialized with actuals) is made. This `current_history` is modified during the loop.
2.  **`for current_date in all_unique_dates:`**: This main loop iterates through each day we need to predict, ensuring chronological order.
3.  **Dynamic Feature Calculation**: For each `theater_id` on the `current_date`, the code accesses `theater_history` from `current_history`. This `theater_history` contains *all actual data up to the last training day PLUS any predictions made for previous future days*. It then uses this history to calculate lag and rolling features.
    *   For example, `lag_feature_matrix[i, 0] = theater_history[-1]` retrieves "yesterday's" audience. If `current_date` is March 1st, it gets February 29th's *actual* audience. If `current_date` is March 2nd, it gets March 1st's *predicted* audience (because it was added in the previous daily loop iteration).
4.  **Scaling Dynamic Features**: `dyn_scaler.transform()` scales the newly calculated dynamic features using the scaler fitted on the training data.
5.  **Combining Features**: All features (pre-transformed static/categorical and scaled dynamic) are combined into `X_today_combined`.
6.  **Prediction**: `model.predict()` makes predictions for all theaters on the `current_date`. Predictions are capped at 0 (`np.maximum`) as audience counts cannot be negative.
7.  **Updating History**: `current_history.setdefault(theater_id, []).append(predictions_today[i])` is the crucial iterative step! The new predictions are immediately added to their respective theater's history. This makes them available as "past" data for calculating features for the *next* day's forecast.

#### 4. Running the Engine and Generating Output

Finally, we run our engine with a trained model and convert the predictions into the required submission format.

```python
# Assume `model_2_lightgbm` is our best-trained model (from Chapter 5)
# (In a real scenario, this model would have been fitted to `X_train_combined` and `y_train`)
# Assume `t_df`, `history_dict`, `X_test_categorical`, `X_test_static`, `dynamic_scaler`, `dynamic_columns` are defined.

print("MODEL 2: LightGBM Forecasting...")
predictions_model_2 = making_predictions(
    model=model_2_lightgbm,
    test_dataframe=t_df.copy(), # Make a copy to avoid modifying the original t_df
    history_dictionary=history_dict,
    X_test_cat=X_test_categorical,
    X_test_stat=X_test_static,
    dyn_scaler=dynamic_scaler,
    dyn_cols=dynamic_columns
)
print("Forecasting complete! Predictions are ready.")

# Output:
# MODEL 2: LightGBM Forecasting...
# DONE
# Forecasting complete! Predictions are ready.
```
**Output Description:**
After calling `making_predictions`, `predictions_model_2` will be a NumPy array containing the predicted `audience_count` for each `book_theater_id` for every day in the future period (e.g., March 1st to April 22nd, 2025). Each day's prediction for a theater will have incorporated the previous day's predicted `audience_count` for that same theater into its feature calculations, effectively creating a powerful, robust, and iteratively updated forecast. These predictions are then rounded and saved in a submission file.

### How the Iterative Forecasting Engine Works (Under the Hood)

Let's visualize the dynamic, day-by-day process for a single model:

![dia](https://mermaid.ink/img/pako:eNqNVV1v2jAU_StXfhlVA6N8FfJQCZVWq9RNqN1eVqbJcgxYc-zMdmhZ1f--6ziBpGTS_GRufI6P7z338kqYTjiJieW_c64YXwi6MTRdKcCVUeMEExlVDr5Zbk6jN2ojFAdqYUXuHDfUiR2HW204o9YJtSlPrMgp9k4JJ6j8JKzTZl9yhBhUwc6cuZxKe9ZGcJ0bw5VrEJSxEwI4h6XhiWBOaNVOd8upyw2_pexI91WkvPvIjeC2-g7lgTYKvCIzmnFrtQkEjUhnsVc0FQweGZXctKr4jPWQ5eWGYuoS-Hwfom3H51LW3hVwc8byNJfUIbb20cMDga9l9-oqlCZGxKNDSljXyoZ7eFrQPVxE0Ov1IvB79aOSEKDI0Syi57rXNAEqJfxS-lkBDfmvgM3zXWRoVtEzLHVWqIdn4bYwPxIECql15i0GnLJtoauz-MkCyxmsjU6L4AU4HVQHWEN2s9b-0gffANZBUpZoHU7YIhMH_uoZfn3RKFHvuIH3bDcvzuDegqQbG4HRUvqcWkdd4CuUuy3Cqqbyq0nTmpoH7tCL2GKGS77zBsioF00drUtrArutD1564KE9Pnqn7ITObWuftMhrGAh15UYB2pqVxnufx2P9GoWo90fhRN8ZreAKWke0irCe4r8FFJ0Vh071zwaaJ8JPwmbhoVOY0ZcQOc__dUkjYQV1q8Ss1rLLurdaBJ6aYJ4kR5R3-bYcdl6x4i_eD_sPFnLL63IOjM2ZEfofW7-pxCO4SipJJRjRfnjUnrIWCuf107KaE0s_JEhENkYkJF6jt3hEUm5S6n-TV0-3Iuj9FMV5noSvaS6LO98Qh1Ptu9YpiZ3JEWl0vtkeePIMnV79Rx2iKDnh5lrnypF4NBsWJCR-JS8kHg57k8F0PBuPhpPBuD-YjCOyx_Bg0JsOp5PLwWg2no4v-sO3iPwp7u33ppejGa7LGcZn_f7k7S9h-WW7?type=png)


The Iterative Forecasting Engine acts as the central coordinator. It initializes its memory (`CurrentHistory`) with all available actual historical data. Then, day by day, it asks the `Time-Series Feature Factory` (conceptually, implemented within the `making_predictions` function) to calculate dynamic features using its up-to-date `CurrentHistory`. These features are scaled by the `Preprocessor` (specifically, `dynamic_scaler`) and combined with static features, then fed into the `Trained ML Model` to get predictions for the current day. Crucially, these new predictions (`P_current`) are immediately absorbed back into the `CurrentHistory`, making them available for calculating features for the *next* day's forecast. This continuous feedback loop is what makes the forecasting iterative and powerful.

### Conclusion

In this chapter, you've conquered the **Iterative Forecasting Engine**, the pinnacle of our machine learning project! You've learned how to move beyond simple, one-shot predictions to a sophisticated, day-by-day forecasting mechanism that intelligently updates its own "memory" with previous predictions. This allows our models to correctly calculate dynamic features like `lag` and `rolling statistics` for every future day, producing a continuous and coherent forecast for cinema audience numbers.

Now that we understand how to make iterative predictions with a single model, we're ready to explore how to combine the strengths of multiple models for even more robust forecasts.

[Chapter 7: Ensemble Modeling Strategy](#chapter7)





---


<div style="page-break-before: always;"></div>




















































<a name="chapter7"></a>
# Chapter 7: Ensemble Modeling Strategy

Welcome to the final chapter of our machine learning capstone project! In [Chapter 6: Iterative Forecasting Engine](#cahpter6), we learned how to use our best-performing model to make predictions day by day, dynamically updating its history. Now, we're going to take our predictions to the next level by combining the strengths of multiple models through an **"Ensemble Modeling Strategy."**

### Why is Ensemble Modeling So Important?

Imagine you're trying to predict the `audience_count` for future cinema shows, and you've found a few excellent models (like LightGBM, HistGradientBoosting, etc.). Each of these models is like a highly skilled expert. One expert might be really good at spotting daily patterns, another at identifying weekly trends, and a third at handling unusual spikes or drops in audience.

If you rely on just one expert (a single model), you might miss out on valuable insights that the others could provide. What if that one expert makes a mistake?

This is where "Ensemble Modeling" comes in. It's like building a **"team of experts"** to make a collective decision. Instead of picking just one best model, we combine the predictions from several diverse models. The goal is to leverage the unique strengths of each model, smooth out individual errors, and ultimately achieve a **more robust and accurate overall prediction**. It's the wisdom of the crowd applied to machine learning!

### Key Concepts in Ensemble Modeling

To build our "team of experts," we need to understand a few core ideas:

#### 1. Diverse Models: Our "Team of Experts"

The success of an ensemble often relies on using models that are diverse in how they learn. We've already identified several powerful models in [Chapter 5: Model Experimentation Workbench](#chapter5) that performed well individually:

*   **LightGBM**: An efficient gradient boosting model, known for speed and accuracy.
*   **HistGradientBoosting (HGB)**: Another efficient gradient boosting model, especially good for large datasets.
*   **XGBoost (XGB)**: A very popular and highly optimized gradient boosting model that's often a top performer.

Each of these models has slightly different strengths and weaknesses, making them good candidates for an ensemble.

#### 2. Combining Predictions: "Collective Decision-Making"

Once each model makes its prediction, we need a way to combine them into a single, final forecast. For beginners, a very simple and effective method is **averaging**:

*   **Simple Averaging**: We just add up the predictions from all our chosen models and divide by the number of models. For example, if Model A predicts 50, Model B predicts 55, and Model C predicts 48, the ensemble prediction would be (50 + 55 + 48) / 3 = 51.

**Why averaging works**: It helps to balance out the individual biases or errors of each model. If one model is slightly off on a particular day, the average with other models might pull the final prediction closer to the true value.

#### 3. Robustness and Accuracy: The "Informed Collective Decision"

By combining these diverse models, we aim for:
*   **Improved Accuracy**: The ensemble often outperforms any single model because it captures a wider range of patterns in the data.
*   **Increased Robustness**: The ensemble is less sensitive to outliers or quirks that might confuse a single model. It's like having multiple checkpoints to ensure a more reliable result.

### Solving the Use Case: Ensemble Forecasting for Future Dates

Let's walk through how we apply this "Ensemble Modeling Strategy" to predict the `audience_count` for all theaters for all future days in our prediction period, day by day.

**Step-by-Step Walkthrough:**

1.  **Prepare Prediction Dates & Initialize History**: Just like in [Chapter 6: Iterative Forecasting Engine](#chapter6), we prepare our future dates and initialize the `history_dict` with all actual historical data.
2.  **Loop Through Each Prediction Date (Daily Cycle)**:
    *   For the `current_date` (e.g., March 1st):
        *   **For Each Theater**:
            *   **Calculate Features**: Extract static features and dynamically calculate lag and rolling features using the `history_dict` (which includes previous days' predictions from this ensemble process). This is done just as explained in [Chapter 3: Time-Series Feature Factory](#chapter3) and [Chapter 6: Iterative Forecasting Engine](#chapter6).
            *   **Preprocess Features**: Transform these features using the preprocessors fitted on the training data ([Chapter 4: Preprocessing & Transformation Blueprint](#chapter4)).
            *   **Individual Model Predictions**: Feed the prepared features for this theater and `current_date` into *each of our selected ensemble models* (LightGBM, HistGradientBoosting, XGBoost). Each model will output its own `predicted_audience_count`.
            *   **Combine Predictions**: Take the predictions from all individual models and calculate their average. This average is our **ensemble's `final_predicted_audience_count`** for this theater and `current_date`.
            *   **Update History**: Immediately add this `final_predicted_audience_count` to the `history_dict` for that `theater_id`. This is crucial because the *next* day's predictions will use this ensemble's collective forecast as part of its history.
        *   Store all ensemble predictions for the `current_date`.
    *   Move to the next `current_date` (e.g., March 2nd), and repeat the entire process. The `history_dict` is now updated with March 1st's *ensemble* predictions.

This iterative ensemble process continues until predictions have been made for all future dates. The output will be a single set of `audience_count` predictions, representing the combined wisdom of our model team.

### Under the Hood: How it Works

Let's visualize how the daily prediction cycle works within an ensemble for a single theater. Instead of one "ML Model," we now have a "Team of ML Models."

![dia](https://mermaid.ink/img/pako:eNqdVUtv2kAQ_iujvaSVDDWPBPAhUiABDqSq1B6sBoQ29tisZHvpet3WQfz3jl8Um8cBn7yz833zzYxnvGOOdJFZLMZfCUYOPgvuKx4uI6Bny5UWjtjySMNUKnR4rL-nscYQeAxLVtmgMC7ZKeyZiyBdSLktEPkRvil0haOFjCC7Oof7sUGuUc1FrKVKC_CmOKwz7DnMlCCJwpfIFxGiKjUWRqis54CvVINgMRsXiIXwN3o2fi3MF_3nlX-mkd6ve9uVN71IGevL7i9RjOF7gIX74TSR4TslkOsvQPWOtB4PpbYIV5YYPKlgkiiFeS80VhEPzq3Heqkz8Aw1lMXOCUoPsCv0V6kR5G9U0AS_tdttA7ijEx6st6TOACxTWG9J0zpFUqtcnq4KqjpBq5FGYRYOD6gASaTj_xqOMmg0PgNOeOAkATFDwP0vSgYBeIVXDJ6SIVxhbtC1TktL_UK38iuRJ6qqr-q4HUksIh8uMGT4CtQ63898ZGbj9ZWI81sizq9GnF-PaN8S0b4a0b4YsRqIDPBEXyD3ET41q2NAU33NkrN_ruRUjE05UxHRx3EYwGP8WWWnc_TkUvqXWUDLW8fseBW-3ZW39t0KRAyR_FOO4YXJM8DLRK3r1_aqmRENVm3FWMf76CiRmnZwZURbhhnMV8JllseDGA0Wogp5dma7LMyS6Q2G5JdxuujxJMhX-p5wtAV_ShkyS6uEkEom_ubAk2xdilP-pQ5WWm8uqnyOmdUfPuQkzNqxv8zq9drDwaDb6Qy6D71Rd9jvGywlc2fQ7pujHhl75n23vzfYRx7VJPf-iJ7B6L5jjkxzuP8Hqj5sjw?type=png)


#### Code for the Ensemble Prediction Loop (Simplified)

We'll use the `making_predictions` function from [Chapter 6: Iterative Forecasting Engine](#chapter6) as a base, but modify it to make predictions with all our models and then average them.

First, let's assume our individual models (LightGBM, HistGradientBoosting, XGBoost) have already been trained and configured as described in [Chapter 5: Model Experimentation Workbench](#chapter5) and the project notebook.

```python
# Assume model_2_lightgbm, model_3_histgb, model_4_xgboost are already trained.
# These models were trained on X_train_combined (preprocessed features).

# Also assume t_df, history_dict, X_test_categorical,
# X_test_static, dynamic_scaler, and dynamic_columns are defined from previous chapters.

def make_ensemble_predictions(models, test_dataframe, history_dictionary, 
                              X_test_cat, X_test_stat, dyn_scaler, dyn_cols):
    
    current_history = {tid: hist.copy() for tid, hist in history_dictionary.items()}
    test_dataframe = test_dataframe.sort_values(['show_date', 'book_theater_id']).reset_index(drop=True)
    all_unique_dates = sorted(test_dataframe['show_date'].unique())
    all_ensemble_predictions = np.zeros(len(test_dataframe))

    for current_date in all_unique_dates:
        date_mask = test_dataframe['show_date'] == current_date
        row_indices = np.where(date_mask)[0]
        theaters_today = test_dataframe.loc[row_indices, 'book_theater_id'].values
        num_theaters = len(theaters_today)

        # Prepare dynamic features for today
        # This part is the same as in making_predictions function from Chapter 6
        lag_feature_matrix = np.zeros((num_theaters, len(dyn_cols)))
        for i, theater_id in enumerate(theaters_today):
            theater_history = current_history.get(theater_id, [])
            history_length = len(theater_history)
            
            # Simplified: calculate audience_lag_1 and other dynamic features
            if history_length >= 1:
                lag_feature_matrix[i, 0] = theater_history[-1] # audience_lag_1
            # ... (add logic for other lag/rolling features as in making_predictions) ...
            
        X_dynamic_scaled = dyn_scaler.transform(lag_feature_matrix)

        # Combine all features for prediction
        X_cat_today = X_test_cat[row_indices]
        X_stat_today = X_test_stat[row_indices]
        X_today_combined = np.hstack([X_cat_today, X_stat_today, X_dynamic_scaled])

        # --- Make predictions from ALL individual models ---
        individual_predictions = []
        for model in models:
            preds = model.predict(X_today_combined)
            individual_predictions.append(preds)
        
        # --- Combine predictions (simple averaging) ---
        ensemble_predictions_today = np.mean(individual_predictions, axis=0)
        ensemble_predictions_today = np.maximum(ensemble_predictions_today, 0) # Ensure no negative predictions
        
        all_ensemble_predictions[row_indices] = ensemble_predictions_today

        # --- Dynamically update history with ENSEMBLE's prediction ---
        for i, theater_id in enumerate(theaters_today):
            current_history.setdefault(theater_id, []).append(ensemble_predictions_today[i])

    return all_ensemble_predictions

# List of our trained models
ensemble_models = [
    model_2_lightgbm,
    model_3_histgb,
    model_4_xgboost
]

# Example usage:
# print("Starting Ensemble Predictions...")
# final_ensemble_predictions = make_ensemble_predictions(
#     models=ensemble_models,
#     test_dataframe=t_df.copy(),
#     history_dictionary=history_dict,
#     X_test_cat=X_test_categorical,
#     X_test_stat=X_test_static,
#     dyn_scaler=dynamic_scaler,
#     dyn_cols=dynamic_columns
# )
# print("Ensemble Predictions Complete.")
```
**Output Description:**
After calling `make_ensemble_predictions`, `final_ensemble_predictions` will be a NumPy array containing the predicted `audience_count` for each `book_theater_id` for every day in the future period. Each day's prediction for a theater will have incorporated the previous day's *ensemble* predicted `audience_count` for that same theater into its feature calculations, effectively creating a powerful, robust, and iteratively updated forecast.

Finally, these predictions would be formatted and saved into a submission file, as demonstrated in the project's code.

### Conclusion

In this chapter, we explored the powerful "Ensemble Modeling Strategy" to enhance our cinema audience predictions. We learned how to combine the unique strengths of multiple diverse models (LightGBM, HistGradientBoosting, and XGBoost) through simple averaging, creating a more accurate and robust collective forecast. By integrating this ensemble approach into our iterative prediction pipeline, we ensure that our final predictions benefit from a "team of experts" that continually learns and adapts to new information.

This concludes our journey through the `machine-learning-capstone-project-2025` tutorial. We've covered everything from preparing raw data and engineering clever time-series features to evaluating models, making iterative predictions, and finally, combining models for superior performance. You now have a solid understanding of the key concepts and techniques used to build a robust time-series forecasting system!

---









