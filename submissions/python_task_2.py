import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    df = pd.read_csv("D:\MapUp-Data-Assessment-F\datasets\dataset-3.csv")

    unique_ids = sorted(set(df['id_1'].unique()) | set(df['id_2'].unique()))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)
    distance_matrix = distance_matrix.fillna(0)


    for _, row in df.iterrows():
        id_1, id_2, distance = row['id_1'], row['id_2'], row['distance']

        
        distance_matrix.at[id_1, id_2] += distance
        distance_matrix.at[id_2, id_1] += distance

    return distance_matrix


dataset_path = "D:\MapUp-Data-Assessment-F\datasets\dataset-3.csv"
result_distance_matrix = calculate_distance_matrix(dataset_path)
print(result_distance_matrix)


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    upper_triangle = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))

    
    unrolled_series = upper_triangle.stack()

   
    unrolled_df = unrolled_series.reset_index()

    
    unrolled_df.columns = ['id_start', 'id_end', 'distance']

    return unrolled_df

result_unrolled_df = unroll_distance_matrix(result_distance_matrix)
print(result_unrolled_df)

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

   
    reference_rows = df[df['id_start'] == reference_id]

    average_distance = reference_rows['distance'].mean()

   
    threshold = 0.1 * average_distance

   
    within_threshold_rows = df[
        (df['distance'] >= average_distance - threshold) &
        (df['distance'] <= average_distance + threshold)
    ]

   
    within_threshold_ids = sorted(within_threshold_rows['id_start'].unique())

    return within_threshold_ids

reference_id = 1
result_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled_df, reference_id)
print(result_within_threshold)

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
  
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

   
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


result_with_toll_rate = calculate_toll_rate(result_unrolled_df)
print(result_with_toll_rate)


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Define time ranges and discount factors
    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0)),
        (time(10, 0, 0), time(18, 0, 0)),
        (time(18, 0, 0), time(23, 59, 59))
    ]
    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Initialize empty lists for new columns
    start_day_list = []
    start_time_list = []
    end_day_list = []
    end_time_list = []

    # Iterate over time ranges and calculate toll rates for each range
    for start_time, end_time in time_ranges:
        # Apply discount factors based on weekday or weekend
        discount_factors = weekday_discount_factors if start_time.hour < 10 else [weekend_discount_factor] * len(df)

        # Add new columns for time-based toll rates
        start_day_list.extend(df['start_datetime'].dt.day_name())
        start_time_list.extend([start_time] * len(df))
        end_day_list.extend(df['end_datetime'].dt.day_name())
        end_time_list.extend([end_time] * len(df))

        # Modify vehicle columns based on discount factors
        for column in df.columns[3:]:
            df[column] = df[column] * discount_factors

    # Add new columns to the DataFrame
    df['start_day'] = start_day_list
    df['start_time'] = start_time_list
    df['end_day'] = end_day_list
    df['end_time'] = end_time_list

    return df

# Example usage:
# Assuming result_unrolled_df is the DataFrame obtained from Question 3
result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_unrolled_df)
print(result_with_time_based_toll_rates)