import os
import glob
import numpy as np
import pandas as pd
from operator import attrgetter

def load_data(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    dfs = [pd.read_csv(file, sep=';') for file in csv_files]
    raw_data = pd.concat(dfs, ignore_index=True)
    
    return raw_data

def normalize_column(df, col):
    min_val = df[col].min()
    max_val = df[col].max()
    values = (df[col] - min_val) / (max_val - min_val + 1e-8)
    return values

def merge_data(df1, df2, left_col_list, right_col_list, suffixes_set=None, how_type='left'):
    if suffixes_set is not None:
        merged_data = pd.merge(
            df1, df2,
            left_on=left_col_list,
            right_on=right_col_list,
            suffixes=suffixes_set,
            how=how_type
        )
    else:
        merged_data = pd.merge(
            df1, df2,
            left_on=left_col_list,
            right_on=right_col_list,
            how=how_type
        )
    return merged_data

def map_id(df, col1, col2, col3, col4):
    return (
        df[col1].astype(str) + '|' +
        df[col2].astype(str) + '|' +
        df[col3].astype(str) + '|' +
        df[col4].astype(str)
    )


def days_diff_calculate(df, col_items = 'stock_code', col_customers = 'customer_id', col_time = 'invoice_date'):
    df = df[[col_customers, col_items, col_time]].copy()
    df[col_time]  = pd.to_datetime(df[col_time], errors='coerce')
    df[col_time] = pd.to_datetime(df[col_time], errors='coerce').dt.normalize()
    df = df.drop_duplicates(subset=[col_customers, col_items, col_time], keep='first')

    df_sorted = df.sort_values(by=[col_customers, col_items, col_time])
    df_sorted['days_diff'] = df_sorted.groupby([col_customers, col_items])[col_time].diff().dt.days

    df_sorted = df_sorted[df_sorted['days_diff'].notna()]
    df_grouped = df_sorted.groupby([col_customers, col_items]).agg({'days_diff':'mean'}).reset_index()

    diff_merge = merge_data(df[[col_items, col_customers]].drop_duplicates( keep='first').reset_index(drop=True), \
                            df_grouped, [col_customers, col_items], [col_customers, col_items], None, 'left')
    
    # diff_merge['days_diff'] = diff_merge['days_diff'].fillna(0)

    return diff_merge


def calculate_month_gap(df, train_months, test_months):
    train_months = pd.to_datetime(train_months)
    test_months = pd.to_datetime(test_months)
    df = df.copy()
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    
    # Split data by training and testing months
    training_data = df[df['month'].isin(train_months)]
    testing_data = df[df['month'].isin(test_months)]
    
    # Get list of customers in testing_data, remove duplicates
    first_occurrence_testing_data = testing_data[['customer_id', 'month']].drop_duplicates(keep='first')
    
    # Update max month from training_data
    first_occurrence_testing_data['last_train_month'] = training_data['month'].max()
    
    # Identify customers in training_data
    first_occurrence_testing_data['return_status'] = np.where(
        first_occurrence_testing_data['customer_id'].isin(training_data['customer_id'].unique()), 
        'yes', 
        'no'
    )
    
    # Calculate month interval
    first_occurrence_testing_data['month_diff'] = np.where(
        first_occurrence_testing_data['return_status'] == 'yes', 
        (first_occurrence_testing_data['month'].dt.to_period('M') - first_occurrence_testing_data['last_train_month'].dt.to_period('M')).apply(attrgetter('n')), None )

    # Group by 'customer_id' and calculate the min value of 'month_diff'
    first_occurrence_testing_data = first_occurrence_testing_data.groupby('customer_id').agg({'month_diff': 'min'}).reset_index()
    
    # Aggregate by 'month_diff'
    results_by_diff_month = first_occurrence_testing_data.groupby('month_diff').agg({'customer_id': 'nunique'}).reset_index()
    results_by_diff_month.rename(columns={'customer_id': 'returning_customers'}, inplace=True)

    # Calculate the rate of returning customers
    total_customers = training_data['customer_id'].nunique()
    # results_by_diff_month = results_by_diff_month[results_by_diff_month['month_diff'] != 0]
    results_by_diff_month['return_rate'] = (results_by_diff_month['returning_customers'] / total_customers) * 100
    results_by_diff_month['total_customers'] = total_customers
    
    return results_by_diff_month

def calculate_customer_return_rate(df, interval_length, window_length):
    aggregated_results = pd.DataFrame()
    total_customers = 0
    l = 0
    
    # Run calculation loop for time intervals
    for i in range(0, len(df['month'].unique()) - window_length - interval_length, interval_length):
        train_months = df['month'].unique()[i:i+interval_length]
        test_months = df['month'].unique()[i+interval_length:i+interval_length+window_length]
        
        if any(month in test_months for month in ['2010-10', '2010-11']):
            pass
        else:
            # Call the function to calculate the month gap
            gap_results = calculate_month_gap(df, train_months, test_months)
            
            total_customers += gap_results['total_customers'].iloc[0]
            aggregated_results = pd.concat([aggregated_results, gap_results], ignore_index=True)
            l += 1

    # Average return rate
    return_rate_results = aggregated_results.groupby('month_diff').agg({'return_rate': 'mean'}).reset_index()
    return_rate_results['interval_length'] = interval_length
    return_rate_results['average_total_customers'] = int(total_customers / l)

    return return_rate_results

  