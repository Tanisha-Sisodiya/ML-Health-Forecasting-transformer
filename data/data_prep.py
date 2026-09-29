'''
data_prep.py will load raw Nasa C-MAPSS data, 
assign proper column names and calculate the Remaining Useful Life (RUL) for every data point.
Then saves the cleaned dataset

The RUL is our target variable, which we will use to train our tranformer for forecaasting

'''

import pandas as pd
import os

# Function to read the raw file and assign column names based on the C-MAPSS data documentation
def load_and_prep_data(file_path):
    #define the column names for NASA C-MAPSS FD001 data
    index_names = ['engine_id','time_cycles'] #these are the two primary identifiers (unit, operation cycle)
    setting_names = [f'op_setting_{i}' for i in range(1,4)] #3 operational settings
    sensor_names = [f'sensor_{i}' for i in range(1,22)] #21 sensor measurements
    cols_names = index_names + setting_names + sensor_names #combine all column names in a list to use for column headers


    #load the data
    df = pd.read_csv(
        file_path,
        sep='\s+',   #regular expression separator, seperate columns based on whitespace (1 or more spaces) (to handle inconsistent spacing between columns numbers in the dataset)
        header = None,
        names = cols_names,  #assign column names
        index_col= False #do not use any column as index
    ).dropna(axis=1,how='all') #drops columns with all NaN values

    return df

#RUL is the target variable
def calculate_rul(df):
    #Calculate the max cycle for each engine
    max_cycles = df.groupby('engine_id')['time_cycles'].max()

    #map back the max cycles to the DataFrame
    df = df.merge(max_cycles.rename('max_cycle'), on='engine_id', how='left')

    #RUL calculation (RUL = max_cycle - current_cycle)
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df.drop('max_cycle', axis=1, inplace=True) #drop the max_cycle column as it's no longer needed after calculating RUL

    return df


if __name__ == "__main__":
    DATA_DIR = 'data'
    TRAIN_FILE = 'train_FD001.txt'
    CLEAN_FILE = 'train_FD001_clean.parquet'

    file_path = os.path.join(DATA_DIR, TRAIN_FILE)



    '''
    Main execution block

    This pipeline first calls the load_and_prep_data function to read the raw data file 
    Then calcualtes the RUL using the calcualte_rul function on the raw data df
    Then saves the clenaed data to a parquet file
    Parquet is used since its effiecent for storing larger datasets and offers faster read/writes
    and smaller file sizes compared to CSV

    **Error handling**: If the raw data file is not found, an error message is printed that file DNE
    '''

    if os.path.exists(file_path):
        #load and prep data
        df_raw = load_and_prep_data(file_path)
        
        #Calculate RUL from raw data
        df_processed = calculate_rul(df_raw)

        output_path = os.path.join(DATA_DIR, CLEAN_FILE)
        df_processed.to_parquet(output_path, index=False)

        print(f"Successfully processed {TRAIN_FILE} (Rows: {len(df_processed)})")
        print(f"Clean file saved to: {output_path}")
        print("Columns:", list(df_processed.columns))
        print("Check RUL for Engine 1:")
        print(df_processed[df_processed['engine_id'] == 1][['time_cycles', 'RUL']].tail())

    else:
        print(f"ERROR: Raw data file not found at {file_path}. Please ensure you've downloaded {TRAIN_FILE} into the '{DATA_DIR}' directory.")

    



