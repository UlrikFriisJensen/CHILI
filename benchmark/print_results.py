##! Imports
import os
import pandas as pd

##! Main    
def main(results_dir, fstring_format='.4f'):
    # Initialize an empty list to store the dataframes
    dfs = []

    # Traverse the directory structure
    for root, dirs, files in os.walk(results_dir):
        # Iterate over the files in the current directory
        for file in files:
            # Check if the file is a CSV file
            if file.endswith('.csv'):
                # Construct the full path to the CSV file
                file_path = os.path.join(root, file)
                
                # Read the CSV file into a dataframe
                df = pd.read_csv(file_path)
                
                # Append the dataframe to the list
                dfs.append(df)

    # Concatenate all the dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True).drop('Unnamed: 0', axis=1)

    combined_df['Val F1-score'] = combined_df['Val F1-score'].str.split('(').str[1].str.split(')').str[0].astype(float)
    combined_df['Test F1-score'] = combined_df['Test F1-score'].str.split('(').str[1].str.split(')').str[0].astype(float)

    # Print the results
    for task in combined_df['Task'].unique():
        print('')
        print('-'*50)
        print('')
        print(task)
        for model in combined_df['Model'].unique():
            print('')
            print(model)
            val_mae_mean = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Val posMAE/MSE'].mean()
            val_mae_std = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Val posMAE/MSE'].std()
            test_mae_mean = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Test posMAE/MSE'].mean()
            test_mae_std = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Test posMAE/MSE'].std()
            
            print(f'Val posMAE/MSE: ${val_mae_mean:{fstring_format}} \pm {val_mae_std:{fstring_format}}$')
            print(f'Test posMAE/MSE: ${test_mae_mean:{fstring_format}} \pm {test_mae_std:{fstring_format}}$')
            
            val_f1_mean = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Val F1-score'].mean()
            val_f1_std = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Val F1-score'].std()
            test_f1_mean = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Test F1-score'].mean()
            test_f1_std = combined_df[(combined_df['Task'] == task) & (combined_df['Model'] == model)]['Test F1-score'].std()
            
            print(f'Val F1-score: ${val_f1_mean:{fstring_format}} \pm {val_f1_std:{fstring_format}}$')
            print(f'Test F1-score: ${test_f1_mean:{fstring_format}} \pm {test_f1_std:{fstring_format}}$')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, type=str)
    parser.add_argument('--fstring_format', default='.4f', type=str)
    args = vars(parser.parse_args())

    main(args['results_dir'], args['fstring_format'])