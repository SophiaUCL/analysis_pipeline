
import numpy as np
import pandas as pd
from pathlib import Path

def make_epoch_times_csv(derivatives_base: Path, trials_to_include: list):
    """ Creates epoch_times.csv in rawsession_folder/task_metadata
    epochs are taken from behaviour csv/excel file
    
    
    Inputs
    ------
    derivatives_base (Path): path to derivatives folder
    trials_to_include (list): list of trial numbers
    
    
    Creates
    --------
    rawsession_folder/'task_metadata'/'epoch_times.csv': df with epoch times
    """
        # Loading rawsession folder
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    epoch_times_path = rawsession_folder/'task_metadata'/'epoch_times.csv'

    if epoch_times_path.exists():
        return 1
    else:
        folder = rawsession_folder/ 'task_metadata'
        csv_path = list(folder.glob('behaviour*.csv'))
        if len(csv_path) > 0:
            epoch_times_allcols = pd.read_csv(csv_path[0], header=None)
        else:
            excel_path = list(folder.glob('behaviour*.xlsx'))
            if len(excel_path) > 0:
                epoch_times_allcols = pd.read_excel(excel_path[0], header=None)
            else:
                raise FileNotFoundError('No behaviour CSV or Excel file found in the specified folder.')


    epoch_times= epoch_times_allcols.iloc[:, [10, 12, 14, 16, 18]]
    epoch_times.columns = ['epoch 1 end', 'epoch 2 start', 'epoch 2 end', 'epoch 3 start', 'epoch 3 end']
    epoch_times.insert(0, "epoch 1 start", np.zeros(len(epoch_times)))
    epoch_times.insert(0,'trialnumber',  trials_to_include)

    epoch_times.to_csv(epoch_times_path, index=False)
    print(f"Data saved to {epoch_times_path}")
    
if __name__ == "__main__":
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-002_id-1U\ses-05_date-18072025"
    trials_to_include = np.arange(1,11)
    make_epoch_times_csv(rawsession_folder, trials_to_include)
    