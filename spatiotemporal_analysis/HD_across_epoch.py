import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

def get_MRL_data(derivatives_base, path_to_df = None):
    """ Gets the path for the MRL data used, either the path is provided or user provides it"""
    # df path
    if path_to_df is not None:
        df_path = pd.read_csv(path_to_df)
        print(f"Making roseplot from data from {os.path.basename(path_to_df)}")
    else:
        df_path_base = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data')
        df_options = glob.glob(os.path.join(df_path_base, "directional_tuning*.csv"))
        if len(df_options) == 1:
            df_path = df_options[0]
        else:
            print([os.path.basename(f) for f in df_options])
            user_input = input('Please provide the number of the file in the list you would like to look at (starting at 1): ')
            user_input = np.int32(user_input)
            df_path = df_options[user_input - 1]
            print(f"Making roseplot from data from {os.path.basename(df_options[user_input - 1])}")
            
    df_all = pd.read_csv(df_path)
    return df_all

def make_heatmap(derivatives_base, df_results):
    """ Plots and saves the significant cells across epochs"""
    pivoted = df_results.pivot(index='trial', columns='epoch', values='proportion')

    plt.figure(figsize=(5, 6))
    plt.title('Significant Cells Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Trial')

    im = plt.imshow(pivoted, aspect='auto', cmap='viridis', origin='upper', vmin=0, vmax=1)
    plt.colorbar(im, label='Proportion of Significant Cells')

    # Fix ticks
    plt.xticks(np.arange(len(pivoted.columns)), pivoted.columns)
    plt.yticks(np.arange(len(pivoted.index)), pivoted.index)

    plt.savefig(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'sig_across_epochs.png'))
    plt.show()
    
    
def get_significant_cells(df_results, df, e, tr):
    """ Gets significant cells of epoch e -1 and trial tr (cells)
    and the significant cells of epoch e and trial tr (cells_e)
    Divides intersection of the two by their union"""
    # Filter so epoch in df is e-1 and so its only significant == sig
    df_filtered = df[(df['epoch'] == e - 1) & (df['significant'] == 'sig') & (df['trial'] == tr)]
    
    cells = df_filtered['cell'].unique()

    df_e = df[(df['epoch'] == e) & (df['significant'] == 'sig') & (df['trial'] == tr)]
    cells_e = df_e['cell'].unique()
    
    # Count what proportion of cells in cells is in cells_e
    intersection = len(np.intersect1d(cells, cells_e))
    union = len(np.union1d(cells, cells_e))

    proportion = intersection / union if union > 0 else 0
    
    df_results = pd.concat([df_results, pd.DataFrame({'trial': [tr], 'epoch': [e], 'proportion': [proportion]})], ignore_index=True)
    return df_results

    
def get_significant_cells_tr(df_results, df, tr, tr_2):
    """ Gets significant cells for and trial tr (cells)
    and the significant cells for trial tr2 (cells_tr2)
    Looks at the proportion of divides intersection of cells and cells_tr2 by their untion"""

    # Fuilter for trial =  tr
    df_filtered = df[(df['significant'] == 'sig') & (df['trial'] == tr)]
    cells = df_filtered['cell'].unique()

    # Filter for trial = tr_2
    df_tr2 = df[(df['significant'] == 'sig') & (df['trial'] == tr_2)]
    cells_tr2 = df_tr2['cell'].unique()
    
    # Count what proportion of cells in cells is in cells_e
    intersection = len(np.intersect1d(cells, cells_tr2))
    union = len(np.union1d(cells, cells_tr2))

    proportion = intersection / union if union > 0 else 0

    df_results = pd.concat([df_results, pd.DataFrame({'trial I': [tr], 'trial II': [tr_2], 'proportion': [proportion]})], ignore_index=True)
    return df_results

def make_heatmap_tr(derivatives_base, df_results):
    """ Makes heatmap for signficiance acrss trials """
    pivoted = df_results.pivot(index='trial I', columns='trial II', values='proportion')

    plt.figure(figsize=(7, 6))
    plt.title('Significant Cells Across Trials')
    plt.xlabel('Trial II')
    plt.ylabel('Trial I')

    im = plt.imshow(pivoted, aspect='auto', cmap='viridis', origin='upper', vmin=0, vmax=1)
    plt.colorbar(im, label='Proportion of Significant Cells Shared')

    # Fix ticks
    plt.xticks(np.arange(len(pivoted.columns)), pivoted.columns)
    plt.yticks(np.arange(len(pivoted.index)), pivoted.index)

    plt.savefig(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_plots', 'sig_across_trials.png'))
    plt.axis('scaled')
    plt.show()
    
def sig_across_epochs(derivatives_base, trials_to_include, num_epochs = 3):
    """
    Creates 2 things:
        df with significance for each epoch e and epoch e + 1 and the proportion of cells that are significant in both
        df with comparison of how many cells are signfiicant in each trial
        Forms heatmap for both

    Args:
        derivatives_base: Path to derivatives folder
        trials_to_include: trials that we'll look at
        num_epoch: number of epochs (defaults to 3)

    Outputs:
    derivatives_base/analysis/cell_characteristics/spatial_features/spatial_data/sig_across_epochs.csv - df with signfigicance across epochs
    derivatives_base/analysis/cell_characteristics/spatial_features/spatial_data/sig_across_trials.csv - df with signfigicance across trials
    derivatives_base/analysis/cell_characteristics/spatial_features/spatial_plots/sig_across_epochs.png - heatmap of epochs significance df
    derivatives_base/analysis/cell_characteristics/spatial_features/spatial_plots/sig_across_trials.png - heatmap of trials significance df
    """
    # Df with all MRL values
    df  = get_MRL_data(derivatives_base)
    
    # Initialise empty to store results of data with trial and epoch column
    df_results = pd.DataFrame(columns = ['trial', 'epoch', 'proportion'])
    
    for tr in trials_to_include:
        for e in np.arange(2, num_epochs + 1):
            df_results = get_significant_cells(df_results, df, e, tr)

    # Make heatmap plots
    make_heatmap(derivatives_base, df_results)

    # Save results to CSV
    df_results.to_csv(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data', 'sig_across_epochs.csv'), index=False)
    
    
    # ====== Significance across trials======= 
    df_results = pd.DataFrame(columns = ['trial I','trial II', 'proportion'])
    
    for i in np.arange(0, len(trials_to_include)):
        for j in np.arange(0, len(trials_to_include)):
            tr = trials_to_include[i]
            tr_2 = trials_to_include[j]
            df_results = get_significant_cells_tr(df_results, df, tr, tr_2)

    # Plot
    make_heatmap_tr(derivatives_base, df_results)

    
    # Save results to CSV
    df_results.to_csv(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data', 'sig_across_trials.csv'), index=False)
    
    
if __name__ == "__main__":
    """
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1, 11)
    """
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-05_date-18072025\all_trials"
    trials_to_include = np.arange(1,11)
    sig_across_epochs(derivatives_base, trials_to_include)
   