import pandas as pd
from pathlib import Path
import numpy as np

def get_MRL_data(derivatives_base: Path, path_to_df: Path = None) -> pd.DataFrame:
    """ Gets the path for the MRL data used, either the path is provided or user provides it"""
    # df path
    if path_to_df is not None:
        df_all = pd.read_csv(path_to_df)
        print(f"Making roseplot from data from {path_to_df.base}")
    else:
        df_path_base = derivatives_base/'analysis'/'cell_characteristics'/'spatial_features'/'spatial_data'
        df_options = list(df_path_base.glob("directional_tuning*.csv"))
        if len(df_options) == 1:
            df_path = df_options[0]
        else:
            print([f.base for f in df_options])
            user_input = input('Please provide the number of the file in the list you would like to look at (starting at 1): ')
            user_input = np.int32(user_input)
            df_path = df_options[user_input - 1]
            print(f"Making roseplot from data from {df_options[user_input - 1].base}")
        df_all = pd.read_csv(df_path)
            
    return df_all

def get_directories(derivatives_base: Path, deg: int) -> tuple[pd.DataFrame, Path]:
    """ Returns directories"""
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Dataframe with raised arms
    metadata_folder = rawsession_folder / 'task_metadata'
    csv_path = list(metadata_folder.glob( 'behaviour*.csv'))

    if len(csv_path) > 0:
        behaviour_df = pd.read_csv(csv_path[0], header=None)
    else:
        excel_path = list(metadata_folder.glob( 'behaviour*.xlsx'))
        if len(excel_path) > 0:
            behaviour_df = pd.read_excel(excel_path[0], header=None)
        else:
            raise FileNotFoundError('No behaviour CSV or Excel file found in the specified folder.')
    
    # Output path: 
    output_folder_plot = derivatives_base/'analysis'/'cell_characteristics'/'spatial_features'/'spatial_plots'/'roseplots'
    output_folder_plot.mkdir(parents = True, exist_ok = True)
    output_path_plot = output_folder_plot /  f'roseplot_{deg}_degrees.png'
    return behaviour_df, output_path_plot

def get_sum_bin(mean_dir_arr: np.ndarray, num_spikes_arr: np.ndarray, bin_edges: np.ndarray | list) -> list:
    """ Gets the number of spikes in each bin"""
    sum_count_bin = []

    bin_idx = np.digitize(mean_dir_arr, bin_edges) - 1 

    # Count the number of spikes for each element in bin
    for i in range(len(bin_edges)-1):
        indices = np.where(bin_idx == i)
        num_spikes_i = num_spikes_arr[indices]
        sum_count_bin.append(np.sum(num_spikes_i))
    return sum_count_bin


def load_directories(derivatives_base: Path):
    """ Used in make_spatiotemp_plots.py"""
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    pos_data_dir = derivatives_base / "analysis" / "spatial_behav_data" / "XY_and_HD"
    if not pos_data_dir.exists():
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")

    epoch_times = pd.read_csv(rawsession_folder / "task_metadata" / "epoch_times.csv")
    trials_length = pd.read_csv(rawsession_folder / "task_metadata" / "trials_length.csv")
    df_unit_metrics = pd.read_csv(derivatives_base / "analysis" / "cell_characteristics" / "unit_features" / "all_units_overview" / "unit_metrics.csv")

    output_folder_plot = derivatives_base / "analysis" / "cell_characteristics" / "spatial_features" / "spatial_plots" / "task_overview"
    output_folder_data = derivatives_base / "analysis" / "cell_characteristics" / "spatial_features" / "spatial_data"

    output_folder_plot.mkdir(parents=True, exist_ok=True)
    output_folder_data.mkdir(parents=True, exist_ok=True)

    print(f"Figures will be saved to {output_folder_plot}")

    return pos_data_dir, output_folder_data, output_folder_plot, epoch_times, df_unit_metrics, trials_length


def get_xy_pos(pos_data_dir: Path, tr: int):
    """ Gets xy pos of the animal for this trial"""
    trial_csv_name = f'XY_HD_t{tr}.csv'
    trial_csv_path = pos_data_dir/trial_csv_name
    xy_hd_trial = pd.read_csv(trial_csv_path)            
    x = xy_hd_trial.iloc[:, 0].to_numpy()
    y = xy_hd_trial.iloc[:, 1].to_numpy()
    hd = xy_hd_trial.iloc[:, 2].to_numpy()
    if np.nanmax(hd) > 2*np.pi + 0.1:
        hd_rad = np.deg2rad(hd)
    else:
        hd_rad = hd
    return x, y, hd_rad

def make_new_element(unit_id: int, tr: int, e: int, MRL: float, mu: float, percentiles_95_value: float, percentiles_99_value: float, spike_train_this_epoch: list) -> dict:
    """ Makes new element for df"""
    new_element = {
        'cell': unit_id,
        'trial': tr,
        'epoch': e,
        'MRL': MRL,
        'mean_direction': np.rad2deg(mu),
        'mean_direction_rad':mu,
        'percentiles95': percentiles_95_value,
        'percentiles99': percentiles_99_value,
        'significant': 'ns', 
        'num_spikes': len(spike_train_this_epoch)
    } 
    if MRL > percentiles_95_value:
        new_element['significant'] = 'sig'   
    return new_element

def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    """
    Copied from Pycircstat documentation
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r


def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # Copied from picircstat documentation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"

    return ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) /
            np.sum(w, axis=axis))
