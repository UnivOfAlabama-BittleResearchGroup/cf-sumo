import numpy as np
import pandas as pd

from omegaconf import OmegaConf


def error_metrics(
    pos_list: list, conf: OmegaConf, rw_array: np.ndarray
) -> OmegaConf:
    """Calculate error metrics and update config file."""

    pos_list = np.array(pos_list)
    n = rw_array.shape[0]

    # why two different pandas files here? Shouldn't we just use the sim_df? -> fixed
    # actual_df = leader_df
    # error_df = actual_df.copy()
    # error_df = error_df.dropna()
    # error_df = error_df.drop(
    #     columns=["epoch_time", "leadvelocity", "leadposition", "leadacceleration"]
    # )
    # error_df = error_df.reset_index(drop=True)
    # sim_df = sim_df.reset_index(drop=True)

    for f in [rmsn, rmspe, mpe]:
        conf.CFParameters[f.__name__] = float(f(pos_list, rw_array, n))

    return conf


def rmsn(pos_list: np.ndarray, rw_array: np.ndarray, n) -> float:
    # root mean square error normalized
    error_lead_pos = np.nan_to_num(rw_array[:, 0])  # lead pos array
    error_follow_pos = np.nan_to_num(rw_array[:, 1])  # follow pos array
    

    sim_follow_pos = pos_list[:, 0]  # follow pos
    print("error_follow_pos shape:", error_follow_pos.shape)
    print("sim_follow_pos shape:", sim_follow_pos.shape)

    pos_observed_sum = np.sum(error_lead_pos)
    rmsn_numerator = np.sqrt(n * np.sum(np.square(error_follow_pos - sim_follow_pos)))
    return rmsn_numerator / pos_observed_sum


def rmspe(pos_list: np.ndarray, rw_array: np.ndarray, n) -> float:
    # root mean square percentage error
    error_follow_pos = rw_array[:, 1]  # follow pos
    sim_follow_pos = pos_list[:, 0]  # follow pos

    dev = np.square((sim_follow_pos - error_follow_pos) / error_follow_pos)
    return np.sqrt(np.sum(dev) / n)


def mpe(pos_list: np.ndarray, rw_array: np.ndarray, n) -> float:
    # mean percentage error
    error_lead_pos = rw_array[:, 0]  # lead pos
    error_follow_pos = rw_array[:, 1]  # follow pos

    sim_follow_pos = pos_list[:, 0]  # follow pos
    mean = np.sum((sim_follow_pos - error_follow_pos) / error_follow_pos)
    return mean / n
