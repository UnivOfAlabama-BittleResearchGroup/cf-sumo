import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import sumolib
import sys
import optparse
import random
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from omegaconf import DictConfig

if __name__ == "__main__":
    ROOT = Path(__file__).parent
    os.environ["ROOT"] = str(ROOT)

    actual_df = pd.read_csv(ROOT / "data" / "test.csv")
    pd.options.plotting.backend = "plotly"
    error_df = actual_df.copy()
    error_df = error_df.dropna()
    error_df = error_df.drop(
        columns=["epoch_time", "leadvelocity", "leadposition", "leadacceleration"]
    )
    error_df = error_df.reset_index(drop=True)

    def xml_to_df(xml_path):
        rows = []
        xml_path = str(xml_path)

        for r in sumolib.xml.parse_fast_nested(
            xml_path,
            "timestep",
            ["time"],
            "vehicle",
            ["id", "speed", "pos", "acceleration"],
        ):
            rows.append({**r[0]._asdict(), **r[1]._asdict()})

        return pd.DataFrame(rows)

    sumo_df = xml_to_df(ROOT / "sumo-xml" / "output" / "d_fcd.xml")
    leader_df = sumo_df.query("id == 'leader'")
    follower_df = sumo_df.query("id == 'follower'")

    config_dir = ROOT / "sumo-xml" / "output" / "configs"
    config_files = [OmegaConf.load(f) for f in config_dir.glob("*.yaml")]

    df_dict = {}
    for config in config_files:
        df_dict[config.Config.run_id] = xml_to_df(
            Path(config.Config.output_path) / f"{config.Config.run_id}_fcd.xml"
        )
        df_dict[config.Config.run_id] = df_dict[config.Config.run_id].query(
            "id == 'follower'"
        )

    def rmsn(run_id, sim_df, error_df):
        # root mean square error normalized
        n = len(error_df)
        sim_df = sim_df.reset_index(drop=True)
        # print(error_df['followposition'] - sim_df['pos'].astype(float))
        pos_observed_sum = np.sum(error_df["followposition"].astype(float))

        rmsn_numerator = np.sqrt(
            n
            * np.sum(
                np.square(error_df["followposition"] - sim_df["pos"].astype(float))
            )
        )
        rmsn_val = rmsn_numerator / (pos_observed_sum)

        config_file = config_dir / f"{run_id}_config.yaml"
        config = OmegaConf.load(config_file)
        config.CFParameters.rmsn = float(rmsn_val)
        OmegaConf.save(config, config_file)

        return rmsn_val

    def rmspe(run_id, sim_df, error_df):
        # root mean square percentage error
        n = len(error_df)
        sim_df = sim_df.reset_index(drop=True)
        dev = np.square(
            (sim_df["pos"].astype(float) - error_df["followposition"])
            / error_df["followposition"]
        )
        rmspe_val = np.sqrt(np.sum(dev) / n)

        config_file = config_dir / f"{run_id}_config.yaml"
        config = OmegaConf.load(config_file)
        config.CFParameters.rmspe = float(rmspe_val)
        OmegaConf.save(config, config_file)

        return rmspe_val

    def mpe(run_id, sim_df, error_df):
        # mean percentage error
        n = len(error_df)
        sim_df = sim_df.reset_index(drop=True)
        mean = np.sum(
            (sim_df["pos"].astype(float) - error_df["followposition"])
            / error_df["followposition"]
        )
        mpe_val = mean / n

        config_file = config_dir / f"{run_id}_config.yaml"
        config = OmegaConf.load(config_file)
        config.CFParameters.mpe = float(mpe_val)
        OmegaConf.save(config, config_file)

        return mpe_val

    for key, value in df_dict.items():
        sim_df = value
        rmsn(key, sim_df, error_df)
        rmspe(key, sim_df, error_df)
        mpe(key, sim_df, error_df)

    # heatmap code
    data = []
    for config_file in config_files:
        config = config_file
        acceleration = config.CFParameters.acceleration
        deceleration = config.CFParameters.deceleration
        rmsn = config.CFParameters.rmsn
        data.append((acceleration, deceleration, rmsn))

    unique_accel = sorted(set(x[0] for x in data))
    unique_decel = sorted(set(x[1] for x in data))

    z_matrix = [[None] * len(unique_decel) for _ in range(len(unique_accel))]

    for accel, decel, rmsn in data:
        accel_idx = unique_accel.index(accel)
        decel_idx = unique_decel.index(decel)
        z_matrix[accel_idx][decel_idx] = rmsn

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix, x=unique_decel, y=unique_accel, colorscale="Viridis"
        )
    )
    fig.update_layout(
        title="RMSN Heatmap Acceleration vs Deceleration",
        xaxis_title="Deceleration",
        yaxis_title="Acceleration",
    )
    fig.show()

    # Read the data from the config files into a list of dictionaries
    data = []
    for config_file in config_files:
        config = config_file
        acceleration = config.CFParameters.acceleration
        deceleration = config.CFParameters.deceleration
        tau = config.CFParameters.tau
        speedFactor = config.CFParameters.speedFactor
        rmsn = config.CFParameters.rmsn
        rmspe = config.CFParameters.rmspe
        data.append(
            {
                "acceleration": acceleration,
                "deceleration": deceleration,
                "tau": tau,
                "speedFactor": speedFactor,
                "rmsn": rmsn,
                "rmspe": rmspe,
                "mpe": mpe,
            }
        )

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Create the parallel coordinates plot
    fig = go.Figure(
        data=go.Parcoords(
            dimensions=[
                dict(
                    range=[df["tau"].min(), df["tau"].max()],
                    label="Tau",
                    values=df["tau"],
                ),
                dict(
                    range=[df["acceleration"].min(), df["acceleration"].max()],
                    label="Acceleration",
                    values=df["acceleration"],
                ),
                dict(
                    range=[df["deceleration"].min(), df["deceleration"].max()],
                    label="Deceleration",
                    values=df["deceleration"],
                ),
                dict(
                    range=[df["speedFactor"].min(), df["speedFactor"].max()],
                    label="Speed Factor",
                    values=df["speedFactor"],
                ),
                dict(
                    range=[df["rmsn"].min(), df["rmsn"].max()],
                    label="RMSN",
                    values=df["rmsn"],
                ),
                dict(),
            ],
            line=dict(
                color=df["rmsn"],
                colorscale="Viridis",
                showscale=True,
                cmin=df["rmsn"].min(),
                cmax=df["rmsn"].max(),
            ),
        )
    )

    fig.update_layout(
        title="Parallel Coordinates Plot (Tau, Acceleration, Deceleration, Speed Factor, RMSN)",
        title_x=0.5,  # Position of the title along the x-axis, 0.5 centers it
        title_font=dict(size=20),
    )

    fig.show()

    vt_fig = go.Figure(
        data=go.Scatter(
            x=actual_df["seconds"], y=actual_df["leadvelocity"], name="RW leader"
        )
    )
    vt_fig.add_scatter(
        x=actual_df["seconds"], y=actual_df["followvelocity"], name="RW follower"
    )

    vt_fig.add_scatter(x=leader_df["time"], y=leader_df["speed"], name="d_sumo leader")
    vt_fig.add_scatter(
        x=follower_df["time"], y=follower_df["speed"], name="d_sumo follower"
    )

    for key, value in df_dict.items():
        vt_fig.add_scatter(
            x=value["time"], y=value["speed"], name=f"{key}_sumo follower"
        )

    vt_fig.update_layout(
        title="Velocity vs Time (RW vs SUMO base 32)",
        xaxis_title="Time (s)",
        yaxis_title="Velocity (m/s)",
        legend_title="Source",
    )

    vt_fig.show()

    dt_fig = go.Figure(
        data=go.Scatter(
            x=actual_df["seconds"], y=actual_df["leadposition"], name="RW leader"
        )
    )

    dt_fig.add_scatter(
        x=actual_df["seconds"], y=actual_df["followposition"], name="RW follower"
    )
    dt_fig.add_scatter(x=leader_df["time"], y=leader_df["pos"], name="d_sumo leader")
    dt_fig.add_scatter(
        x=follower_df["time"], y=follower_df["pos"], name="d_sumo follower"
    )

    for key, value in df_dict.items():
        dt_fig.add_scatter(x=value["time"], y=value["pos"], name=f"{key}_sumo follower")

    dt_fig.update_layout(
        title="Distance vs Time (RW vs SUMO)",
        xaxis_title="Time (s)",
        yaxis_title="Distance (m)",
        legend_title="Source",
    )

    dt_fig.show()

    at_fig = go.Figure(
        data=go.Scatter(
            x=actual_df["seconds"], y=actual_df["leadacceleration"], name="RW leader"
        )
    )

    at_fig.add_scatter(
        x=actual_df["seconds"], y=actual_df["followacceleration"], name="RW follower"
    )
    at_fig.add_scatter(
        x=leader_df["time"], y=leader_df["acceleration"], name="d_sumo leader"
    )
    at_fig.add_scatter(
        x=follower_df["time"], y=follower_df["acceleration"], name="d_sumo follower"
    )

    for key, value in df_dict.items():
        at_fig.add_scatter(
            x=value["time"], y=value["acceleration"], name=f"{key}_sumo follower"
        )

    at_fig.update_layout(
        title="Acceleration vs Time (RW vs SUMO)",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s^2)",
        legend_title="Source",
    )

    at_fig.show()
