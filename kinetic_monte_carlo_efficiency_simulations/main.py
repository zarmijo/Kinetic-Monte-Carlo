import logging
import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from logger import FMOLogger
from plotter import FMOPlotter
from kmc_simulation import run_kmc_simulation
from utils import choose_k_sites_to_remove, remove_site_from_each_monomer, remove_sites


def main(config: Dict[str, Any]) -> None:
    """Execute the main workflow for FMO network efficiency simulation using a configuration dictionary.

    Args:
        config: Dictionary containing configuration parameters.
    """
    # Extract configuration parameters
    output_dir = config.get("output_dir")
    data_file = config.get("data_file")
    fmo_rc_labels = config.get("fmo_rc_labels", ("FMO1->RC", "FMO2->RC", "FMO3->RC"))
    init_state_names = config.get("init_state_names", [])
    k = config.get("k", 1)
    sites_to_remove = config.get("sites_to_remove", [])
    num_trajectories = config.get("num_trajectories", 10000)
    max_time = config.get("max_time", 1000.0)
    recording_interval = config.get("recording_interval", 0.5)
    chunk_size = config.get("chunk_size", 2000)

    logger = FMOLogger(output_dir)
    try:
        logging.info("Starting main simulation workflow.")
        logger.log_separator()

        try:
            # Read the Excel file, specifying the sheet and skipping the metadata at the bottom
            df = pd.read_excel(data_file, index_col=0, sheet_name=0)
            # Filter out rows that are not part of the rate matrix (e.g., metadata like 'Description')
            df = df[df.index.str.contains('FMO|Dissipative|PscA1_BCL', na=False)]
        except FileNotFoundError as e:
            logging.error("Failed to read data file %s: %s", data_file, e)
            raise
        except Exception as e:
            logging.error("Error processing Excel file %s: %s", data_file, e)
            raise
        logging.info("DataFrame shape: %s", df.shape)
        fmo_rate_matrix = df.to_numpy(dtype=np.float64)
        logger.log_matrix(fmo_rate_matrix, labels=df.index.to_list())
        normalized_rates = (
            fmo_rate_matrix / fmo_rate_matrix.max()
            if fmo_rate_matrix.max() > 0
            else fmo_rate_matrix.copy()
        )

        df_index = df.index.to_list()
        n_states = len(df_index)
        fmo_block_labels = np.full(n_states, -1, dtype=np.int64)
        rc_list, nonrad_list = [], []
        for i, name in enumerate(df_index):
            if name.startswith("FMO1_"):
                fmo_block_labels[i] = 0
            elif name.startswith("FMO2_"):
                fmo_block_labels[i] = 1
            elif name.startswith("FMO3_"):
                fmo_block_labels[i] = 2
            elif name.startswith("PscA1_BCL_"):
                rc_list.append(i)
            elif name == "Dissipative":
                nonrad_list.append(i)
        rc_indices = np.array(rc_list, dtype=np.int64)
        nonrad_indices = np.array(nonrad_list, dtype=np.int64)
        logging.info("RC indices: %s", rc_indices)

        all_fmo_indices = [i for i in range(n_states) if fmo_block_labels[i] >= 0]

        plotter = FMOPlotter(df_index=df_index, output_dir=output_dir, fmo_rc_labels=fmo_rc_labels)

        for init_st_name in init_state_names:
            if init_st_name not in df_index:
                logging.error("Initial state %s not found in DataFrame, skipping.", init_st_name)
                continue
            init_st_idx = df_index.index(init_st_name)

            removable_sites = [i for i in all_fmo_indices if i != init_st_idx]
            logging.info(
                "Initial state: %s (index %d), removable sites: %s",
                init_st_name,
                init_st_idx,
                [df_index[i] for i in removable_sites],
            )

            # Combinatorial site removals
            knockout_subsets = choose_k_sites_to_remove(removable_sites, k) + [[]]
            logging.info("Total combinatorial subsets generated with k=%d: %d", k, len(knockout_subsets))
            logger.log_separator()

            combinatorial_scenarios = []
            for subset in knockout_subsets:
                rates_copy = normalized_rates.copy()
                remove_sites(rates_copy, subset)
                start_time = time.time()
                sim_result = run_kmc_simulation(
                    transition_rates=rates_copy,
                    initial_state=init_st_idx,
                    max_time=max_time,
                    num_trajectories=num_trajectories,
                    recording_interval=recording_interval,
                    chunk_size=chunk_size,
                    fmo_block_labels=fmo_block_labels,
                    rc_indices=rc_indices,
                    nonrad_indices=nonrad_indices,
                )
                time_points, avg_occ, avg_rc, num_trajectories = sim_result
                sim_result = (time_points.tolist(), avg_occ, avg_rc, num_trajectories)
                final_eff = avg_rc[-1, :].sum()
                combinatorial_scenarios.append((subset, final_eff, sim_result))
                subset_labels = [df_index[x] for x in subset]
                logger.log_simulation_result(
                    init_state_name=init_st_name,
                    subset=subset,
                    subset_labels=subset_labels,
                    final_eff=final_eff,
                    duration=time.time() - start_time,
                )

            # Sort and select combinatorial scenarios
            combinatorial_scenarios.sort(key=lambda x: x[1])
            full_network = next((x for x in combinatorial_scenarios if not x[0]), None)
            combinatorial_entries = combinatorial_scenarios[:5]
            if full_network and full_network not in combinatorial_entries:
                combinatorial_entries.append(full_network)

            seen_subsets = set()
            final_combinatorial_entries = []
            for entry in combinatorial_entries:
                subset_tuple = tuple(sorted(entry[0]))
                if subset_tuple not in seen_subsets:
                    seen_subsets.add(subset_tuple)
                    final_combinatorial_entries.append(entry)
            final_combinatorial_entries.sort(key=lambda x: (bool(x[0]), x[1]))

            # Plot combinatorial scenarios
            plotter.plot_combinatorial_removals(
                final_combinatorial_entries,
                init_st_idx,
                f" ({init_st_name})",
                show_plot=True,
            )

            # Monomeric site removals
            rates_copy = normalized_rates.copy()
            monomer_subset = remove_site_from_each_monomer(rates_copy, sites_to_remove, df_index)
            start_time = time.time()
            sim_result = run_kmc_simulation(
                transition_rates=rates_copy,
                initial_state=init_st_idx,
                max_time=max_time,
                num_trajectories=num_trajectories,
                recording_interval=recording_interval,
                chunk_size=chunk_size,
                fmo_block_labels=fmo_block_labels,
                rc_indices=rc_indices,
                nonrad_indices=nonrad_indices,
            )
            time_points, avg_occ, avg_rc, num_trajectories = sim_result
            monomer_sim_result = (time_points.tolist(), avg_occ, avg_rc, num_trajectories)
            monomer_final_eff = avg_rc[-1, :].sum()
            monomeric_scenarios = [(monomer_subset, monomer_final_eff, monomer_sim_result)]
            subset_labels = [df_index[x] for x in monomer_subset]
            logger.log_simulation_result(
                init_state_name=init_st_name,
                subset=monomer_subset,
                subset_labels=subset_labels,
                final_eff=monomer_final_eff,
                duration=time.time() - start_time,
            )

            # Plot monomeric scenarios
            plotter.plot_monomeric_removals(
                monomeric_scenarios,
                init_st_idx,
                f" ({init_st_name})",
                show_plot=True,
            )

            logger.log_separator()

        logging.info("Main simulation workflow completed.")
        logger.log_separator()
    finally:
        logger.close()  # Ensure the logger is closed to release the file


if __name__ == "__main__":
    config = {
        "output_dir": r"C:\Users\Zane\Desktop\KMC_efficiency simulations\figures",
        "data_file": r"C:\Users\Zane\Desktop\KMC_efficiency simulations\fmo_and_rc_rate_matrix_reversible.xlsx",
        "fmo_rc_labels": ("FMO1->RC", "FMO2->RC", "FMO3->RC"),
        "init_state_names": [
            "FMO1_BCL_371", "FMO1_BCL_376", "FMO1_BCL_378",
            "FMO2_BCL_371", "FMO2_BCL_376", "FMO2_BCL_378",
            "FMO3_BCL_371", "FMO3_BCL_376", "FMO3_BCL_378"
        ],
        "k": 1,
        "sites_to_remove": ["BCL_373", "BCL_374"],
        "num_trajectories": 100000,
        "max_time": 2000.0,
        "recording_interval": 0.5,
        "chunk_size": 2000,
    }
    main(config)