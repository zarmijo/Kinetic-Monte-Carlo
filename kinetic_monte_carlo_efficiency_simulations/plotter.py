import logging
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from numpy.typing import NDArray


class FMOPlotter:
    """Handles plotting of FMO simulation scenarios for monomeric and combinatorial removals.

    Attributes:
        df_index: List of state names for plotting.
        fmo_rc_labels: Labels for reaction center transitions.
        output_dir: Directory to save output figures.
        fig: Matplotlib figure object.
        axes: Matplotlib axes or array of axes.
        handles: Plot line handles for the legend.
        labels: Plot labels for the legend.
    """

    def __init__(
        self,
        df_index: List[str],
        fmo_rc_labels: Tuple[str, ...] = ("FMO1->RC", "FMO2->RC", "FMO3->RC"),
        output_dir: Optional[str] = None,
    ) -> None:
        """Initialize the FMOPlotter with simulation parameters.

        Args:
            df_index: List of state names for plotting.
            fmo_rc_labels: Labels for reaction center transitions.
            output_dir: Directory to save figures (defaults to specified path).
        """
        self.df_index = df_index
        self.fmo_rc_labels = fmo_rc_labels
        self.output_dir = output_dir or r"C:\Users\Zane\Desktopindigo Desktop\KMC_efficiency simulations\figures"
        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[plt.Axes | NDArray[plt.Axes]] = None
        self.handles: List[plt.Line2D] = []
        self.labels: List[str] = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_figure(self, n_rows: int, n_cols: int, figsize: Optional[Tuple[int, int]] = None) -> None:
        """Set up a Matplotlib figure with a grid of subplots.

        Args:
            n_rows: Number of rows in the subplot grid.
            n_cols: Number of columns in the subplot grid.
            figsize: Figure size as (width, height). Defaults to dynamic sizing.

        Raises:
            RuntimeError: If figure creation fails.
        """
        if figsize is None:
            figsize = (12 * n_cols, 8 * n_rows)
        try:
            self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows * n_cols > 1:
                self.axes = np.asarray(self.axes).flatten()
            else:
                self.axes = np.array([self.axes])
        except ValueError as e:
            logging.error("Failed to set up figure: %s", e)
            raise RuntimeError(f"Figure setup failed: {e}")

    def _plot_state_occupancies(
        self,
        ax: plt.Axes,
        time_points: List[float],
        avg_occupancy: NDArray[np.float64],
    ) -> None:
        """Plot state occupancies on a given axis.

        Args:
            ax: Matplotlib axis to plot on.
            time_points: List of time points for the x-axis.
            avg_occupancy: Array of occupancy values for each state.
        """
        color_map = get_cmap("turbo", avg_occupancy.shape[1])
        for state_idx, site_name in enumerate(self.df_index):
            if site_name.startswith("FMO") or site_name == "Dissipative":
                clean_site_name = "Dissipative Loss" if site_name == "Dissipative" else site_name.replace("_", " ")
                line, = ax.plot(
                    time_points,
                    avg_occupancy[:, state_idx],
                    color=color_map(state_idx),
                    linewidth=1.5,
                    label=clean_site_name,
                )
                if clean_site_name not in self.labels:
                    logging.info(f"Adding state occupancy label to legend: {clean_site_name}")
                    self.handles.append(line)
                    self.labels.append(clean_site_name)

    def _plot_rc_occupancies(
        self,
        ax: plt.Axes,
        time_points: List[float],
        rc_occupancy_block: NDArray[np.float64],
    ) -> None:
        """Plot reaction center occupancies on a given axis.

        Args:
            ax: Matplotlib axis to plot on.
            time_points: List of time points for the x-axis.
            rc_occupancy_block: Array of reaction center occupancy values.
        """
        rc_colors = ["black", "dimgray", "lightgray"]
        for b_id in range(rc_occupancy_block.shape[1]):
            clean_rc_label = (
                self.fmo_rc_labels[b_id].replace("_", " ")
                if b_id < len(self.fmo_rc_labels)
                else f"Block {b_id}->RC"
            )
            line, = ax.plot(
                time_points,
                rc_occupancy_block[:, b_id],
                color=rc_colors[b_id] if b_id < len(rc_colors) else "black",
                linewidth=2.5,
                linestyle="--",
                label=clean_rc_label,
            )
            if clean_rc_label not in self.labels:
                logging.info(f"Adding RC transfer label to legend: {clean_rc_label}")
                self.handles.append(line)
                self.labels.append(clean_rc_label)
            else:
                logging.info(f"RC transfer label already in legend: {clean_rc_label}")

    def _plot_total_rc_transfer(
        self,
        ax: plt.Axes,
        time_points: List[float],
        rc_occupancy_block: NDArray[np.float64],
    ) -> None:
        """Plot the total reaction center transfer by summing all RC occupancies.

        Args:
            ax: Matplotlib axis to plot on.
            time_points: List of time points for the x-axis.
            rc_occupancy_block: Array of reaction center occupancy values.
        """
        if rc_occupancy_block.size == 0 or rc_occupancy_block.shape[1] == 0:
            logging.warning("Empty rc_occupancy_block; skipping total RC transfer plot.")
            return
        total_rc = np.sum(rc_occupancy_block, axis=1)
        clean_label = "Total RC Transfer"
        line, = ax.plot(
            time_points,
            total_rc,
            color="blue",
            linewidth=2.0,
            linestyle=":",
            label=clean_label,
        )
        if clean_label not in self.labels:
            logging.info(f"Adding Total RC Transfer label to legend: {clean_label}")
            self.handles.append(line)
            self.labels.append(clean_label)

    def _plot_dissipative_rate(
        self,
        ax: plt.Axes,
        time_points: List[float],
        avg_occupancy: NDArray[np.float64],
        subset: List[int],
        final_occ: float,
        title_prefix: str = "",
    ) -> None:
        """Plot the dissipative rate on a given axis.

        Args:
            ax: Matplotlib axis to plot on.
            time_points: List of time points for the x-axis.
            avg_occupancy: Array of occupancy values for each state.
            subset: List of state indices to include in the plot title.
            final_occ: Final reaction center efficiency for the plot title.
            title_prefix: Optional prefix for the plot title.
        """
        dissipative_idx = next((i for i, name in enumerate(self.df_index) if name == "Dissipative"), None)
        if dissipative_idx is None:
            logging.warning("No 'Dissipative' state found in df_index; skipping dissipative rate plot.")
            return

        line, = ax.plot(
            time_points,
            avg_occupancy[:, dissipative_idx],
            color="red",
            linewidth=2.0,
            label="Dissipative Loss",
        )
        if "Dissipative Loss" not in self.labels:
            logging.info(f"Adding Dissipative Loss label to legend: Dissipative Loss")
            self.handles.append(line)
            self.labels.append("Dissipative Loss")

        subset_str = (
            "None (Full Network)"
            if not subset
            else ", ".join(self.df_index[s].replace("_", " ") for s in subset)
        )
        ax.set_title(
            f"{title_prefix}{subset_str}\nFinal RC Efficiency = {final_occ:.3f}",
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("Time (fs)", fontsize=12)
        ax.set_ylabel("Dissipative Occupancy", fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.7)

    def _plot_scenario(
        self,
        ax: plt.Axes,
        subset: List[int],
        final_occ: float,
        sim_result: Tuple[List[float], NDArray[np.float64], NDArray[np.float64], int],
        title_prefix: str = "",
    ) -> None:
        """Plot a single simulation scenario on a given axis.

        Args:
            ax: Matplotlib axis to plot on.
            subset: List of state indices to include in the plot.
            final_occ: Final reaction center efficiency.
            sim_result: Tuple of time points, state occupancies, RC occupancies, and trajectory count.
            title_prefix: Optional prefix for the plot title.
        """
        time_points, avg_occupancy, rc_occupancy_block, _ = sim_result
        self._plot_state_occupancies(ax, time_points, avg_occupancy)
        self._plot_rc_occupancies(ax, time_points, rc_occupancy_block)
        self._plot_total_rc_transfer(ax, time_points, rc_occupancy_block)
        subset_str = (
            "None (Full Network)"
            if not subset
            else ", ".join(self.df_index[s].replace("_", " ") for s in subset)
        )
        ax.set_title(
            f"{title_prefix}{subset_str}\nFinal RC Efficiency = {final_occ:.3f}",
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("Time (fs)", fontsize=12)
        ax.set_ylabel("Exciton Occupancy", fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.7)

    def _finalize_figure(
        self,
        init_state: int,
        title_extra: str,
        plot_type: str,
        show_plot: bool = False,  # Changed to False for artifact generation
    ) -> None:
        """Finalize and save the figure with a single legend at the bottom.

        Args:
            init_state: Initial state index for the simulation.
            title_extra: Additional text for the figure title.
            plot_type: Type of plot for the filename.
            show_plot: Whether to display the plot (default False for artifact generation).

        Raises:
            RuntimeError: If figure finalization fails.
        """
        if self.fig is None or self.axes is None:
            raise RuntimeError("Figure or axes not initialized.")

        try:
            plt.subplots_adjust(
                left=0.05, right=0.95, top=0.90, bottom=0.30, hspace=0.5, wspace=0.3
            )
            # Group legend entries: chromophores, reaction centers, Total RC, Dissipative Loss
            chromophores = [
                (label, handle) for label, handle in zip(self.labels, self.handles)
                if label.startswith("FMO") and "->" not in label
            ]
            rc_transfers = [
                (label, handle) for label, handle in zip(self.labels, self.handles)
                if "->RC" in label.replace(" ", "")
            ]
            total_rc = [
                (label, handle) for label, handle in zip(self.labels, self.handles)
                if label == "Total RC Transfer"
            ]
            dissipative = [
                (label, handle) for label, handle in zip(self.labels, self.handles)
                if label == "Dissipative Loss"
            ]
            # Log the grouped labels for debugging
            logging.info(f"Chromophores: {[label for label, _ in chromophores]}")
            logging.info(f"RC Transfers: {[label for label, _ in rc_transfers]}")
            logging.info(f"Total RC: {[label for label, _ in total_rc]}")
            logging.info(f"Dissipative: {[label for label, _ in dissipative]}")
            # Sort each group alphabetically
            chromophores.sort(key=lambda x: x[0])
            rc_transfers.sort(key=lambda x: x[0])
            # Combine in specified order: chromophores, RC transfers, Total RC, dissipative
            sorted_pairs = chromophores + rc_transfers + total_rc + dissipative
            if sorted_pairs:
                sorted_labels, sorted_handles = zip(*sorted_pairs)
                self.fig.legend(
                    sorted_handles,
                    sorted_labels,
                    fontsize=8,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                    ncol=min(len(sorted_labels), 10),
                    title="Chromophores, Reaction Centers, Total RC, and Dissipative Loss",
                    title_fontsize=10,
                    frameon=True,
                    borderaxespad=0.5,
                )
            clean_title_extra = title_extra.replace("_", " ").strip()
            self.fig.suptitle(
                f"KMC for Initial State #{init_state + 1}{clean_title_extra} - {plot_type}",
                fontsize=16,
                y=0.98,
            )
            safe_title_extra = "".join(c if c.isalnum() or c == "_" else "_" for c in title_extra)
            fig_filename = os.path.join(
                self.output_dir,
                f"figure_init_{init_state + 1}_{plot_type}{safe_title_extra}.png",
            )
            self.fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
            logging.info("Figure saved as %s", fig_filename)
            if show_plot:
                plt.show()
        except (OSError, ValueError) as e:
            logging.error("Failed to finalize figure: %s", e)
            raise RuntimeError(f"Figure finalization failed: {e}")
        finally:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self.handles = []
            self.labels = []

    def plot_monomeric_removals(
        self,
        scenario_data: List[
            Tuple[
                List[int],
                float,
                Tuple[List[float], NDArray[np.float64], NDArray[np.float64], int],
            ]
        ],
        init_state: int,
        title_extra: str = "",
        show_plot: bool = False,  # Changed to False for artifact generation
    ) -> None:
        """Plot a single monomeric removal scenario in a 1x1 layout.

        Args:
            scenario_data: List of tuples containing subset indices, efficiency, and simulation results.
            init_state: Initial state index for the simulation.
            title_extra: Additional text for the figure title.
            show_plot: Whether to display the plot (default False for artifact generation).
        """
        if not scenario_data:
            logging.warning("No monomeric scenarios to plot.")
            return

        # Select a single scenario: prefer full network if available, else first scenario
        full_network = next((x for x in scenario_data if not x[0]), None)
        plot_scenario = full_network if full_network else scenario_data[0]

        # Set up figure with 1x1 layout
        self._setup_figure(n_rows=1, n_cols=1)

        # Plot the scenario
        axes_array = np.atleast_1d(self.axes)
        self._plot_scenario(axes_array[0], *plot_scenario, title_prefix="")

        self._finalize_figure(init_state, title_extra, "Monomeric_Removals", show_plot)

    def plot_combinatorial_removals(
        self,
        scenario_data: List[
            Tuple[
                List[int],
                float,
                Tuple[List[float], NDArray[np.float64], NDArray[np.float64], int],
            ]
        ],
        init_state: int,
        title_extra: str = "",
        show_plot: bool = False,  # Changed to False for artifact generation
        max_plots: int = 6,
    ) -> None:
        """Plot combinatorial removal scenarios in a 2x3 grid layout.

        Args:
            scenario_data: List of tuples containing subset indices, efficiency, and simulation results.
            init_state: Initial state index for the simulation.
            title_extra: Additional text for the figure title.
            show_plot: Whether to display the plot (default False for artifact generation).
            max_plots: Maximum number of scenarios to plot.
        """
        if not scenario_data:
            logging.warning("No combinatorial scenarios to plot.")
            return

        # Select scenarios: sort by efficiency, include full network, remove duplicates
        sorted_scenarios = sorted(scenario_data, key=lambda x: x[1])
        full_network = next((x for x in sorted_scenarios if not x[0]), None)
        selected = sorted_scenarios[:5]
        if full_network and full_network not in selected:
            selected.append(full_network)

        unique_scenarios = []
        seen_subsets = set()
        for scenario in selected:
            subset_key = tuple(sorted(scenario[0]))
            if subset_key not in seen_subsets:
                seen_subsets.add(subset_key)
                unique_scenarios.append(scenario)
        unique_scenarios.sort(key=lambda x: (bool(x[0]), x[1]))
        plot_scenarios = unique_scenarios[:max_plots]

        # Set up figure
        n_cols = 3
        n_rows = (len(plot_scenarios) + n_cols - 1) // n_cols
        self._setup_figure(n_rows, n_cols)

        # Plot scenarios
        axes_array = np.atleast_1d(self.axes)
        for ax_idx, scenario in enumerate(plot_scenarios):
            ax = axes_array[ax_idx]
            self._plot_scenario(ax, *scenario, title_prefix="")

        # Hide unused subplots
        for ax in axes_array[len(plot_scenarios):]:
            ax.set_visible(False)

        self._finalize_figure(init_state, title_extra, "Combinatorial_Removals", show_plot)