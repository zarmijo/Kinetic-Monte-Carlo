import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(fastmath=True, parallel=True)
def run_kmc_simulation(
    transition_rates: NDArray[np.float64],
    initial_state: int,
    max_time: float,
    num_trajectories: int,
    recording_interval: float,
    chunk_size: int,
    fmo_block_labels: NDArray[np.int64],
    rc_indices: NDArray[np.int64],
    nonrad_indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Run a Kinetic Monte Carlo (KMC) simulation for a given transition rate matrix.

    Args:
        transition_rates: Square matrix of transition rates between states.
        initial_state: Starting state index for all trajectories.
        max_time: Maximum simulation time.
        num_trajectories: Number of trajectories to simulate.
        recording_interval: Time interval for recording state occupancies.
        chunk_size: Number of trajectories to process in each parallel chunk.
        fmo_block_labels: Array mapping states to block IDs for reaction coordinate tracking.
        rc_indices: Indices of reaction coordinate (RC) states.
        nonrad_indices: Indices of non-radiative states.

    Returns:
        A tuple containing:
        - Time points array.
        - Average state occupancy over all trajectories.
        - Average reaction coordinate block occupancy.
        - Number of trajectories completed.
    """
    # Initialize time points and dimensions
    time_points = np.arange(0.0, max_time + recording_interval, recording_interval)
    num_time_points = time_points.shape[0]
    num_states = transition_rates.shape[0]
    num_blocks = max(np.max(fmo_block_labels) + 1, 0)

    # Initialize output arrays
    occupancy_counts = np.zeros((num_time_points, num_states), dtype=np.float64)
    rc_occupancy_block = np.zeros((num_time_points, num_blocks), dtype=np.float64)

    # Precompute cumulative transition rates
    cumulative_rates = np.zeros_like(transition_rates)
    for i in prange(num_states):
        cumulative_rates[i, 0] = transition_rates[i, 0]
        for j in range(1, num_states):
            cumulative_rates[i, j] = cumulative_rates[i, j - 1] + transition_rates[i, j]
    total_exit_rates = cumulative_rates[:, -1]

    # Mark reaction coordinate and non-radiative states
    is_rc_state = np.zeros(num_states, dtype=np.int64)
    is_nonrad_state = np.zeros(num_states, dtype=np.int64)
    for idx in rc_indices:
        is_rc_state[idx] = 1
    for idx in nonrad_indices:
        is_nonrad_state[idx] = 1

    # Process trajectories in chunks
    trajectories_completed = 0
    while trajectories_completed < num_trajectories:
        current_chunk_size = min(chunk_size, num_trajectories - trajectories_completed)
        for _ in prange(current_chunk_size):
            current_time = 0.0
            current_state = initial_state
            time_idx_start = 0

            while current_time < max_time:
                exit_rate = total_exit_rates[current_state]
                if exit_rate <= 1e-15:
                    for t_idx in range(time_idx_start, num_time_points):
                        occupancy_counts[t_idx, current_state] += 1.0
                    break

                # Calculate time step and next state
                dt = -np.log(np.random.random()) / exit_rate
                next_time = current_time + dt
                time_idx_end = np.searchsorted(time_points, next_time, side="right")

                # Record occupancy for current state
                if time_idx_end >= num_time_points:
                    for t_idx in range(time_idx_start, num_time_points):
                        occupancy_counts[t_idx, current_state] += 1.0
                    break

                for t_idx in range(time_idx_start, time_idx_end):
                    occupancy_counts[t_idx, current_state] += 1.0

                # Determine next state
                random_value = np.random.random() * exit_rate
                next_state = np.searchsorted(cumulative_rates[current_state], random_value, side="left")

                # Handle special state transitions
                if is_rc_state[next_state] == 1:
                    block_id = fmo_block_labels[current_state]
                    if 0 <= block_id < num_blocks:
                        for t_idx in range(time_idx_end, num_time_points):
                            rc_occupancy_block[t_idx, block_id] += 1.0
                    break
                if is_nonrad_state[next_state] == 1:
                    for t_idx in range(time_idx_end, num_time_points):
                        occupancy_counts[t_idx, next_state] += 1.0
                    break

                current_time = next_time
                current_state = next_state
                time_idx_start = time_idx_end

        trajectories_completed += current_chunk_size

    # Compute averages
    avg_occupancy = occupancy_counts / num_trajectories
    avg_rc_occupancy_block = rc_occupancy_block / num_trajectories

    return time_points, avg_occupancy, avg_rc_occupancy_block, num_trajectories