import logging
import os
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


class FMOLogger:
    """Logger for scientific simulations with file and console output.

    This class provides a configurable logging setup for simulations, supporting both
    console and file output with customizable formats and levels. It includes methods
    for logging matrices and simulation results with detailed formatting.
    """

    def __init__(
        self,
        output_dir: str,
        log_file_name: str = "simulation.log",
        log_level: int = logging.DEBUG,
        log_format: str = "%(asctime)s [%(levelname)s] %(message)s",
        log_date_format: str = "%Y-%m-%d %H:%M:%S",
        log_to_console: bool = True,
        log_to_file: bool = True,
    ) -> None:
        """Initialize the logger with specified output settings.

        Args:
            output_dir: Directory where log files will be saved.
            log_file_name: Name of the log file (default: "simulation.log").
            log_level: Logging level (default: logging.DEBUG).
            log_format: Format string for log messages (default: includes timestamp,
                level, and message).
            log_date_format: Format for timestamps in logs (default: ISO-like format).
            log_to_console: Whether to log to console (default: True).
            log_to_file: Whether to log to file (default: True).

        Raises:
            OSError: If the output directory cannot be created.
        """
        self.output_dir: str = output_dir
        self.log_file_name: str = log_file_name
        self.log_level: int = log_level
        self.log_format: str = log_format
        self.log_date_format: str = log_date_format
        self.log_to_console: bool = log_to_console
        self.log_to_file: bool = log_to_file
        self.is_initialized: bool = False
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure the logging system with handlers for console and file.

        Sets up a logger with the specified level, format, and handlers. Clears any
        existing handlers to avoid duplication.

        Raises:
            OSError: If the log file cannot be created or written to.
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(self.log_level)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter(self.log_format, self.log_date_format)

        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.log_to_file:
            log_file = os.path.join(self.output_dir, self.log_file_name)
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.is_initialized = True

    def close(self) -> None:
        """Close all logging handlers and release file resources.

        This method ensures that all handlers are properly closed and removed,
        allowing the log file to be deleted or modified.
        """
        if not self.is_initialized:
            return

        for handler in self.logger.handlers[:]:  # Copy to avoid modifying while iterating
            try:
                handler.flush()  # Ensure all logs are written
                handler.close()  # Close the handler
                self.logger.removeHandler(handler)  # Remove it from the logger
            except Exception as e:
                print(f"Error closing handler {handler}: {e}")

        self.is_initialized = False

    def log_separator(self, separator: str = "=", length: int = 80) -> None:
        """Log a separator line of specified character and length.

        Args:
            separator: Character to use for the separator (default: "=").
            length: Number of characters in the separator line (default: 80).

        Raises:
            RuntimeError: If the logger is not initialized.
        """
        if not self.is_initialized:
            raise RuntimeError("Logger must be initialized before use.")
        logging.info(separator * length)

    def log_matrix(
        self,
        matrix: NDArray[np.float64],
        labels: Optional[List[str]] = None,
        max_rows: int = 10,
        title: str = "Transition Rate Matrix Overview",
    ) -> None:
        """Log a formatted overview of a matrix with statistics and excerpt.

        Args:
            matrix: 2D NumPy array to log.
            labels: List of labels for rows/columns (default: None, uses indices).
            max_rows: Maximum number of rows to display (default: 10).
            title: Title for the matrix log entry (default: "Transition Rate Matrix
                Overview").

        Raises:
            RuntimeError: If the logger is not initialized.
        """
        if not self.is_initialized:
            raise RuntimeError("Logger must be initialized before use.")

        n_rows, n_cols = matrix.shape
        logging.info(title)
        logging.info("Matrix size: %d rows x %d columns", n_rows, n_cols)

        non_zero_count = np.count_nonzero(matrix)
        sparsity = 1 - non_zero_count / (n_rows * n_cols)
        value_min = np.min(matrix[matrix > 0]) if non_zero_count > 0 else 0
        value_max = np.max(matrix)
        logging.info("Sparsity: %.2f%%", sparsity * 100)
        logging.info("Non-zero value range: %.3e to %.3e", value_min, value_max)

        if labels is None:
            header = "    " + "  ".join(f"Col {j:2d}" for j in range(n_cols))
        else:
            header = "    " + "  ".join(f"{label[:6]:6s}" for label in labels[:n_cols])
        logging.info("Matrix Excerpt (up to %d rows):", max_rows)
        logging.info(header)
        logging.info("-" * len(header))

        for i in range(min(n_rows, max_rows)):
            row_values = matrix[i]
            row_str = "  ".join(f"{val:10.3e}" for val in row_values)
            row_label = f"Row {i:2d}" if labels is None else labels[i][:10]
            logging.info(f"{row_label:10s}: {row_str}")

        if n_rows > max_rows:
            logging.info("...")

    def log_simulation_result(
        self,
        init_state_name: str,
        subset: List[int],
        subset_labels: List[str],
        final_eff: float,
        duration: float,
    ) -> None:
        """Log the result of a simulation run.

        Args:
            init_state_name: Name of the initial state.
            subset: List of indices of sites removed.
            subset_labels: Labels for the removed sites.
            final_eff: Final efficiency of the simulation.
            duration: Duration of the simulation in seconds.

        Raises:
            RuntimeError: If the logger is not initialized.
        """
        if not self.is_initialized:
            raise RuntimeError("Logger must be initialized before use.")

        subset_str = "None (Full Network)" if not subset else ", ".join(subset_labels)
        logging.info(
            "Init=%s, Subset=%s => RC efficiency=%.4f",
            init_state_name,
            subset_str,
            final_eff,
        )
        logging.info("Simulation completed in %.2f seconds.", duration)