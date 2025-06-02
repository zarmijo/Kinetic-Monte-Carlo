import itertools
import logging
from typing import List

import numpy as np
from numpy.typing import NDArray


def remove_sites(transition_rates: NDArray[np.float64], sites_to_remove: List[int]) -> None:
    """Remove specified sites from the transition rates matrix by setting their rows and columns to zero.

    Args:
        transition_rates: Square matrix of transition rates between states, modified in-place.
        sites_to_remove: List of site indices to remove.
    """
    sites = np.array(sites_to_remove, dtype=np.intp)
    transition_rates[sites, :] = 0.0
    transition_rates[:, sites] = 0.0
    logging.debug("Removed sites: %s", sites_to_remove)


def choose_k_sites_to_remove(site_list: List[int], k: int) -> List[List[int]]:
    """Generate all possible combinations of k sites to remove from the given site list.

    Args:
        site_list: List of site indices to choose from.
        k: Number of sites to select for removal.

    Returns:
        List of lists, where each inner list is a combination of k site indices.
        Returns an empty list if k is negative or exceeds the length of site_list.
    """
    if k > len(site_list) or k < 0:
        return []
    return [list(combo) for combo in itertools.combinations(site_list, k)]


def remove_site_from_each_monomer(
    transition_rates: NDArray[np.float64],
    site_patterns: List[str],
    df_index: List[str],
) -> List[int]:
    """Remove sites from each monomer matching the given patterns and update the transition rates matrix.

    Args:
        transition_rates: Square matrix of transition rates between states, modified in-place.
        site_patterns: List of site pattern names to match (e.g., ['site1', 'site2']).
        df_index: List of site names in the DataFrame index.

    Returns:
        List of indices of the sites that were removed.
    """
    sites_to_remove = []
    monomers = ["FMO1", "FMO2", "FMO3"]

    for pattern in site_patterns:
        for monomer in monomers:
            site_name = f"{monomer}_{pattern}"
            if site_name in df_index:
                site_idx = df_index.index(site_name)
                if site_idx not in sites_to_remove:
                    sites_to_remove.append(site_idx)
            else:
                logging.warning("Site %s not found in DataFrame index.", site_name)

    if sites_to_remove:
        remove_sites(transition_rates, sites_to_remove)
        logging.info(
            "Removed sites from each monomer: %s",
            [df_index[idx] for idx in sites_to_remove],
        )
    else:
        logging.warning(
            "No sites matching patterns %s found in any monomer.", site_patterns
        )

    return sites_to_remove