"""Locating the bundled sample dataset."""

from pathlib import Path

SAMPLE_DATA_FILE = "EL_temp.csv"


def sample_data_path():
    """Path to the bundled East Lansing daily weather CSV.

    Ships inside the package, so this resolves the same way whether tmapper
    was pip-installed or is being run from a clone -- code that hard-codes
    ``"sampledata/EL_temp.csv"`` only works from the repository root.

    Note this is the *full* historical record (57709 daily rows). Building a
    network needs a slice of it: the pairwise distance matrix is O(N^2), so
    the whole thing would ask for tens of GB. See the Quickstart.
    """
    return Path(__file__).resolve().parent / "sampledata" / SAMPLE_DATA_FILE
