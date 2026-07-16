"""
ibm_amlsim_loader.py
--------------------
Concrete loader for the IBM AMLSim dataset (HI-Small variant).

Dataset source
--------------
Kaggle: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Expected file
-------------
``<data_dir>/HI-Small_Trans.csv``

Transaction CSV schema
----------------------
Timestamp           : datetime  – when the transaction occurred
From Bank           : int       – numeric code of the originating bank
Account             : str       – hexadecimal sender account ID
To Bank             : int       – numeric code of the receiving bank
Account.1           : str       – hexadecimal receiver account ID
Amount Received     : float     – amount in the receiving currency
Receiving Currency  : str       – e.g. "US Dollar", "Euro"
Amount Paid         : float     – amount in the paying currency
Payment Currency    : str       – e.g. "US Dollar", "Bitcoin"
Payment Format      : str       – e.g. "Cheque", "ACH", "Wire", "Credit Card"
Is Laundering       : int       – 0 (normal) or 1 (fraudulent)
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from dataset_loader import AMLDatasetLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR: str = os.path.join(
    os.path.dirname(__file__), "data", "ibm_amlsim"
)
_FILENAME: str = "HI-Small_Trans.csv"

_FRAUD_COL: str = "Is Laundering"
_SOURCE_COL: str = "Account"
_DEST_COL: str = "Account.1"
_AMOUNT_COL: str = "Amount Paid"


class IBMAMLSimLoader(AMLDatasetLoader):
    """Loader for the IBM AMLSim dataset (HI-Small_Trans.csv).

    Parameters
    ----------
    data_dir:
        Directory containing the CSV file.  Defaults to
        ``Code/data/ibm_amlsim/`` relative to this file.
    filename:
        CSV filename inside *data_dir*.  Defaults to ``HI-Small_Trans.csv``.
    """

    def __init__(
        self,
        data_dir: str = _DATA_DIR,
        filename: str = _FILENAME,
    ) -> None:
        super().__init__(data_dir)
        self._filename = filename

    # ------------------------------------------------------------------
    # AMLDatasetLoader interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ``HI-Small_Trans.csv`` into the internal DataFrame."""
        path = os.path.join(self._data_dir, self._filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"IBM AMLSim CSV not found at:\n  {path}\n"
                "Download from: https://www.kaggle.com/datasets/"
                "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
            )

        print(f"Loading IBM AMLSim from: {path}")
        self._transactions = pd.read_csv(path)

        # Rename duplicate 'Account' column (sender/receiver) for clarity.
        # read_csv already names them 'Account' and 'Account.1'.
        # Ensure the fraud column is integer.
        self._transactions[_FRAUD_COL] = (
            self._transactions[_FRAUD_COL].astype(int)
        )
        print(f"  Loaded {len(self._transactions):,} rows.")

    def get_features(self) -> dict[str, dict[str, Any]]:
        """Return metadata for every column in the IBM AMLSim dataset."""
        df = self.get_transactions()
        return {
            "Timestamp": {
                "dtype": str(df["Timestamp"].dtype),
                "description": "Date and time of the transaction",
                "possible_values": None,
            },
            "From Bank": {
                "dtype": str(df["From Bank"].dtype),
                "description": "Numeric identifier of the originating bank",
                "possible_values": sorted(df["From Bank"].dropna().unique()[:5].tolist()),
            },
            "Account": {
                "dtype": str(df["Account"].dtype),
                "description": "Hexadecimal ID of the sender account (node)",
                "possible_values": None,
            },
            "To Bank": {
                "dtype": str(df["To Bank"].dtype),
                "description": "Numeric identifier of the receiving bank",
                "possible_values": sorted(df["To Bank"].dropna().unique()[:5].tolist()),
            },
            "Account.1": {
                "dtype": str(df["Account.1"].dtype),
                "description": "Hexadecimal ID of the receiver account (node)",
                "possible_values": None,
            },
            "Amount Received": {
                "dtype": str(df["Amount Received"].dtype),
                "description": "Transaction amount in the receiving currency",
                "possible_values": None,
            },
            "Receiving Currency": {
                "dtype": str(df["Receiving Currency"].dtype),
                "description": "Currency received by the destination account",
                "possible_values": df["Receiving Currency"].dropna().unique()[:6].tolist(),
            },
            "Amount Paid": {
                "dtype": str(df["Amount Paid"].dtype),
                "description": "Transaction amount in the paying currency",
                "possible_values": None,
            },
            "Payment Currency": {
                "dtype": str(df["Payment Currency"].dtype),
                "description": "Currency used by the sender",
                "possible_values": df["Payment Currency"].dropna().unique()[:6].tolist(),
            },
            "Payment Format": {
                "dtype": str(df["Payment Format"].dtype),
                "description": "Payment method / instrument",
                "possible_values": df["Payment Format"].dropna().unique().tolist(),
            },
            "Is Laundering": {
                "dtype": str(df["Is Laundering"].dtype),
                "description": "Ground-truth fraud label: 1 = money-laundering, 0 = normal",
                "possible_values": [0, 1],
            },
        }

    def build_graph(self) -> nx.DiGraph:
        """Build and cache a directed account-transaction graph."""
        return self._build_graph_from_df(
            source_col=_SOURCE_COL,
            dest_col=_DEST_COL,
            fraud_col=_FRAUD_COL,
            amount_col=_AMOUNT_COL,
        )

    def plot_subgraph(
        self,
        seed_node: str | None = None,
        max_nodes: int = 50,
        type_of_illicit: str = "ML",
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot a small subgraph (~15 nodes) via fast DataFrame BFS.

        Does NOT require :meth:`build_graph` to have been called.
        Red edges = fraudulent, blue = normal.
        """
        # IBM AMLSim doesn't distinguish, fallback to default fraud column
        return self._fast_subgraph_from_df(
            source_col=_SOURCE_COL,
            dest_col=_DEST_COL,
            fraud_col=_FRAUD_COL,
            amount_col=_AMOUNT_COL,
            seed_node=seed_node,
            max_nodes=max_nodes,
            ax=ax,
            title=title or f"IBM AMLSim – {type_of_illicit} subgraph",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fraud_column(self) -> str:
        return _FRAUD_COL

    def _source_column(self) -> str:
        return _SOURCE_COL

    def _dest_column(self) -> str:
        return _DEST_COL
