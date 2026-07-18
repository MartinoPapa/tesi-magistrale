"""
saml_d_loader.py
----------------
Concrete loader for the SAML-D (Synthetic Anti-Money Laundering Dataset).

Dataset source
--------------
Kaggle    : https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
Paper     : Bournemouth University / arxiv

SAML-D contains ~9.5 million transactions across 28 typologies
(11 normal + 17 suspicious). Suspicious transactions represent ~0.10 %
of the dataset.

Expected file
-------------
``<data_dir>/SAML-D.csv``

Transaction CSV schema  (12 columns)
------------------------------------
Time                 : str   – transaction time (HH:MM:SS)
Date                 : str   – transaction date (DD/MM/YYYY or similar)
Sender_account       : str   – sender account ID (node)
Receiver_account     : str   – receiver account ID (node)
Amount               : float – transaction amount
Payment_type         : str   – e.g. "ACH", "Wire", "Cash", "Credit Card"
Sender_bank_location : str   – country/region of the sender bank
Receiver_bank_location: str  – country/region of the receiver bank
Payment_currency     : str   - currency of the sender
Received_currency    : str   - currency received by destination
Payment_type         : str   - payment instrument used
Is_laundering        : int   – 0 (normal) or 1 (fraudulent/laundering)
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
    os.path.dirname(__file__), "..", "data", "saml_d"
)
_FILENAME: str = "SAML-D.csv"

_FRAUD_COL: str = "Is_laundering"
_SOURCE_COL: str = "Sender_account"
_DEST_COL: str = "Receiver_account"
_AMOUNT_COL: str = "Amount"


class SAMLDLoader(AMLDatasetLoader):
    """Loader for the SAML-D synthetic AML dataset.

    Because the full SAML-D file is ~996 MB, the constructor accepts an
    optional *nrows* parameter for quickly loading a subset during
    exploration.

    Parameters
    ----------
    data_dir:
        Directory containing the CSV file.  Defaults to
        ``Code/data/saml_d/`` relative to this file.
    filename:
        CSV filename inside *data_dir*.  Defaults to ``SAML-D.csv``.
    nrows:
        If not ``None``, only this many rows are read (useful for fast
        prototyping on large files).
    """

    def __init__(
        self,
        data_dir: str = _DATA_DIR,
        filename: str | None = None,
        nrows: int | None = None,
    ) -> None:
        super().__init__(data_dir)
        self._filename: str = (
            filename
            if filename is not None
            else self._resolve_filename(data_dir, _FILENAME)
        )
        self._nrows = nrows

    # ------------------------------------------------------------------
    # AMLDatasetLoader interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the SAML-D CSV into the internal DataFrame."""
        path = os.path.join(self._data_dir, self._filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SAML-D CSV not found at:\n  {path}\n"
                f"Expected file: {self._filename}"
            )

        print(f"Loading SAML-D from: {path}")
        if self._nrows is not None:
            print(f"  (reading first {self._nrows:,} rows only)")
        self._transactions = pd.read_csv(path, nrows=self._nrows)
        self._transactions[_FRAUD_COL] = (
            self._transactions[_FRAUD_COL].astype(int)
        )
        print(f"  Loaded {len(self._transactions):,} rows.")

    def get_features(self) -> dict[str, dict[str, Any]]:
        """Return metadata for every column in the SAML-D dataset."""
        df = self.get_transactions()
        return {
            "Time": {
                "dtype": str(df["Time"].dtype),
                "description": "Time of the transaction (HH:MM:SS)",
                "possible_values": None,
            },
            "Date": {
                "dtype": str(df["Date"].dtype),
                "description": "Date of the transaction",
                "possible_values": None,
            },
            "Sender_account": {
                "dtype": str(df["Sender_account"].dtype),
                "description": "Sender account identifier (node)",
                "possible_values": None,
            },
            "Receiver_account": {
                "dtype": str(df["Receiver_account"].dtype),
                "description": "Receiver account identifier (node)",
                "possible_values": None,
            },
            "Amount": {
                "dtype": str(df["Amount"].dtype),
                "description": "Transaction amount",
                "possible_values": None,
            },
            "Payment_type": {
                "dtype": str(df["Payment_type"].dtype),
                "description": "Payment instrument / method",
                "possible_values": df["Payment_type"].dropna().unique().tolist(),
            },
            "Sender_bank_location": {
                "dtype": str(df["Sender_bank_location"].dtype),
                "description": "Country or region of the sender's bank",
                "possible_values": df["Sender_bank_location"].dropna().unique()[:6].tolist(),
            },
            "Receiver_bank_location": {
                "dtype": str(df["Receiver_bank_location"].dtype),
                "description": "Country or region of the receiver's bank",
                "possible_values": df["Receiver_bank_location"].dropna().unique()[:6].tolist(),
            },
            "Payment_currency": {
                "dtype": str(df["Payment_currency"].dtype),
                "description": "Currency used by the sender",
                "possible_values": df["Payment_currency"].dropna().unique()[:6].tolist(),
            },
            "Received_currency": {
                "dtype": str(df["Received_currency"].dtype),
                "description": "Currency received by the destination account",
                "possible_values": df["Received_currency"].dropna().unique()[:6].tolist(),
            },
            "Payment_type": {
                "dtype": str(df["Payment_type"].dtype),
                "description": "Payment instrument or method used",
                "possible_values": df["Payment_type"].dropna().unique()[:6].tolist(),
            },
            "Is_laundering": {
                "dtype": str(df["Is_laundering"].dtype),
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
        # SAML-D does not distinguish, fallback to default fraud column
        return self._fast_subgraph_from_df(
            source_col=_SOURCE_COL,
            dest_col=_DEST_COL,
            fraud_col=_FRAUD_COL,
            amount_col=_AMOUNT_COL,
            seed_node=seed_node,
            max_nodes=max_nodes,
            ax=ax,
            title=title or f"SAML-D – {type_of_illicit} subgraph",
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
