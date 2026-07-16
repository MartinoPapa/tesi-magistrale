"""
amlgentex_loader.py
-------------------
Concrete loader for the **AMLNet** dataset, used as the AMLGentex-equivalent
dataset in this analysis.

Background
----------
AMLGentex (https://github.com/aidotse/AMLGentex) is a *generator framework*
and does not ship a pre-built static CSV. AMLNet is a comparable, freely
available synthetic AML benchmark generated with a similar multi-agent
simulation approach and an overlapping schema (PaySim-compatible + extras).

Dataset source
--------------
Zenodo  : https://zenodo.org/records/21237971  (CC BY-NC 4.0, no auth needed)
Paper   : "AMLNet: A Knowledge-Guided Synthetic Benchmark for Machine
           Learning Evaluation in Anti-Money Laundering"

Expected file
-------------
``<data_dir>/transactions.csv``

Transaction CSV schema
----------------------
step                : int    – sequential simulation step (index)
type                : str    – BPAY, CASH_OUT, DEBIT, EFTPOS, NPP, OSKO,
                               PAYMENT, TRANSFER
amount              : float  – transaction amount (AUD)
category            : str    – housing, food, transport, recreation,
                               healthcare, education, utilities,
                               shell company, property investment,
                               cryptocurrency, other
nameOrig            : str    – originating account/customer ID (node)
nameDest            : str    – destination account/customer/merchant ID (node)
oldbalanceOrg       : float  – originating account balance before transaction
newbalanceOrig      : float  – originating account balance after transaction
isFraud             : int    – 0 or 1 (suspicious/fraudulent label)
isMoneyLaundering   : int    – 0 or 1 (AML-specific label)
laundering_typology : str    – normal, structuring, layering, integration
metadata            : str    – JSON-style metadata (timestamp, location, etc.)
fraud_probability   : float  – risk score (may be empty for some records)
hour                : int    – hour of transaction
day_of_week         : int    – day of week (0=Mon…6=Sun)
day_of_month        : int    – day of month
month               : int    – month number
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
    os.path.dirname(__file__), "data", "amlgentex"
)
_FILENAME: str = "transactions.csv"

# Use the money-laundering label (AML-specific) as the primary fraud label.
_FRAUD_COL: str = "isMoneyLaundering"
_SOURCE_COL: str = "nameOrig"
_DEST_COL: str = "nameDest"
_AMOUNT_COL: str = "amount"

# Columns that we want to load (skip heavy `metadata` JSON blob by default).
_USECOLS: list[str] = [
    "step", "type", "amount", "category",
    "nameOrig", "nameDest",
    "oldbalanceOrg", "newbalanceOrig",
    "isFraud", "isMoneyLaundering", "laundering_typology",
    "hour", "day_of_week", "day_of_month", "month",
]


class AMLGentexLoader(AMLDatasetLoader):
    """Loader for the AMLNet dataset (used as AMLGentex-equivalent).

    AMLNet is a freely downloadable synthetic AML benchmark from Zenodo
    with a PaySim-compatible schema extended with temporal features and
    laundering typology labels.

    Parameters
    ----------
    data_dir:
        Directory containing ``transactions.csv``.  Defaults to
        ``Code/data/amlgentex/`` relative to this file.
    filename:
        CSV filename inside *data_dir*.  Defaults to ``transactions.csv``.
    nrows:
        If not ``None``, only this many rows are read (useful for fast
        prototyping on the full ~1.09 M-row file).
    """

    def __init__(
        self,
        data_dir: str = _DATA_DIR,
        filename: str = _FILENAME,
        nrows: int | None = None,
    ) -> None:
        super().__init__(data_dir)
        self._filename = filename
        self._nrows = nrows

    # ------------------------------------------------------------------
    # AMLDatasetLoader interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the AMLNet transaction CSV into the internal DataFrame."""
        path = os.path.join(self._data_dir, self._filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"AMLNet CSV not found at:\n  {path}\n"
                "Download from Zenodo (no account needed):\n"
                "  https://zenodo.org/records/21237971\n"
                "Then rename the file to 'transactions.csv' and place it in the data dir."
            )

        print(f"Loading AMLNet (AMLGentex-equivalent) from: {path}")
        if self._nrows is not None:
            print(f"  (reading first {self._nrows:,} rows only)")

        # Load only the columns we need; skip the heavy metadata JSON blob.
        available_cols = pd.read_csv(path, nrows=0).columns.tolist()
        cols_to_load = [c for c in _USECOLS if c in available_cols]

        self._transactions = pd.read_csv(path, usecols=cols_to_load, nrows=self._nrows)
        self._transactions[_FRAUD_COL] = (
            self._transactions[_FRAUD_COL].fillna(0).astype(int)
        )
        print(f"  Loaded {len(self._transactions):,} rows.")

    def get_features(self) -> dict[str, dict[str, Any]]:
        """Return metadata for every loaded column in the AMLNet dataset."""
        df = self.get_transactions()

        def _vals(col: str, n: int = 8) -> list | None:
            if col not in df.columns:
                return None
            unique = df[col].dropna().unique()
            return unique[:n].tolist() if len(unique) <= 20 else None

        def _dtype(col: str) -> str:
            return str(df[col].dtype) if col in df.columns else "N/A"

        return {
            "step": {
                "dtype": _dtype("step"),
                "description": "Sequential simulation step (transaction index)",
                "possible_values": None,
            },
            "type": {
                "dtype": _dtype("type"),
                "description": "Transaction instrument / type",
                "possible_values": _vals("type"),
            },
            "amount": {
                "dtype": _dtype("amount"),
                "description": "Transaction amount in Australian dollars (AUD)",
                "possible_values": None,
            },
            "category": {
                "dtype": _dtype("category"),
                "description": "Spending/transaction category",
                "possible_values": _vals("category"),
            },
            "nameOrig": {
                "dtype": _dtype("nameOrig"),
                "description": "Originating account or customer identifier (node)",
                "possible_values": None,
            },
            "nameDest": {
                "dtype": _dtype("nameDest"),
                "description": "Destination account, customer, or merchant identifier (node)",
                "possible_values": None,
            },
            "oldbalanceOrg": {
                "dtype": _dtype("oldbalanceOrg"),
                "description": "Originating account balance before the transaction",
                "possible_values": None,
            },
            "newbalanceOrig": {
                "dtype": _dtype("newbalanceOrig"),
                "description": "Originating account balance after the transaction",
                "possible_values": None,
            },
            "isFraud": {
                "dtype": _dtype("isFraud"),
                "description": "Binary label for suspicious/fraudulent transactions",
                "possible_values": [0, 1],
            },
            "isMoneyLaundering": {
                "dtype": _dtype("isMoneyLaundering"),
                "description": "AML-specific label: 1 = money laundering, 0 = normal",
                "possible_values": [0, 1],
            },
            "laundering_typology": {
                "dtype": _dtype("laundering_typology"),
                "description": "Laundering typology (for labeled suspicious transactions)",
                "possible_values": _vals("laundering_typology"),
            },
            "hour": {
                "dtype": _dtype("hour"),
                "description": "Hour of day the transaction occurred (0–23)",
                "possible_values": None,
            },
            "day_of_week": {
                "dtype": _dtype("day_of_week"),
                "description": "Day of week (0 = Monday, 6 = Sunday)",
                "possible_values": list(range(7)),
            },
            "day_of_month": {
                "dtype": _dtype("day_of_month"),
                "description": "Day of month (1–31)",
                "possible_values": None,
            },
            "month": {
                "dtype": _dtype("month"),
                "description": "Month number (1–12)",
                "possible_values": list(range(1, 13)),
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
        if type_of_illicit == "F":
            fraud_col = "isFraud"
        else:
            fraud_col = _FRAUD_COL
            
        return self._fast_subgraph_from_df(
            source_col=_SOURCE_COL,
            dest_col=_DEST_COL,
            fraud_col=fraud_col,
            amount_col=_AMOUNT_COL,
            seed_node=seed_node,
            max_nodes=max_nodes,
            ax=ax,
            title=title or f"AMLNet – {type_of_illicit} subgraph",
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
