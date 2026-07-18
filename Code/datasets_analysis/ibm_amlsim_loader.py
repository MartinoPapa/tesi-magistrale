"""
ibm_amlsim_loader.py
--------------------
Concrete loader for the IBM AMLSim dataset (HI-Small variant).

Dataset source
--------------
IBM Transactions for Anti-Money Laundering (AML)
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Expected files
--------------
``<data_dir>/HI-Small_Trans.csv``       (required)
``<data_dir>/HI-Small_accounts.csv``    (optional – loaded for node metadata)

Transaction CSV schema  (HI-Small_Trans.csv)
--------------------------------------------
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

Accounts CSV schema  (HI-Small_accounts.csv)
--------------------------------------------
Bank Name           : str       – full name of the bank
Bank ID             : int       – numeric identifier of the bank
Account Number      : str       – hexadecimal account ID (matches Account / Account.1)
Entity ID           : str       – unique entity identifier
Entity Name         : str       – e.g. "Corporation #12345", "Sole Proprietorship #99"
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
    os.path.dirname(__file__), "..", "data", "ibm_amlsim"
)
_TRANS_FILENAME: str = "HI-Small_Trans.csv"
_ACCOUNTS_FILENAME: str = "HI-Small_accounts.csv"

_FRAUD_COL: str = "Is Laundering"
_SOURCE_COL: str = "Account"
_DEST_COL: str = "Account.1"
_AMOUNT_COL: str = "Amount Paid"


class IBMAMLSimLoader(AMLDatasetLoader):
    """Loader for the IBM AMLSim dataset (HI-Small variant).

    Loads the required transaction file and, optionally, the companion
    accounts file which provides node-level metadata (bank name, entity
    type). If the accounts file is absent, a warning is printed and the
    loader continues normally.

    Parameters
    ----------
    data_dir:
        Directory containing the CSV files.  Defaults to
        ``Code/data/ibm_amlsim/`` relative to this file.
    filename:
        Transaction CSV filename inside *data_dir*.  Defaults to
        ``HI-Small_Trans.csv``.  Pass ``None`` to auto-detect.
    accounts_filename:
        Accounts CSV filename inside *data_dir*.  Defaults to
        ``HI-Small_accounts.csv``.  Pass ``None`` to skip loading.
    """

    def __init__(
        self,
        data_dir: str = _DATA_DIR,
        filename: str | None = None,
        accounts_filename: str | None = _ACCOUNTS_FILENAME,
    ) -> None:
        super().__init__(data_dir)
        # For the transactions file: use auto-detection only when the
        # directory contains a single CSV (simple case); otherwise fall back
        # to the known default name.
        self._filename: str = (
            filename
            if filename is not None
            else self._resolve_filename(data_dir, _TRANS_FILENAME)
        )
        # The accounts file uses an explicit default; pass None to disable.
        self._accounts_filename: str | None = accounts_filename

    # ------------------------------------------------------------------
    # AMLDatasetLoader interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load transaction and (optionally) accounts CSVs into memory."""
        self._load_transactions()
        self._load_accounts()

    def _load_transactions(self) -> None:
        """Load ``HI-Small_Trans.csv`` into ``self._transactions``."""
        path = os.path.join(self._data_dir, self._filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"IBM AMLSim transaction CSV not found at:\n  {path}\n"
                f"Expected file: {self._filename}"
            )

        print(f"Loading IBM AMLSim transactions from: {path}")
        self._transactions = pd.read_csv(path)

        # Ensure the fraud column is integer.
        self._transactions[_FRAUD_COL] = (
            self._transactions[_FRAUD_COL].astype(int)
        )
        print(f"  Loaded {len(self._transactions):,} rows.")

    def _load_accounts(self) -> None:
        """Load ``HI-Small_accounts.csv`` into ``self._accounts`` (optional)."""
        if self._accounts_filename is None:
            return

        path = os.path.join(self._data_dir, self._accounts_filename)
        if not os.path.exists(path):
            print(
                f"  [INFO] Accounts file not found at {path!r} — "
                "node metadata will not be available."
            )
            return

        print(f"Loading IBM AMLSim accounts from: {path}")
        self._accounts = pd.read_csv(path)
        print(f"  Loaded {len(self._accounts):,} account records.")

    def get_features(self) -> dict[str, dict[str, Any]]:
        """Return metadata for every column in the IBM AMLSim dataset."""
        df = self.get_transactions()
        features: dict[str, dict[str, Any]] = {
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

        # Append account metadata columns only if the accounts file was loaded.
        if self._accounts is not None:
            acc = self._accounts
            features["Bank Name"] = {
                "dtype": str(acc["Bank Name"].dtype),
                "description": "Full name of the bank (from accounts file)",
                "possible_values": acc["Bank Name"].dropna().unique()[:6].tolist(),
            }
            features["Bank ID"] = {
                "dtype": str(acc["Bank ID"].dtype),
                "description": "Numeric bank identifier (from accounts file)",
                "possible_values": None,
            }
            features["Account Number"] = {
                "dtype": str(acc["Account Number"].dtype),
                "description": "Hexadecimal account ID — matches Account / Account.1",
                "possible_values": None,
            }
            features["Entity ID"] = {
                "dtype": str(acc["Entity ID"].dtype),
                "description": "Unique entity identifier (from accounts file)",
                "possible_values": None,
            }
            features["Entity Name"] = {
                "dtype": str(acc["Entity Name"].dtype),
                "description": "Human-readable entity name (e.g. Corporation #12345)",
                "possible_values": acc["Entity Name"].dropna().unique()[:4].tolist(),
            }

        return features

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
        """Plot a small subgraph via fast DataFrame BFS.

        Does NOT require :meth:`build_graph` to have been called.
        Red edges = fraudulent, blue = normal.
        """
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
