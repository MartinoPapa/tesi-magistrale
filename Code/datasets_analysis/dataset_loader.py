"""
dataset_loader.py
-----------------
Abstract base class for all AML dataset loaders.

Every concrete loader (IBM AMLSim, AMLGentex, SAML-D) must inherit from
`AMLDatasetLoader` and implement all abstract methods. This enforces a
uniform interface across datasets, enabling the rest of the codebase (and
the analysis notebook) to treat every dataset polymorphically.

Graph convention
----------------
* Nodes  → bank accounts (string IDs)
* Edges  → directed transactions  (sender → receiver)
* Edge attributes include at minimum:
    - ``amount``        : transaction amount (float)
    - ``is_laundering`` : fraud label (int, 0 or 1)
"""

from __future__ import annotations

import abc
import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class AMLDatasetLoader(abc.ABC):
    """Abstract base class for AML dataset loaders.

    Subclasses must implement :meth:`load`, :meth:`get_features`,
    :meth:`build_graph`, and :meth:`plot_subgraph`.

    The dataset is loaded lazily: call :meth:`load` before accessing
    :meth:`get_transactions`, :meth:`build_graph`, etc.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data_dir: str) -> None:
        """
        Parameters
        ----------
        data_dir:
            Root directory that contains the raw dataset files.
        """
        self._data_dir: str = data_dir
        self._transactions: pd.DataFrame | None = None
        self._accounts: pd.DataFrame | None = None
        self._graph: nx.DiGraph | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Read raw CSV file(s) from *data_dir* into ``_transactions``.

        After this call, :meth:`get_transactions` must return a non-empty
        :class:`pandas.DataFrame`.
        """

    @abc.abstractmethod
    def get_features(self) -> dict[str, dict[str, Any]]:
        """Return a description of every column in the transaction table.

        Returns
        -------
        dict
            Keys are column names. Each value is a dict with:

            * ``dtype``       : pandas dtype as a string
            * ``description`` : human-readable explanation
            * ``possible_values``: list of representative values, or
              ``None`` for continuous columns.
        """

    @abc.abstractmethod
    def build_graph(self) -> nx.DiGraph:
        """Build and return a directed transaction graph.

        The result is also cached in ``_graph`` for reuse.

        Returns
        -------
        nx.DiGraph
            Nodes = accounts; edges = directed transactions with attributes.
        """

    @abc.abstractmethod
    def plot_subgraph(
        self,
        seed_node: str | None = None,
        max_nodes: int = 50,
        type_of_illicit: str = "ML",
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot an ego-subgraph around *seed_node* (BFS up to *depth*).

        Fraudulent transaction edges are drawn in **red**;
        normal edges are drawn in **blue**.
        Nodes touched by at least one fraud edge are coloured red.

        Parameters
        ----------
        seed_node:
            Account ID to centre the subgraph on.  If ``None`` the
            implementation should pick a node that participates in at
            least one fraudulent transaction.
        depth:
            BFS radius around *seed_node*.
        ax:
            Matplotlib axes to draw on.  Created automatically if ``None``.
        title:
            Axes title.  A default is chosen if ``None``.

        Returns
        -------
        plt.Axes
            The axes containing the plot.
        """

    # ------------------------------------------------------------------
    # Concrete helpers (available once ``load()`` has been called)
    # ------------------------------------------------------------------

    def get_transactions(self) -> pd.DataFrame:
        """Return the raw transaction DataFrame.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called yet.
        """
        if self._transactions is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: call load() before accessing transactions."
            )
        return self._transactions

    def get_accounts(self) -> pd.DataFrame | None:
        """Return the accounts/nodes DataFrame, if one was loaded.

        Returns ``None`` when the concrete loader does not support a
        companion accounts file or when the file was not found on disk.
        """
        return self._accounts

    def summary(self) -> None:
        """Print a concise summary of the loaded dataset."""
        df = self.get_transactions()
        fraud_col = self._fraud_column()
        n_fraud = int(df[fraud_col].sum())
        n_total = len(df)
        pct = 100 * n_fraud / n_total if n_total else 0.0

        print(f"{'=' * 60}")
        print(f"  Dataset : {self.__class__.__name__}")
        print(f"  Rows    : {n_total:,}")
        print(f"  Columns : {len(df.columns)}")
        print(f"  Fraud   : {n_fraud:,}  ({pct:.4f} %)")

        if self._graph is not None:
            print(f"  Nodes   : {self._graph.number_of_nodes():,}")
            print(f"  Edges   : {self._graph.number_of_edges():,}")
        print(f"{'=' * 60}\n")

    def print_features(self) -> None:
        """Pretty-print a table of all features and their metadata."""
        features = self.get_features()
        col_w = max(len(k) for k in features) + 2
        print(f"\n{'Column':<{col_w}} {'Type':<12} {'Description'}")
        print("-" * 80)
        for col, meta in features.items():
            pv = meta.get("possible_values")
            pv_str = f"  (e.g. {pv})" if pv else ""
            print(f"{col:<{col_w}} {meta['dtype']:<12} {meta['description']}{pv_str}")
        print()

    def plot_distributions(
        self,
        df: pd.DataFrame | None = None,
        title_prefix: str = "",
        n_cols: int = 3,
        max_cols: int | None = None,
        max_categories: int = 12,
        col_config: dict | None = None,
    ) -> None:
        """Plot feature distributions with fraud/normal overlays and per-column config.

        For numeric columns draws overlapping density histograms with mean and
        median reference lines.  For categorical columns draws horizontal
        stacked bars with an optional fraud-rate secondary axis.
        Time columns can be regrouped by hour of day or day of week.

        Colors follow the Wong (2011) colorblind-safe palette.

        Parameters
        ----------
        df : pd.DataFrame | None
            DataFrame to plot.  Defaults to ``self.get_transactions()``.
        title_prefix : str
            Prefix prepended to the overall figure title.
        n_cols : int
            Number of columns in the subplot grid.
        max_cols : int | None
            Deprecated alias for *n_cols* (takes precedence if provided).
        max_categories : int
            Maximum number of top categories shown in bar charts.
        col_config : dict | None
            Per-column customisation.  Keys are column names; values are dicts
            with any of the following options:

            * ``"skip"``             – bool: exclude column entirely
            * ``"exclude_categories"``– list[str]: drop these exact categories
            * ``"fraud_rate"``       – bool: show fraud-rate secondary axis
              (default True)
            * ``"log_counts"``       – bool: log scale on count/density axis
            * ``"time_format"``      – str: ``"hour"`` | ``"weekday"`` |
              ``"weekday_num"`` | ``"month_num"``
            * ``"vlines_annotation"``– bool: add legend note for mean/median lines
            * ``"description"``      – str: subtitle shown below the column name
        """
        import math

        import numpy as np

        # Backward-compat alias
        if max_cols is not None:
            n_cols = max_cols

        col_config = col_config or {}

        # ---- Wong (2011) palette ----------------------------------------
        NORMAL_COLOR = "#56B4E9"       # Sky Blue
        FRAUD_COLOR = "#D55E00"        # Vermilion
        FRAUD_RATE_COLOR = "#E69F00"   # Orange

        if df is None:
            df = self.get_transactions()

        fraud_col = self._fraud_column()
        has_fraud = fraud_col in df.columns
        fraud_mask = df[fraud_col] == 1 if has_fraud else None

        # Exclude skipped columns so the grid reflows correctly
        all_cols = [
            c for c in df.columns
            if not col_config.get(c, {}).get("skip", False)
        ]
        n = len(all_cols)
        n_cols_eff = min(n_cols, n)
        n_rows = math.ceil(n / n_cols_eff)

        fig, axes = plt.subplots(n_rows, n_cols_eff, figsize=(6 * n_cols_eff, 4.5 * n_rows))
        fig.patch.set_facecolor("#f8f9fa")
        title = (
            f"{title_prefix} — Feature Distributions" if title_prefix
            else "Feature Distributions"
        )
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

        # Normalise axes to a 2-D array for uniform indexing.
        axes = np.array(axes)
        if axes.ndim == 0:
            axes = axes.reshape(1, 1)
        elif axes.ndim == 1:
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.reshape(-1, 1)

        # ---- Time-parsing helpers ------------------------------------------

        def _to_hour(s: pd.Series) -> pd.Series:
            """Try every reasonable strategy to extract hour-of-day (0–23)."""
            # Strategy 1: explicit HH:MM:SS format
            parsed = pd.to_datetime(s.astype(str), format="%H:%M:%S", errors="coerce")
            if parsed.notna().mean() >= 0.5:
                return parsed.dt.hour
            # Strategy 2: general datetime string
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() >= 0.5:
                return parsed.dt.hour
            # Strategy 3: regex – grab the leading HH digits from any string
            hour_extracted = s.astype(str).str.extract(r"^(\d{1,2}):", expand=False)
            numeric = pd.to_numeric(hour_extracted, errors="coerce")
            if numeric.notna().mean() >= 0.5 and numeric.dropna().between(0, 23).all():
                return numeric.astype("Int64")
            # Strategy 4: already numeric hours 0-23
            numeric = pd.to_numeric(s, errors="coerce")
            valid = numeric.dropna()
            if numeric.notna().mean() >= 0.5 and valid.between(0, 23).all():
                return numeric.astype("Int64")
            return pd.Series(pd.NA, index=s.index, dtype="Int64")

        def _to_weekday(s: pd.Series) -> pd.Series:
            """Try every reasonable strategy to extract day-of-week (0=Mon…6=Sun)."""
            # Strategy 1: general datetime string
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() >= 0.5:
                return parsed.dt.dayofweek
            # Strategy 2: UNIX epoch in seconds
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().mean() >= 0.5:
                for unit in ("s", "ms", "us", "ns"):
                    try:
                        parsed = pd.to_datetime(numeric, unit=unit, errors="coerce")
                        if parsed.notna().mean() >= 0.5:
                            return parsed.dt.dayofweek
                    except Exception:
                        continue
            return pd.Series(pd.NA, index=s.index, dtype="Int64")

        # ---- Time bar-chart helper -----------------------------------------

        _WEEKDAY_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
                           4: "Fri", 5: "Sat", 6: "Sun"}
        _MONTH_LABELS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                         5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                         9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

        def _plot_time_bars(
            ax: plt.Axes,
            ordered_keys: list,
            label_map: dict,
            sn: pd.Series,
            sf: pd.Series,
            show_fr: bool,
            xlabel: str = "",
        ) -> None:
            counts_n = sn.value_counts().reindex(ordered_keys, fill_value=0)
            counts_f = (
                sf.value_counts().reindex(ordered_keys, fill_value=0)
                if has_fraud and len(sf) > 0
                else pd.Series(0, index=ordered_keys)
            )
            x = np.arange(len(ordered_keys))
            tick_labels = [str(label_map.get(k, k)) for k in ordered_keys]

            ax.bar(x, counts_n.values, color=NORMAL_COLOR, alpha=0.85, label="Normal")
            if has_fraud and len(sf) > 0:
                ax.bar(
                    x, counts_f.values, bottom=counts_n.values,
                    color=FRAUD_COLOR, alpha=0.85, label="Fraud",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")

            if has_fraud and show_fr and len(sf) > 0:
                totals = counts_n.values + counts_f.values
                rates = np.where(totals > 0, counts_f.values / totals * 100, 0.0)
                ax2 = ax.twinx()
                ax2.plot(
                    x, rates, "o-",
                    color=FRAUD_RATE_COLOR, markersize=5, linewidth=1.5,
                )
                ax2.set_ylabel("Fraud rate (%)", color=FRAUD_RATE_COLOR, fontsize=8)
                ax2.tick_params(axis="y", colors=FRAUD_RATE_COLOR, labelsize=7)

        # ---- Main loop -----------------------------------------------------

        for i, col in enumerate(all_cols):
            r, c = divmod(i, n_cols_eff)
            ax = axes[r, c]
            ax.set_facecolor("#ffffff")
            for spine in ax.spines.values():
                spine.set_edgecolor("#dddddd")

            cfg = col_config.get(col, {})
            show_fraud_rate = cfg.get("fraud_rate", True)
            log_counts = cfg.get("log_counts", False)
            time_fmt = cfg.get("time_format", None)
            vlines_ann = cfg.get("vlines_annotation", False)
            description = cfg.get("description", None)

            series_all = df[col].dropna()
            if len(series_all) == 0:
                ax.set_title(col, fontsize=10)
                ax.text(0.5, 0.5, "All NaN", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            series_normal = (
                df.loc[~fraud_mask, col].dropna() if has_fraud else series_all
            )
            series_fraud = (
                df.loc[fraud_mask, col].dropna()
                if has_fraud else pd.Series(dtype=series_all.dtype)
            )

            # Title (with optional description subtitle)
            col_title = f"{col}\n{description}" if description else col
            ax.set_title(col_title, fontsize=10, fontweight="bold")

            # ---- Time-format branches ----------------------------------------
            if time_fmt == "hour":
                h_n = _to_hour(series_normal)
                h_f = _to_hour(series_fraud) if has_fraud else pd.Series(dtype="Int64")
                _plot_time_bars(
                    ax, list(range(24)), {h: str(h) for h in range(24)},
                    h_n, h_f, show_fraud_rate, xlabel="Hour of day",
                )

            elif time_fmt == "weekday":
                wd_n = _to_weekday(series_normal)
                wd_f = _to_weekday(series_fraud) if has_fraud else pd.Series(dtype="Int64")
                _plot_time_bars(
                    ax, [0, 1, 2, 3, 4, 5, 6], _WEEKDAY_LABELS,
                    wd_n, wd_f, show_fraud_rate, xlabel="Day of week",
                )

            elif time_fmt == "weekday_num":
                # Column already contains integer 0–6; just relabel it
                def _map_safe(s: pd.Series) -> pd.Series:
                    return s.round().astype("Int64").map(_WEEKDAY_LABELS)

                sn_mapped = _map_safe(series_normal)
                sf_mapped = _map_safe(series_fraud) if has_fraud else pd.Series(dtype="object")
                ordered_str = list(_WEEKDAY_LABELS.values())
                counts_n = sn_mapped.value_counts().reindex(ordered_str, fill_value=0)
                counts_f = (
                    sf_mapped.value_counts().reindex(ordered_str, fill_value=0)
                    if has_fraud and len(sf_mapped) > 0
                    else pd.Series(0, index=ordered_str)
                )
                x = np.arange(7)
                ax.bar(x, counts_n.values, color=NORMAL_COLOR, alpha=0.85, label="Normal")
                if has_fraud and len(series_fraud) > 0:
                    ax.bar(x, counts_f.values, bottom=counts_n.values,
                           color=FRAUD_COLOR, alpha=0.85, label="Fraud")
                ax.set_xticks(x)
                ax.set_xticklabels(ordered_str, fontsize=8)
                ax.set_xlabel("Day of week", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.legend(fontsize=7, loc="upper right")
                if has_fraud and show_fraud_rate and len(series_fraud) > 0:
                    totals = counts_n.values + counts_f.values
                    rates = np.where(totals > 0, counts_f.values / totals * 100, 0.0)
                    ax2 = ax.twinx()
                    ax2.plot(x, rates, "o-", color=FRAUD_RATE_COLOR, markersize=5, linewidth=1.5)
                    ax2.set_ylabel("Fraud rate (%)", color=FRAUD_RATE_COLOR, fontsize=8)
                    ax2.tick_params(axis="y", colors=FRAUD_RATE_COLOR, labelsize=7)

            elif time_fmt == "month_num":
                # Column already contains integer 1–12; relabel it
                def _map_month(s: pd.Series) -> pd.Series:
                    return s.round().astype("Int64").map(_MONTH_LABELS)

                sn_mapped = _map_month(series_normal)
                sf_mapped = _map_month(series_fraud) if has_fraud else pd.Series(dtype="object")
                ordered_str = list(_MONTH_LABELS.values())
                counts_n = sn_mapped.value_counts().reindex(ordered_str, fill_value=0)
                counts_f = (
                    sf_mapped.value_counts().reindex(ordered_str, fill_value=0)
                    if has_fraud and len(sf_mapped) > 0
                    else pd.Series(0, index=ordered_str)
                )
                x = np.arange(12)
                ax.bar(x, counts_n.values, color=NORMAL_COLOR, alpha=0.85, label="Normal")
                if has_fraud and len(series_fraud) > 0:
                    ax.bar(x, counts_f.values, bottom=counts_n.values,
                           color=FRAUD_COLOR, alpha=0.85, label="Fraud")
                ax.set_xticks(x)
                ax.set_xticklabels(ordered_str, fontsize=7, rotation=45, ha="right")
                ax.set_xlabel("Month", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.legend(fontsize=7, loc="upper right")
                if has_fraud and show_fraud_rate and len(series_fraud) > 0:
                    totals = counts_n.values + counts_f.values
                    rates = np.where(totals > 0, counts_f.values / totals * 100, 0.0)
                    ax2 = ax.twinx()
                    ax2.plot(x, rates, "o-", color=FRAUD_RATE_COLOR, markersize=5, linewidth=1.5)
                    ax2.set_ylabel("Fraud rate (%)", color=FRAUD_RATE_COLOR, fontsize=8)
                    ax2.tick_params(axis="y", colors=FRAUD_RATE_COLOR, labelsize=7)

            else:
                # ---- Standard categorical / numeric --------------------------
                nunique = series_all.nunique()
                dtype_kind = series_all.dtype.kind
                is_categorical = dtype_kind == "O" or nunique <= max_categories

                if is_categorical:
                    exclude_cats = cfg.get("exclude_categories", [])
                    if exclude_cats:
                        counts_all = series_all[~series_all.isin(exclude_cats)].value_counts().head(max_categories)
                    else:
                        counts_all = series_all.value_counts().head(max_categories)
                        
                    categories = counts_all.index.astype(str).tolist()
                    y_pos = np.arange(len(categories))

                    counts_normal_vals = (
                        series_normal.value_counts()
                        .reindex(categories, fill_value=0).values
                    )
                    counts_fraud_vals = (
                        series_fraud.value_counts()
                        .reindex(categories, fill_value=0).values
                        if has_fraud else np.zeros(len(categories), dtype=int)
                    )

                    bar_h = 0.6
                    ax.barh(
                        y_pos, counts_normal_vals, bar_h,
                        color=NORMAL_COLOR, alpha=0.85, label="Normal",
                    )
                    ax.barh(
                        y_pos, counts_fraud_vals, bar_h,
                        left=counts_normal_vals,
                        color=FRAUD_COLOR, alpha=0.85, label="Fraud",
                    )
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(categories, fontsize=8)
                    ax.set_xlabel("Count", fontsize=8)
                    ax.legend(fontsize=7, loc="lower right")

                    if log_counts:
                        max_val = (counts_normal_vals + counts_fraud_vals).max()
                        ax.set_xscale("log")
                        ax.set_xlim(left=0.9, right=max(max_val * 2, 2))

                    if has_fraud and show_fraud_rate:
                        totals = (
                            counts_all.reindex(categories, fill_value=0)
                            .values.astype(float)
                        )
                        fraud_rates = np.where(
                            totals > 0,
                            counts_fraud_vals.astype(float) / totals * 100,
                            0.0,
                        )
                        ax2 = ax.twiny()
                        ax2.plot(
                            fraud_rates, y_pos, "o-",
                            color=FRAUD_RATE_COLOR, markersize=5,
                            linewidth=1.5, label="Fraud rate %",
                        )
                        ax2.set_xlabel(
                            "Fraud rate (%)", color=FRAUD_RATE_COLOR, fontsize=8
                        )
                        ax2.tick_params(
                            axis="x", colors=FRAUD_RATE_COLOR, labelsize=7
                        )
                        max_rate = (
                            float(fraud_rates.max()) if fraud_rates.max() > 0 else 1.0
                        )
                        ax2.set_xlim(0, max_rate * 1.6)

                elif dtype_kind in "iufc":
                    upper = float(series_all.quantile(0.99))
                    lower = float(series_all.clip(upper=upper).min())
                    bins = np.linspace(lower, upper, 50)

                    ax.hist(
                        series_normal.clip(upper=upper),
                        bins=bins, color=NORMAL_COLOR, alpha=0.6, density=True,
                        label=f"Normal (n={len(series_normal):,})",
                    )
                    if has_fraud and len(series_fraud) > 0:
                        ax.hist(
                            series_fraud.clip(upper=upper),
                            bins=bins, color=FRAUD_COLOR, alpha=0.7, density=True,
                            label=f"Fraud (n={len(series_fraud):,})",
                        )

                    # Mean (dashed) and median (dash-dot) reference lines
                    for series, color in [
                        (series_normal, NORMAL_COLOR),
                        (series_fraud, FRAUD_COLOR),
                    ]:
                        if len(series) > 0:
                            ax.axvline(
                                series.mean(), color=color,
                                linestyle="--", linewidth=1.5, alpha=0.9,
                            )
                            ax.axvline(
                                series.median(), color=color,
                                linestyle="-.", linewidth=1.0, alpha=0.7,
                            )

                    ax.set_xlabel(col, fontsize=8)
                    ax.set_ylabel("Density", fontsize=8)
                    ax.legend(fontsize=7, loc="upper right")

                    if log_counts:
                        ax.set_yscale("log")

                else:
                    ax.text(
                        0.5, 0.5,
                        f"{nunique:,} unique values\n(no plot available)",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=9,
                    )

            # ---- vlines annotation (for numeric columns that have mean/median lines) ----
            if vlines_ann:
                ax.text(
                    0.98, 0.98,
                    "-- mean\n-·- median\n(one line per class)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6,
                    bbox=dict(
                        facecolor="white", alpha=0.80,
                        edgecolor="#cccccc", boxstyle="round,pad=0.3",
                    ),
                )

            ax.tick_params(labelsize=7)

        # Hide unused subplots
        for i in range(n, n_rows * n_cols_eff):
            r, c = divmod(i, n_cols_eff)
            axes[r, c].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()


    def plot_graph_metrics(self, sample_size: int = 10_000) -> None:
        """Plot topology metrics for the loaded transaction graph.

        Produces a 2×2 figure:

        * **Top-left**    : In-degree distribution (fraud vs normal, log scale)
        * **Top-right**   : Out-degree distribution (fraud vs normal, log scale)
        * **Bottom-left** : Weakly-connected-component size distribution
        * **Bottom-right**: Betweenness centrality — fraud vs normal (box plot)

        Uses the transaction DataFrame directly; :meth:`build_graph` does
        **not** need to be called first.  For large graphs betweenness
        centrality is computed on a random sample of *sample_size* nodes
        with a printed warning.

        Parameters
        ----------
        sample_size : int
            Maximum number of nodes used for betweenness centrality.
        """
        import math
        from collections import Counter

        import numpy as np

        NORMAL_COLOR = "#56B4E9"   # Sky Blue
        FRAUD_COLOR = "#D55E00"    # Vermilion

        df = self.get_transactions()
        fraud_col = self._fraud_column()
        src_col = self._source_column()
        dst_col = self._dest_column()

        src = df[src_col].astype(str)
        dst = df[dst_col].astype(str)
        fraud = df[fraud_col].astype(int)

        fraud_mask_s = fraud == 1
        fraud_accounts: set[str] = set(src[fraud_mask_s]) | set(dst[fraud_mask_s])

        # Degrees from the DataFrame directly (no nx.DiGraph needed yet).
        in_deg: Counter = Counter(dst)
        out_deg: Counter = Counter(src)
        all_nodes: set[str] = set(in_deg) | set(out_deg)

        fraud_nodes = [n for n in all_nodes if n in fraud_accounts]
        normal_nodes = [n for n in all_nodes if n not in fraud_accounts]

        fraud_in = [in_deg.get(n, 0) for n in fraud_nodes]
        normal_in = [in_deg.get(n, 0) for n in normal_nodes]
        fraud_out_vals = [out_deg.get(n, 0) for n in fraud_nodes]
        normal_out_vals = [out_deg.get(n, 0) for n in normal_nodes]

        # Build an nx graph for WCC and betweenness centrality.
        print("Building graph for topology analysis...")
        G: nx.DiGraph = nx.from_pandas_edgelist(
            df[[src_col, dst_col]].rename(
                columns={src_col: "_s", dst_col: "_t"}
            ),
            source="_s", target="_t", create_using=nx.DiGraph(),
        )

        print("Computing weakly connected components...")
        wcc_sizes = sorted(
            [len(c) for c in nx.weakly_connected_components(G)], reverse=True
        )
        wcc_counter: Counter = Counter(wcc_sizes)

        n_total_nodes = G.number_of_nodes()
        if n_total_nodes > sample_size:
            print(
                f"[INFO] {n_total_nodes:,} nodes total. "
                f"Betweenness computed on a {sample_size:,}-node sample."
            )
            rng = nx.utils.py_random_state(42)
            sample = set(rng.sample(list(G.nodes()), sample_size))
            G_bc = G.subgraph(sample).copy()
        else:
            G_bc = G

        print("Computing betweenness centrality...")
        bc = nx.betweenness_centrality(G_bc, normalized=True)
        fraud_bc = [bc[n] for n in fraud_nodes if n in bc]
        normal_bc = [bc[n] for n in normal_nodes if n in bc]

        # ---- Figure --------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Transaction Graph — Topology Metrics",
            fontsize=15, fontweight="bold",
        )
        fig.patch.set_facecolor("#f8f9fa")

        def _log_hist(
            ax: plt.Axes,
            data_n: list,
            data_f: list,
            xlabel: str,
            title: str,
        ) -> None:
            max_val = max(max(data_n, default=1), max(data_f, default=1))
            bins = np.logspace(0, math.log10(max_val + 1), 40)
            ax.hist(
                data_n, bins=bins, color=NORMAL_COLOR, alpha=0.6, density=True,
                label=f"Normal (n={len(data_n):,})",
            )
            ax.hist(
                data_f, bins=bins, color=FRAUD_COLOR, alpha=0.7, density=True,
                label=f"Fraud (n={len(data_f):,})",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Density (log)", fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_facecolor("#ffffff")

        _log_hist(
            axes[0, 0], normal_in, fraud_in,
            "In-Degree", "In-Degree Distribution",
        )
        _log_hist(
            axes[0, 1], normal_out_vals, fraud_out_vals,
            "Out-Degree", "Out-Degree Distribution",
        )

        # WCC size distribution
        ax_wcc = axes[1, 0]
        ax_wcc.set_facecolor("#ffffff")
        xs = sorted(wcc_counter.keys())
        ys = [wcc_counter[x] for x in xs]
        ax_wcc.bar(range(len(xs)), ys, color=NORMAL_COLOR, alpha=0.8)
        ax_wcc.set_xticks(range(len(xs)))
        ax_wcc.set_xticklabels(
            [str(x) for x in xs], rotation=45, ha="right", fontsize=7
        )
        ax_wcc.set_yscale("log")
        ax_wcc.set_xlabel("Component Size (nodes)", fontsize=9)
        ax_wcc.set_ylabel("Count (log)", fontsize=9)
        ax_wcc.set_title(
            "Weakly Connected Component Sizes", fontsize=11, fontweight="bold"
        )

        # Betweenness centrality box plot
        ax_bc = axes[1, 1]
        ax_bc.set_facecolor("#ffffff")
        bp = ax_bc.boxplot(
            [normal_bc, fraud_bc],
            labels=["Normal", "Fraud"],
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(NORMAL_COLOR)
        bp["boxes"][0].set_alpha(0.7)
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor(FRAUD_COLOR)
            bp["boxes"][1].set_alpha(0.7)
        ax_bc.set_ylabel("Betweenness Centrality", fontsize=9)
        ax_bc.set_title(
            "Betweenness Centrality: Fraud vs Normal",
            fontsize=11, fontweight="bold",
        )
        if n_total_nodes > sample_size:
            ax_bc.set_xlabel(
                f"(Computed on {sample_size:,}-node sample)",
                fontsize=8, color="#888888",
            )

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal helpers for subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _fraud_column(self) -> str:
        """Return the name of the binary fraud/laundering label column."""

    @abc.abstractmethod
    def _source_column(self) -> str:
        """Return the name of the sender-account column."""

    @abc.abstractmethod
    def _dest_column(self) -> str:
        """Return the name of the receiver-account column."""

    @staticmethod
    def _resolve_filename(data_dir: str, default_filename: str) -> str:
        """Return the CSV filename to use inside *data_dir*.

        If *data_dir* exists and contains exactly one CSV file, that file
        is returned (allows the user to place the file without worrying
        about the exact name). Otherwise *default_filename* is returned.

        Parameters
        ----------
        data_dir:
            Directory to inspect.
        default_filename:
            Fallback filename when auto-detection is not possible.
        """
        if os.path.isdir(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
            if len(csv_files) == 1:
                return csv_files[0]
        return default_filename

    # ------------------------------------------------------------------
    # Shared graph-building helper
    # ------------------------------------------------------------------

    def _build_graph_from_df(
        self,
        source_col: str,
        dest_col: str,
        fraud_col: str,
        amount_col: str,
    ) -> nx.DiGraph:
        """Build a DiGraph from the transaction DataFrame using vectorised ops.

        Uses :func:`networkx.from_pandas_edgelist` (much faster than
        row-by-row iteration for large datasets).

        Parameters
        ----------
        source_col, dest_col : column names for sender / receiver accounts.
        fraud_col            : column name for the 0/1 fraud label.
        amount_col           : column name for transaction amount.
        """
        df = self.get_transactions()

        # Cast account IDs to strings so all datasets share the same node type.
        tmp = df[[source_col, dest_col, amount_col, fraud_col]].copy()
        tmp[source_col] = tmp[source_col].astype(str)
        tmp[dest_col] = tmp[dest_col].astype(str)

        G = nx.from_pandas_edgelist(
            tmp,
            source=source_col,
            target=dest_col,
            edge_attr=[amount_col, fraud_col],
            create_using=nx.DiGraph(),
        )

        # Normalise edge attributes to canonical names used by the plotter.
        if fraud_col != "is_laundering" or amount_col != "amount":
            nx.set_edge_attributes(
                G,
                {
                    (u, v): {
                        "is_laundering": int(d.get(fraud_col, 0)),
                        "amount": float(d.get(amount_col, 0.0)),
                    }
                    for u, v, d in G.edges(data=True)
                },
            )

        self._graph = G
        return G

    # ------------------------------------------------------------------
    # Fast subgraph helper for visualisation (no full graph required)
    # ------------------------------------------------------------------

    def _fast_subgraph_from_df(
        self,
        source_col: str,
        dest_col: str,
        fraud_col: str,
        amount_col: str,
        seed_node: str | None = None,
        max_nodes: int = 50,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        import random

        df = self.get_transactions()
        src_s = df[source_col].astype(str)
        dst_s = df[dest_col].astype(str)

        if seed_node is not None:
            # Treat seed_node as a random seed for reproducibility
            random.seed(seed_node)

        fraud_mask = df[fraud_col] == 1
        if not fraud_mask.any():
            raise ValueError("No fraudulent nodes found to use as a starting node.")

        # Pick a random fraudulent edge to guarantee at least one fraud edge in the subgraph
        fraud_edges_idx = df.index[fraud_mask].tolist()
        random_idx = random.choice(fraud_edges_idx)
        start_u = str(df.at[random_idx, source_col])
        start_v = str(df.at[random_idx, dest_col])

        start_node = start_u  # center of ego graph

        # Strict textbook queue-based BFS
        all_nodes = {start_node}
        queue = [start_node]

        # Force start_v as the very first neighbor to guarantee the fraudulent connection
        queue.append(start_v)
        all_nodes.add(start_v)

        while queue and len(all_nodes) < max_nodes:
            current = queue.pop(0)

            # Find all direct neighbors of the current node
            mask = (src_s == current) | (dst_s == current)

            new_nodes = set(src_s[mask]) | set(dst_s[mask])
            neighbors = new_nodes - all_nodes

            if not neighbors:
                continue

            # Randomize neighbor expansion order for this node
            neighbors_list = sorted(list(neighbors))
            random.shuffle(neighbors_list)

            for n in neighbors_list:
                if len(all_nodes) >= max_nodes:
                    break
                all_nodes.add(n)
                queue.append(n)

        # Extract edges among the selected nodes
        sub_mask = src_s.isin(all_nodes) & dst_s.isin(all_nodes)
        sub_df = df[sub_mask]

        G = nx.DiGraph()
        # Rename to safe attribute names before itertuples() so that column
        # names containing dots (e.g. "Account.1") don't raise AttributeError.
        # Note: pandas also strips leading underscores, so use a letter prefix.
        sub_iter = (
            sub_df[[source_col, dest_col, amount_col, fraud_col]]
            .rename(columns={source_col: "col_src", dest_col: "col_dst",
                             amount_col: "col_amt", fraud_col: "col_fraud"})
        )
        for row in sub_iter.itertuples(index=False):
            u = str(row.col_src)
            v = str(row.col_dst)
            is_fraud = int(row.col_fraud)
            amt = float(row.col_amt)

            if G.has_edge(u, v):
                # If there's already a fraudulent transaction between these nodes, keep the edge marked as fraudulent
                if G[u][v].get('is_laundering', 0) == 1:
                    is_fraud = 1
                # Accumulate the amount for multi-edges (optional, but good practice)
                amt += G[u][v].get('amount', 0.0)

            G.add_edge(u, v, amount=amt, is_laundering=is_fraud)

        if start_node not in G:
            G.add_node(start_node)

        return self._plot_ego_subgraph(
            graph=G,
            seed_node=start_node,
            depth=10,
            ax=ax,
            title=title,
        )

    # ------------------------------------------------------------------
    # Shared subgraph-plotting helper
    # ------------------------------------------------------------------

    def _plot_ego_subgraph(
        self,
        graph: nx.DiGraph,
        seed_node: str | None,
        depth: int,
        ax: plt.Axes | None,
        title: str | None,
        fraud_col_attr: str = "is_laundering",
    ) -> plt.Axes:
        """Draw a BFS ego-subgraph with role-based colouring and metrics.

        Node colour encodes the **structural role** in the fraud sub-network
        using the Wong (2011) colorblind-safe palette:

        * Sky Blue     (#56B4E9) – no fraud connection (normal)
        * Vermilion    (#D55E00) – source: only *outgoing* fraud edges
        * Orange       (#E69F00) – mule/relay: both in- and out-going fraud edges
        * Bluish Green (#009E73) – sink: only *incoming* fraud edges

        Node **size** scales with log(degree) so hub nodes stand out without
        dominating the layout.  Edge **width** scales with log(amount),
        normalised within the subgraph.

        A metrics panel and an AML typology annotation are added as text
        boxes inside the axes.

        Parameters
        ----------
        graph         : The full or pre-built transaction DiGraph.
        seed_node     : Centre node; auto-selected if None.
        depth         : BFS depth from the seed node.
        ax            : Matplotlib axes (created if None).
        title         : Plot title.
        fraud_col_attr: Edge attribute name for the fraud label.
        """
        from math import log1p

        import numpy as np

        # ---- Wong (2011) palette -------------------------------------------
        NORMAL_COLOR = "#56B4E9"       # Sky Blue  – no fraud involvement
        SOURCE_COLOR = "#D55E00"       # Vermilion – only sends fraud
        MULE_COLOR = "#E69F00"         # Orange    – sends & receives fraud
        SINK_COLOR = "#009E73"         # Bluish Green – only receives fraud
        FRAUD_EDGE_COLOR = "#D55E00"   # Vermilion
        NORMAL_EDGE_COLOR = "#56B4E9"  # Sky Blue

        # ---- Select seed node ------------------------------------------------
        if seed_node is None:
            fraud_src = {
                u for u, v, d in graph.edges(data=True)
                if d.get(fraud_col_attr, 0) == 1
            }
            fraud_dst = {
                v for u, v, d in graph.edges(data=True)
                if d.get(fraud_col_attr, 0) == 1
            }
            fraud_nodes_set = fraud_src | fraud_dst
            seed_node = (
                next(iter(fraud_nodes_set)) if fraud_nodes_set
                else next(iter(graph.nodes))
            )

        # ---- Extract ego-subgraph -------------------------------------------
        undirected = graph.to_undirected()
        ego_nodes = set(nx.ego_graph(undirected, seed_node, radius=depth).nodes())
        sub = graph.subgraph(ego_nodes).copy()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        # ---- Layout ----------------------------------------------------------
        if sub.number_of_nodes() <= 60:
            try:
                pos = nx.kamada_kawai_layout(sub)
            except Exception:
                pos = nx.spring_layout(sub, seed=42)
        else:
            pos = nx.spring_layout(sub, seed=42)

        # ---- Classify edges --------------------------------------------------
        fraud_edges_data = [
            (u, v, d) for u, v, d in sub.edges(data=True)
            if d.get(fraud_col_attr, 0) == 1
        ]
        normal_edges_data = [
            (u, v, d) for u, v, d in sub.edges(data=True)
            if d.get(fraud_col_attr, 0) == 0
        ]
        fraud_edge_list = [(u, v) for u, v, _ in fraud_edges_data]
        normal_edge_list = [(u, v) for u, v, _ in normal_edges_data]

        # ---- Build fraud-only subgraph for typology detection ----------------
        fraud_sub = nx.DiGraph()
        for u, v, d in fraud_edges_data:
            fraud_sub.add_edge(u, v, **d)

        # ---- Classify nodes by structural role -------------------------------
        fraud_out_nodes = {u for u, v, _ in fraud_edges_data}
        fraud_in_nodes = {v for u, v, _ in fraud_edges_data}

        def _node_role(n: str) -> str:
            is_out = n in fraud_out_nodes
            is_in = n in fraud_in_nodes
            if is_out and is_in:
                return "mule"
            if is_out:
                return "source"
            if is_in:
                return "sink"
            return "normal"

        role_palette = {
            "normal": NORMAL_COLOR,
            "source": SOURCE_COLOR,
            "mule": MULE_COLOR,
            "sink": SINK_COLOR,
        }
        node_list = list(sub.nodes())
        node_colors = [role_palette[_node_role(n)] for n in node_list]

        # ---- Node sizes (log-degree) -----------------------------------------
        degrees = dict(sub.degree())
        node_sizes = [
            float(min(100.0 + 200.0 * log1p(degrees.get(n, 0)), 800.0))
            for n in node_list
        ]

        # ---- Edge widths (log-amount, normalised per subgraph) ---------------
        all_amounts = [d.get("amount", 1.0) for _, _, d in sub.edges(data=True)]
        max_amount = max(all_amounts) if all_amounts else 1.0

        def _edge_width(d: dict) -> float:
            amt = float(d.get("amount", 1.0))
            return float(np.clip(
                log1p(amt) / log1p(max_amount + 1) * 3.0, 0.5, 4.0
            ))

        fraud_widths = [_edge_width(d) for _, _, d in fraud_edges_data]
        normal_widths = [_edge_width(d) for _, _, d in normal_edges_data]

        # ---- Draw ------------------------------------------------------------
        nx.draw_networkx_nodes(
            sub, pos, ax=ax, nodelist=node_list,
            node_color=node_colors, node_size=node_sizes, alpha=0.92,
        )
        nx.draw_networkx_labels(sub, pos, ax=ax, font_size=5, font_color="white")
        if normal_edge_list:
            nx.draw_networkx_edges(
                sub, pos, edgelist=normal_edge_list, ax=ax,
                edge_color=NORMAL_EDGE_COLOR, arrows=True,
                alpha=0.4, arrowsize=10, width=normal_widths,
                connectionstyle="arc3,rad=0.1",
            )
        if fraud_edge_list:
            nx.draw_networkx_edges(
                sub, pos, edgelist=fraud_edge_list, ax=ax,
                edge_color=FRAUD_EDGE_COLOR, arrows=True,
                alpha=0.92, arrowsize=14, width=fraud_widths,
                connectionstyle="arc3,rad=0.1",
            )

        ax.set_title(
            title or f"Subgraph around '{seed_node}' (depth={depth})",
            fontsize=11, fontweight="bold",
        )
        ax.axis("off")

        # ---- Metrics panel (top-left) ----------------------------------------
        n_fraud_e = len(fraud_edge_list)
        n_total_e = sub.number_of_edges()
        pct_fraud_e = 100.0 * n_fraud_e / n_total_e if n_total_e else 0.0
        hub_count = sum(1 for d in degrees.values() if d > 5)
        avg_amount = (sum(all_amounts) / len(all_amounts)) if all_amounts else 0.0

        metrics_text = (
            f"Nodes: {sub.number_of_nodes()} | Edges: {n_total_e}\n"
            f"Fraud edges: {n_fraud_e} ({pct_fraud_e:.1f}%)\n"
            f"Hub nodes (deg>5): {hub_count}\n"
            f"Avg. amount: {avg_amount:,.0f}"
        )
        ax.text(
            0.02, 0.98, metrics_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                alpha=0.85, edgecolor="#cccccc",
            ),
        )

        # ---- Typology annotation (bottom-left) -------------------------------
        typologies = self._detect_aml_typologies(fraud_sub)
        typo_text = (
            "Detected typologies:\n" + "\n".join(f"• {t}" for t in typologies)
            if typologies else "No typologies detected"
        )
        ax.text(
            0.02, 0.02, typo_text, transform=ax.transAxes,
            va="bottom", ha="left", fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                alpha=0.85, edgecolor="#cccccc",
            ),
        )

        # ---- Legend ----------------------------------------------------------
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=SOURCE_COLOR, label="Source (fraud origin)"),
            Patch(facecolor=MULE_COLOR, label="Mule / Relay"),
            Patch(facecolor=SINK_COLOR, label="Sink (fraud destination)"),
            Patch(facecolor=NORMAL_COLOR, label="Normal node"),
            Line2D([0], [0], color=FRAUD_EDGE_COLOR, lw=2, label="Fraud edge"),
            Line2D([0], [0], color=NORMAL_EDGE_COLOR, lw=2, label="Normal edge"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right",
            fontsize=7, framealpha=0.85,
        )

        return ax

    def _detect_aml_typologies(self, fraud_subgraph: nx.DiGraph) -> list[str]:
        """Detect AML typologies in the fraud-edge sub-network.

        Returns a list of detected typology names (empty list when none found).

        Typologies checked
        ------------------
        * **Fan-Out (Layering)**       – any fraud node has out-degree ≥ 3
        * **Fan-In (Aggregation)**     – any fraud node has in-degree ≥ 3
        * **Cycle (Structuring)**      – at least one directed cycle exists
        * **Long Chain (Integration)** – a directed path of ≥ 4 hops exists

        Parameters
        ----------
        fraud_subgraph : nx.DiGraph
            Subgraph containing only fraud edges.
        """
        if fraud_subgraph.number_of_edges() == 0:
            return []

        typologies: list[str] = []

        # Fan-Out: any node sends fraud to 3+ targets
        if any(
            fraud_subgraph.out_degree(n) >= 3 for n in fraud_subgraph.nodes()
        ):
            typologies.append("Fan-Out (Layering)")

        # Fan-In: any node receives fraud from 3+ sources
        if any(
            fraud_subgraph.in_degree(n) >= 3 for n in fraud_subgraph.nodes()
        ):
            typologies.append("Fan-In (Aggregation)")

        # Cycle: at least one directed cycle present
        try:
            next(nx.simple_cycles(fraud_subgraph))
            typologies.append("Cycle (Structuring)")
        except StopIteration:
            pass

        # Long Chain: a directed path of 4+ hops via BFS
        for source in fraud_subgraph.nodes():
            depths = nx.single_source_shortest_path_length(
                fraud_subgraph, source, cutoff=4
            )
            if any(d >= 4 for d in depths.values()):
                typologies.append("Long Chain (Integration)")
                break

        return typologies
