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
        for row in sub_df[[source_col, dest_col, amount_col, fraud_col]].itertuples(index=False):
            u = str(getattr(row, source_col))
            v = str(getattr(row, dest_col))
            is_fraud = int(getattr(row, fraud_col))
            amt = float(getattr(row, amount_col))
            
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
        """Draw a BFS ego-subgraph with red=fraud, blue=normal colouring.

        Parameters
        ----------
        graph         : The full transaction graph (NetworkX DiGraph).
        seed_node     : Centre node; auto-selected if None.
        depth         : BFS depth from the seed.
        ax            : Matplotlib axes (created if None).
        title         : Plot title.
        fraud_col_attr: Edge attribute name for the fraud label.
        """
        if seed_node is None:
            # Pick a node that participates in at least one fraud edge.
            fraud_nodes = {
                u
                for u, v, d in graph.edges(data=True)
                if d.get(fraud_col_attr, 0) == 1
            } | {
                v
                for u, v, d in graph.edges(data=True)
                if d.get(fraud_col_attr, 0) == 1
            }
            if fraud_nodes:
                seed_node = next(iter(fraud_nodes))
            else:
                seed_node = next(iter(graph.nodes))

        # Extract ego-graph (undirected BFS, then induced subgraph on DiGraph).
        undirected = graph.to_undirected()
        ego_nodes = set(nx.ego_graph(undirected, seed_node, radius=depth).nodes())
        sub = graph.subgraph(ego_nodes).copy()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        # Compute layout.
        pos = nx.spring_layout(sub, seed=42)

        # Classify edges.
        fraud_edges = [
            (u, v)
            for u, v, d in sub.edges(data=True)
            if d.get(fraud_col_attr, 0) == 1
        ]
        normal_edges = [
            (u, v)
            for u, v, d in sub.edges(data=True)
            if d.get(fraud_col_attr, 0) == 0
        ]

        # Classify nodes: red if connected to any fraud edge.
        fraud_node_set = {n for e in fraud_edges for n in e}
        node_colors = [
            "#e74c3c" if n in fraud_node_set else "#3498db"
            for n in sub.nodes()
        ]

        # Draw components.
        nx.draw_networkx_nodes(
            sub, pos, ax=ax, node_color=node_colors,
            node_size=300, alpha=0.9,
        )
        nx.draw_networkx_labels(
            sub, pos, ax=ax, font_size=6, font_color="white",
        )
        nx.draw_networkx_edges(
            sub, pos, edgelist=normal_edges, ax=ax,
            edge_color="#3498db", arrows=True, alpha=0.6,
            arrowsize=12, connectionstyle="arc3,rad=0.1",
        )
        nx.draw_networkx_edges(
            sub, pos, edgelist=fraud_edges, ax=ax,
            edge_color="#e74c3c", arrows=True, alpha=0.9,
            arrowsize=15, width=2.0, connectionstyle="arc3,rad=0.1",
        )

        ax.set_title(
            title or f"Subgraph around node '{seed_node}' (depth={depth})",
            fontsize=10,
        )
        ax.axis("off")

        # Legend.
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="#e74c3c", lw=2, label="Fraudulent edge"),
            Line2D([0], [0], color="#3498db", lw=2, label="Normal edge"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        return ax
