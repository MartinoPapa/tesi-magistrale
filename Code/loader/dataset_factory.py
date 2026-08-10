"""
dataset_factory.py
------------------
Factory class that instantiates the correct AML dataset loader by name.

Usage
-----
>>> from dataset_factory import DatasetFactory
>>> loader = DatasetFactory.get_loader("ibm_amlsim")
>>> loader.load()
>>> loader.summary()
"""

from __future__ import annotations

from .dataset_loader import AMLDatasetLoader
from .ibm_amlsim_loader import IBMAMLSimLoader, IBMAMLSimLargeLoader
from .amlgentex_loader import AMLGentexLoader
from .saml_d_loader import SAMLDLoader

# Mapping from canonical dataset name → loader class.
_REGISTRY: dict[str, type[AMLDatasetLoader]] = {
    "ibm_amlsim": IBMAMLSimLoader,
    "ibm_amlsim_large": IBMAMLSimLargeLoader,
    "amlgentex": AMLGentexLoader,
    "saml_d": SAMLDLoader,
}


class DatasetFactory:
    """Static factory for AML dataset loaders.

    All data-directory paths are hard-coded inside the individual loader
    classes, so callers only need to provide the dataset name.

    Supported names (case-insensitive)
    ------------------------------------
    ``"ibm_amlsim"``       →  :class:`~ibm_amlsim_loader.IBMAMLSimLoader`
    ``"ibm_amlsim_large"`` →  :class:`~ibm_amlsim_loader.IBMAMLSimLargeLoader`
    ``"amlgentex"``        →  :class:`~amlgentex_loader.AMLGentexLoader`
    ``"saml_d"``           →  :class:`~saml_d_loader.SAMLDLoader`
    """

    def __init__(self) -> None:
        raise TypeError("DatasetFactory is a static utility class and cannot be instantiated.")

    @staticmethod
    def get_loader(name: str, **kwargs) -> AMLDatasetLoader:
        """Return an *unloaded* loader instance for the requested dataset.

        Parameters
        ----------
        name:
            Dataset identifier (case-insensitive).
            One of ``"ibm_amlsim"``, ``"ibm_amlsim_large"``, ``"amlgentex"``, ``"saml_d"``.
        **kwargs:
            Extra keyword arguments forwarded to the loader constructor
            (e.g. ``nrows=100_000`` for SAMLDLoader).

        Returns
        -------
        AMLDatasetLoader
            A concrete loader instance.  Call ``.load()`` on it before use.

        Raises
        ------
        ValueError
            If *name* is not recognised.

        Example
        -------
        >>> loader = DatasetFactory.get_loader("saml_d", nrows=50_000)
        >>> loader.load()
        >>> loader.summary()
        """
        key = name.strip().lower()
        if key not in _REGISTRY:
            available = ", ".join(f'"{k}"' for k in _REGISTRY)
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {available}"
            )
        return _REGISTRY[key](**kwargs)

    @staticmethod
    def available_datasets() -> list[str]:
        """Return a list of all registered dataset names."""
        return list(_REGISTRY.keys())
