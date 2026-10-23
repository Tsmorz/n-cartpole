"""System identification from device recordings (sim-to-real).

- :mod:`n_cartpole.sysid.recording`: read/write the firmware's ``.ncpr`` files.
- :mod:`n_cartpole.sysid.fit`: fit the plant parameters to recordings.
- :mod:`n_cartpole.sysid.config_out`: write a fitted config TOML for ``train.py``.
- :mod:`n_cartpole.sysid.report`: interactive HTML report of a fit.
"""
