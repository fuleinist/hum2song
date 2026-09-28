# Makes `tests` a package so `from tests import fixtures` resolves under a bare
# `pytest` invocation. Without it, pytest only finds the module when the current
# directory happens to be on sys.path (which `python -m pytest` arranges and the
# `pytest` console script does not), so the two commands disagreed.