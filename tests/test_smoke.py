from importlib.metadata import version

import equitypanel


def test_package_imports():
    import equitypanel.data  # noqa: F401


def test_installed_version_matches_source():
    assert version("equitypanel") == equitypanel.__version__
