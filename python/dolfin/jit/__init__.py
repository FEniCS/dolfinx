import os


def get_pybind_include():
    """Find the pybind11 include path"""

    # Look in PYBIND11_DIR
    pybind_dir = os.getenv('PYBIND11_DIR', None)
    if pybind_dir:
        p = os.path.join(pybind_dir, "include")
        if (_check_pybind_path(p)):
            return [p]

    # Try extracting from pybind11 module
    try:
        # Get include paths from module
        import pybind11
        return [pybind11.get_include(True), pybind11.get_include()]
    except:
        pass

    # Look in /usr/local/include and /usr/include
    root = os.path.abspath(os.sep)
    for p in (os.path.join(root, "usr", "local", "include"), os.path.join(root, "usr", "include")):
        if (_check_pybind_path(p)):
            return [p]

    raise RuntimeError("Unable to locate pybind11 header files")


def _check_pybind_path(root):
    p = os.path.join(root, "pybind11", "pybind11.h")
    if os.path.isfile(p):
        return True
    else:
        return False
