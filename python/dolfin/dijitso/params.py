# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Martin Sandve Alnæs
#
# This file is part of DIJITSO.
#
# DIJITSO is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DIJITSO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DIJITSO. If not, see <http://www.gnu.org/licenses/>.

"""Utilities for dijitso parameters."""

from glob import glob
import os
import sys
import copy
import numbers
import configparser

from dolfin.dijitso.log import info, error, warning
from dolfin.dijitso.str import as_unicode


# Warning for fenics backwards compatibility:
if (os.environ.get("INSTANT_CACHE_DIR") and not os.environ.get("DIJITSO_CACHE_DIR")):
    warning("INSTANT_CACHE_DIR is no longer used by dijitso."
            " To set the cache directory for dijitso,"
            " set DIJITSO_CACHE_DIR.")


def discover_config_filename():
    basename = ".dijitso.conf"
    search_paths = [
        os.curdir,
        os.environ.get("DIJITSO_CONF"),
        os.path.expanduser("~"),
        "/etc/dijitso",
    ]
    for path in search_paths:
        if path is None:
            continue
        names = glob(os.path.join(path, basename))
        if names:
            assert len(names) == 1
            return names[0]
    return None


_config_file_contents = None


def read_config_file():
    "Read config file and cache the contents for the duration of the process."
    global _config_file_contents
    if _config_file_contents is None:
        filename = discover_config_filename()
        _config_file_contents = {}
        if filename is not None:
            info("Using config file '%s'." % (filename,))
            parser = configparser.SafeConfigParser()
            parser.read(filename)
            for category in parser.sections():
                category = as_unicode(category)
                _config_file_contents[category] = {}
                for name, value in parser.items(category):
                    name = as_unicode(name)
                    value = as_unicode(value)
                    _config_file_contents[category][name] = value
    return _config_file_contents


def default_cache_params():
    if sys.platform in ("win32", "cygwin"):
        default_lib_postfix = ".dll"
    else:
        default_lib_postfix = ".so"
    p = dict(
        cache_dir=None,  # See validate_params
        inc_dir="include",
        src_dir="src",
        lib_dir="lib",
        fail_dir_root=None,
        temp_dir_root=None,
        comm_dir="comm",
        log_dir="log",
        enable_build_log=True,
        src_storage="compress",
        src_postfix=".cpp",
        log_postfix=".txt",
        inc_postfix=".h",
        lib_postfix=default_lib_postfix,
        lib_prefix="lib",
        lib_basename="dijitso-",
        lib_loader="ctypes"
    )
    return p


def default_cxx_compiler():
    "Default C++ compiler"
    return os.getenv("CXX", "c++")


def default_cxx_flags():
    "Default C++ flags for all build modes."
    # Dropped because of some symbol sharing across dependent modules from ffc:
    # "-fvisibility=hidden",
    return ("-Wall", "-shared", "-fPIC", "-std=c++11")


def default_cxx_debug_flags():
    "Default C++ flags for debug=True. Note: FFC always overrides these."
    return ("-g", "-O0")


def default_cxx_release_flags():
    "Default C++ flags for debug=False. Note: FFC always overrides these."

    # These flags deal with handling of nan, inf, underflow, division
    # by zero, etc.  which should be ok for most of our purposes. It
    # might be better to place them in ffc or make them dependent on
    # compiler or optional or whatever, just throwing them in here now
    # to see how it works out.
    safe_fastmath_parts = ("-fno-math-errno", "-fno-trapping-math",
                           "-ffinite-math-only")
    return ("-O3",) + safe_fastmath_parts


def default_build_params():
    p = dict(
        cxx=default_cxx_compiler(),
        cxxflags=default_cxx_flags(),
        cxxflags_debug=default_cxx_debug_flags(),
        cxxflags_opt=default_cxx_release_flags(),
        include_dirs=(),
        lib_dirs=(),
        rpath_dirs=(),
        libs=(),
        debug=False,
    )
    return p


def default_generator_params():
    return {}


def default_params():
    p = dict(
        cache=default_cache_params(),
        build=default_build_params(),
        generator=default_generator_params(),)
    return p


_session_defaults = None


def session_default_params():
    global _session_defaults
    if _session_defaults is None:
        _session_defaults = validate_params()
    return copy.deepcopy(_session_defaults)


def as_bool(value):
    if isinstance(value, bool):
        return value
    elif value in ("True", "true", "1"):
        return True
    elif value in ("False", "false", "0"):
        return False
    else:
        error("Invalid boolean value %s" % (value,))


def as_str_tuple(p):
    """Convert p to a tuple of strings, allowing a list or tuple of
strings or a single string as input."""
    if isinstance(p, str):
        return (as_unicode(p),)
    elif isinstance(p, (tuple, list)):
        if all(isinstance(item, str) for item in p):
            return tuple(as_unicode(item) for item in p)
    raise RuntimeError("Expecting a string or list of strings, not %s." % (p,))


def copy_params(params):
    "Copy two-level dict of params."
    return {k: v.copy() for k, v in params.items()}


def check_params_keys(default, params):
    "Check that keys in params exist in defaults."
    for category in params:
        if category == "generator":
            continue
        if category not in default:
            error("Invalid parameter category '%s'." % category)
        if params[category] is not None:
            invalid = set(params[category]) - set(default[category])
            if invalid:
                error("Invalid parameter names %s in category '%s'." % (sorted(invalid), category))


def merge_params(default, params):
    "Merge two-level param dicts."
    p = {}
    for category in default:
        d = default[category].copy()
        p[category] = d
        v = params.get(category)
        if v is not None:
            p[category].update(v)
    return p


def validate_params(params):
    """Validate parameters to dijitso and fill in with defaults where missing."""

    # Start with defaults
    p0 = default_params()
    p = p0

    # Override with config file if any
    c = read_config_file()
    if c:
        check_params_keys(p, c)
        p = merge_params(p, c)

    # Override with runtime params if any
    if params:
        check_params_keys(p, params)
        p = merge_params(p, params)

    # Convert parameter types
    for category in p:
        category = as_unicode(category)
        if category == "generator":
            continue
        for name, value in p[category].items():
            name = as_unicode(name)
            v0 = p0[category][name]
            if isinstance(v0, str):
                value = as_unicode(value)
                # Expand paths including "~" to include
                # full user home directory path
                if name.endswith("_dir") and "~" in value:
                    value = os.path.expanduser(value)
            elif isinstance(v0, bool):
                value = as_bool(value)
            elif isinstance(v0, numbers.Number):
                value = type(v0)(value)
            elif isinstance(v0, tuple):
                value = as_str_tuple(value)
            p[category][name] = value

    # Allow environment variables to override default cache dir
    # Let dijitso specific dir win the contest
    cache_dir = os.environ.get("DIJITSO_CACHE_DIR")
    if not cache_dir:
        # Place default cache dir in virtualenv or conda prefix
        # if one of them are active, or under user's home directory
        home = os.path.expanduser("~")
        venv = os.environ.get("VIRTUAL_ENV")
        cenv = os.environ.get("CONDA_PREFIX")
        if venv == sys.prefix:
            env = venv
        elif cenv == sys.prefix:
            env = cenv
        else:
            env = home
        cache_dir = os.path.join(env, ".cache", "dijitso")
    p["cache"]["cache_dir"] = as_unicode(cache_dir)
    return p
