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

"""Utilities for disk cache features of dijitso."""

from glob import glob
import os
import re
import sys
import ctypes
from dolfin.dijitso.system import ldd
from dolfin.dijitso.system import make_dirs
from dolfin.dijitso.system import try_delete_file, try_copy_file
from dolfin.dijitso.system import gzip_file, gunzip_file
from dolfin.dijitso.system import read_textfile, store_textfile
from dolfin.dijitso.log import debug, error, warning


def extract_files(signature, cache_params, prefix="", path=os.curdir,
                  categories=("inc", "src", "lib", "log")):
    """Make a copy of files stored under this signature.

    Target filenames are '<path>/<prefix>-<signature>.*'
    """
    path = os.path.join(path, prefix + signature)
    make_dirs(path)

    if "inc" in categories:
        inc_filename = create_inc_filename(signature, cache_params)
        try_copy_file(inc_filename, path)
    if "src" in categories:
        src_filename = create_src_filename(signature, cache_params)
        if not os.path.exists(src_filename):
            src_filename = src_filename + ".gz"
        if os.path.exists(src_filename):
            try_copy_file(src_filename, path)
            if src_filename.endswith(".gz"):
                gunzip_file(os.path.join(path, os.path.basename(src_filename)))
    if "lib" in categories:
        lib_filename = create_lib_filename(signature, cache_params)
        try_copy_file(lib_filename, path)
    if "log" in categories:
        log_filename = create_log_filename(signature, cache_params)
        try_copy_file(log_filename, path)

    return path


def extract_lib_signatures(cache_params):
    "Extract signatures from library files in cache."
    p = os.path.join(cache_params["cache_dir"], cache_params["lib_dir"])
    filenames = glob(os.path.join(p, "*"))

    r = re.compile(create_lib_filename("(.*)", cache_params))
    sigs = []
    for f in filenames:
        m = r.match(f)
        if m:
            sigs.append(m.group(1))
    return sigs


def clean_cache(cache_params, dryrun=True,
                categories=("inc", "src", "lib", "log")):
    "Delete files from cache."
    gc = glob_cache(cache_params, categories=categories)
    for category in gc:
        for fn in gc[category]:
            if dryrun:
                print("rm %s" % (fn,))
            else:
                try_delete_file(fn)


def glob_cache(cache_params, categories=("inc", "src", "lib", "log")):
    """Return dict with contents of cache subdirectories."""
    g = {}
    for foo in categories:
        p = os.path.join(cache_params["cache_dir"], cache_params[foo + "_dir"])
        g[foo] = glob(os.path.join(p, "*"))
    return g


def grep_cache(regex, cache_params,
               linenumbers=False, countonly=False,
               signature=None,
               categories=("inc", "src", "log")):
    "Search through files in cache for a pattern."
    allmatches = {}
    gc = glob_cache(cache_params, categories=categories)
    for category in categories:
        for fn in gc.get(category, ()):
            # Skip non-matches if specific signature is specified
            if signature is not None and signature not in fn:
                continue

            if countonly:
                matches = 0
            else:
                matches = []

            if category == "lib":
                # If category is "lib", use ldd
                # TODO: on mac need to use otool
                libs = ldd(fn)
                for k, libpath in sorted(libs.items()):
                    if not libpath:
                        continue
                    m = regex.match(libpath)
                    if m:
                        if countonly:
                            matches += 1
                        else:
                            line = "%s => %s" % (k, libpath)
                            matches.append(line)
            else:
                content = read_textfile(fn)
                lines = content.splitlines() if content else ()
                for i, line in enumerate(lines):
                    m = regex.match(line)
                    if m:
                        if countonly:
                            matches += 1
                        else:
                            line = line.rstrip("\n\r")
                            if linenumbers:
                                line = (i, line)
                            matches.append(line)

            if matches:
                allmatches[fn] = matches
    return allmatches


def extract_function(lines):
    "Extract function code starting at first line of lines."
    n = len(lines)

    # Function starts at line 0 by assumption
    begin = 0

    # Worst case body range
    body_begin = begin
    body_end = n

    # Body starts at first {
    for i in range(begin, n):
        if "{" in lines[i]:
            body_begin = i
            break

    # Body ends when {} are balanced back to 0
    braces = 0
    for i in range(body_begin, n):
        if "{" in lines[i]:
            braces += 1
        if "}" in lines[i]:
            braces -= 1
        if braces == 0:
            body_end = i
            break

    # Include the last line in range
    end = body_end + 1
    sublines = lines[begin:end]
    return "".join(sublines)


def _create_basename(foo, signature, cache_params):
    return "".join((cache_params.get(foo + "_prefix", ""),
                    cache_params.get(foo + "_basename", ""),
                    signature,
                    cache_params.get(foo + "_postfix", "")))


def _create_filename(foo, signature, cache_params):
    basename = _create_basename(foo, signature, cache_params)
    return os.path.join(cache_params["cache_dir"],
                        cache_params[foo + "_dir"], basename)


def create_log_filename(signature, cache_params):
    "Create log filename based on signature and params."
    return _create_filename("log", signature, cache_params)


def create_inc_basename(signature, cache_params):
    "Create header filename based on signature and params."
    return _create_basename("inc", signature, cache_params)


def create_inc_filename(signature, cache_params):
    "Create header filename based on signature and params."
    return _create_filename("inc", signature, cache_params)


def create_src_filename(signature, cache_params):
    "Create source code filename based on signature and params."
    return _create_filename("src", signature, cache_params)


def create_src_basename(signature, cache_params):
    "Create source code filename based on signature and params."
    return _create_basename("src", signature, cache_params)


def create_lib_basename(signature, cache_params):
    "Create library filename based on signature and params."
    return _create_basename("lib", signature, cache_params)


def create_lib_filename(signature, cache_params):
    "Create library filename based on signature and params."
    return _create_filename("lib", signature, cache_params)


def create_libname(signature, cache_params):
    """Create library name based on signature and params,
    without path, prefix 'lib', or extension '.so'."""
    return cache_params["lib_basename"] + signature


def create_fail_dir_path(signature, cache_params):
    "Create path name to place files after a module build failure."
    fail_root = cache_params["fail_dir_root"] or os.curdir
    fail_dir = os.path.join(fail_root, "jitfailure-" + signature)
    return os.path.abspath(fail_dir)


def make_inc_dir(cache_params):
    d = os.path.join(cache_params["cache_dir"], cache_params["inc_dir"])
    make_dirs(d)
    return d


def make_src_dir(cache_params):
    d = os.path.join(cache_params["cache_dir"], cache_params["src_dir"])
    make_dirs(d)
    return d


def make_lib_dir(cache_params):
    d = os.path.join(cache_params["cache_dir"], cache_params["lib_dir"])
    make_dirs(d)
    return d


def make_log_dir(cache_params):
    d = os.path.join(cache_params["cache_dir"], cache_params["log_dir"])
    make_dirs(d)
    return d


_ensure_dirs_called = {}


def ensure_dirs(cache_params):
    global _ensure_dirs_called
    # This ensures directories are created only once during a process
    # for each value that cache_dir takes, in case it changes during
    # the process lifetime.
    c = cache_params["cache_dir"]
    if c not in _ensure_dirs_called:
        make_inc_dir(cache_params)
        make_src_dir(cache_params)
        make_lib_dir(cache_params)
        make_log_dir(cache_params)
        _ensure_dirs_called[c] = True


def read_library_binary(lib_filename):
    "Read compiled shared library as binary blob into a numpy byte array."
    import numpy
    return numpy.fromfile(lib_filename, dtype=numpy.uint8)


def write_library_binary(lib_data, signature, cache_params):
    "Store compiled shared library from binary blob in numpy byte array to cache."
    make_lib_dir(cache_params)
    lib_filename = create_lib_filename(signature, cache_params)
    lib_data.tofile(lib_filename)
    # TODO: Set permissions?


def analyse_load_error(e, lib_filename, cache_params):
    # Try to analyse error further for better error message:
    msg = str(e)
    r = re.compile("(" + create_lib_basename(".*", cache_params) + ")")
    m = r.match(msg)
    if m:
        # Found libname mentioned in message
        mlibname = m.group(1)
        mlibname = os.path.join(cache_params["cache_dir"],
                                cache_params["lib_dir"], mlibname)
    else:
        mlibname = lib_filename

    if lib_filename != mlibname:
        # Message mentions some other dijitso library,
        # double check if this other file exists
        # (if it does, could be paths or rpath issue)
        if os.path.exists(mlibname):
            emsg = ("dijitso failed to load library:\n\t%s\n"
                    "but dependency file exists:\n\t%s\nerror is:\n\t%s" % (
                        lib_filename, mlibname, str(e)))
        else:
            emsg = ("dijitso failed to load library:\n\t%s\n"
                    "dependency file missing:\n\t%s\nerror is:\n\t%s" % (
                        lib_filename, mlibname, str(e)))
    else:
        # Message doesn't mention another dijitso library,
        # double check if library file we tried to load exists
        # (if it does, could be paths issue)
        if os.path.exists(lib_filename):
            emsg = ("dijitso failed to load existing file:\n"
                    "\t%s\nerror is:\n\t%s" % (lib_filename, str(e)))
        else:
            emsg = ("dijitso failed to load missing file:\n"
                    "\t%s\nerror is:\n\t%s" % (lib_filename, str(e)))
    return emsg


def load_library(signature, cache_params):
    """Load existing dynamic library from disk.

    Returns library module if found, otherwise None.

    If found, the module is placed in memory cache for later lookup_lib calls.
    """
    lib_filename = create_lib_filename(signature, cache_params)
    if not os.path.exists(lib_filename):
        debug("File %s does not exist" % (lib_filename,))
        return None
    debug("Loading %s from %s" % (signature, lib_filename))

    if cache_params["lib_loader"] == "ctypes":
        try:
            lib = ctypes.cdll.LoadLibrary(lib_filename)
        except os.error as e:
            lib = None
            emsg = analyse_load_error(e, lib_filename, cache_params)
            warning(emsg)
        else:
            debug("Loaded %s from %s" % (signature, lib_filename))
    elif cache_params["lib_loader"] == "import":
        sys.path.append(os.path.dirname(lib_filename))
        # Will raise an exception if it does not load correctly
        lib = __import__(signature)
        debug("Loaded %s from %s" % (signature, lib_filename))
    else:
        error("Invalid loader: %s" % cache_params["lib_loader"])

    if lib is not None:
        # Disk loading succeeded, register loaded library in memory
        # cache for next time
        _lib_cache[signature] = lib
    return lib


# A cache is always something to be careful about.  This one stores
# references to loaded jit-compiled libraries, which will stay in
# memory unless manually unloaded anyway and should not cause any
# trouble.
_lib_cache = {}


def lookup_lib(lib_signature, cache_params):
    """Lookup library in memory cache then in disk cache.

    Returns library module if found, otherwise None.
    """
    # Look for already loaded library in memory cache
    lib = _lib_cache.get(lib_signature)
    if lib is None:
        # Cache miss in memory, try looking on disk
        lib = load_library(lib_signature, cache_params)
    else:
        debug("Fetched %s from memory cache" % (lib_signature,))
    # Return library or None
    return lib


def read_src(signature, cache_params):
    """Lookup source code in disk cache and return file contents or None."""
    filename = create_src_filename(signature, cache_params)
    return read_textfile(filename)


def read_inc(signature, cache_params):
    """Lookup header file in disk cache and return file contents or None."""
    filename = create_inc_filename(signature, cache_params)
    return read_textfile(filename)


def read_log(signature, cache_params):
    """Lookup log file in disk cache and return file contents or None."""
    filename = create_log_filename(signature, cache_params)
    return read_textfile(filename)


def store_src(signature, content, cache_params):
    "Store source code in file within dijitso directories."
    make_src_dir(cache_params)
    filename = create_src_filename(signature, cache_params)
    store_textfile(filename, content)
    return filename


def store_inc(signature, content, cache_params):
    "Store header file within dijitso directories."
    make_inc_dir(cache_params)
    filename = create_inc_filename(signature, cache_params)
    store_textfile(filename, content)
    return filename


def store_log(signature, content, cache_params):
    "Store log file within dijitso directories."
    make_log_dir(cache_params)
    filename = create_log_filename(signature, cache_params)
    store_textfile(filename, content)
    return filename


def compress_source_code(src_filename, cache_params):
    """Keep, delete or compress source code based on value of cache parameter 'src_storage'.

    Can be "keep", "delete", or "compress".
    """
    src_storage = cache_params["src_storage"]
    if src_storage == "keep":
        filename = src_filename
    elif src_storage == "delete":
        try_delete_file(src_filename)
        filename = None
    elif src_storage == "compress":
        filename = gzip_file(src_filename)
        try_delete_file(src_filename)
    else:
        error("Invalid src_storage parameter. Expecting 'keep', 'delete', or 'compress'.")
    return filename


def get_dijitso_dependencies(libname, cache_params):
    "Run ldd and filter output to only include dijitso cache entries."
    libs = ldd(libname)
    dlibs = {}
    for k in libs:
        if k.startswith(cache_params["lib_prefix"]):
            dlibs[k] = libs[k]
    return dlibs


# TODO: Use this in command-line tools?
def check_cache_integrity(cache_params):
    "Check dijitso cache integrity."
    libnames = set(glob(cache_params["lib_prefix"] + "*" + cache_params["lib_postfix"]))
    dmissing = {}
    for libname in libnames:
        dlibs = get_dijitso_dependencies(libname, cache_params)
        # Missing on file system:
        missing = [k for k in dlibs if k not in libnames]
        for k in dlibs:
            if k not in missing:
                # ldd thinks file is missing but it's there, linker issue?
                pass
        if missing:
            dmissing[libname] = sorted(missing)
    return dmissing


def report_cache_integrity(dmissing, out=warning):
    "Print cache integrity report."
    if dmissing:
        out("%d libraries are missing one or more dependencies:" % len(dmissing))
        for k in sorted(dmissing):
            out("\t%s depends on missing libraries:" % k)
            for m in dmissing[k]:
                out("\t\t%s" % m)
