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

"""Utilities for building libraries with dijitso."""

import tempfile
import os
import sys

from dolfin.dijitso.system import get_status_output, lockfree_move_file
from dolfin.dijitso.system import make_dirs, make_executable, store_textfile
from dolfin.dijitso.log import warning, info, debug
from dolfin.dijitso.cache import ensure_dirs, make_lib_dir, make_inc_dir
from dolfin.dijitso.cache import create_fail_dir_path
from dolfin.dijitso.cache import create_lib_filename, create_lib_basename, create_libname
from dolfin.dijitso.cache import create_src_filename, create_src_basename
from dolfin.dijitso.cache import create_inc_filename, create_inc_basename
from dolfin.dijitso.cache import create_log_filename
from dolfin.dijitso.cache import compress_source_code


def make_unique(dirs):
    """Take a sequence of hashable items and return a tuple including each
    only once.

    Preserves original ordering.

    """
    udirs = []
    found = set()
    for d in dirs:
        if d not in found:
            udirs.append(d)
            found.add(d)
    return tuple(udirs)


def make_compile_command(src_filename, lib_filename, dependencies,
                         build_params, cache_params):
    """Piece together the compile command from build params.

    Returns the command as a list with the command and its arguments.
    """
    # Get dijitso dirs based on cache_params
    inc_dir = make_inc_dir(cache_params)
    lib_dir = make_lib_dir(cache_params)

    # Add dijitso directories to includes, libs, and rpaths
    include_dirs = make_unique(build_params["include_dirs"] + (inc_dir,))
    lib_dirs = make_unique(build_params["lib_dirs"] + (lib_dir,))
    rpath_dirs = make_unique(build_params["rpath_dirs"] + (lib_dir,))

    # Make all paths absolute
    include_dirs = [os.path.abspath(d) for d in include_dirs]
    lib_dirs = [os.path.abspath(d) for d in lib_dirs]
    rpath_dirs = [os.path.abspath(d) for d in rpath_dirs]

    # Build options (defaults assume gcc compatibility)
    cxxflags = list(build_params["cxxflags"])
    if build_params["debug"]:
        cxxflags.extend(build_params["cxxflags_debug"])
    else:
        cxxflags.extend(build_params["cxxflags_opt"])

    # Create library names for all dependencies and additional given
    # libs
    deplibs = [create_libname(depsig, cache_params)
               for depsig in dependencies]

    deplibs.extend(build_params["libs"])

    # Get compiler name
    args = [build_params["cxx"]]

    # Compiler args
    args.extend(cxxflags)
    args.extend("-I" + path for path in include_dirs)

    # The input source
    args.append(src_filename)

    # Linker args
    args.extend("-L" + path for path in lib_dirs)
    args.extend("-Wl,-rpath," + path for path in rpath_dirs)
    args.extend("-l" + lib for lib in deplibs)

    # OSX specific:
    if sys.platform == "darwin":
        full_lib_filename = os.path.join(cache_params["cache_dir"],
                                         cache_params["lib_dir"],
                                         os.path.basename(lib_filename))
        args.append("-Wl,-install_name,%s" % full_lib_filename)

    # The output library
    args.append("-o" + lib_filename)

    return args


def temp_dir(cache_params):
    """Return a uniquely named temp directory.

    Optionally residing under temp_dir_root from cache_params.
    """
    return tempfile.mkdtemp(dir=cache_params["temp_dir_root"])


def build_shared_library(signature, header, source, dependencies, params):
    """Build shared library from a source file and store library in
    cache.

    """
    cache_params = params["cache"]
    build_params = params["build"]

    # Create basenames
    inc_basename = create_inc_basename(signature, cache_params)
    src_basename = create_src_basename(signature, cache_params)
    lib_basename = create_lib_basename(signature, cache_params)

    # Create a temp directory and filenames within it
    tmpdir = temp_dir(cache_params)
    temp_inc_filename = os.path.join(tmpdir, inc_basename)
    temp_src_filename = os.path.join(tmpdir, src_basename)
    temp_lib_filename = os.path.join(tmpdir, lib_basename)

    # Store source and header in temp dir
    if header:
        store_textfile(temp_inc_filename, header)
    store_textfile(temp_src_filename, source)

    # Build final command as list of arguments
    cmd = make_compile_command(temp_src_filename, temp_lib_filename,
                               dependencies, build_params, cache_params)

    # Execute command to compile generated source code to dynamic
    # library
    status, output = get_status_output(cmd)

    # Move files to cache on success or a local dir on failure,
    # using safe lockfree move
    if status == 0:
        # Ensure dirnames exist in cache dirs
        ensure_dirs(cache_params)

        # Move library first
        lib_filename = create_lib_filename(signature, cache_params)
        assert os.path.exists(os.path.dirname(lib_filename))
        lockfree_move_file(temp_lib_filename, lib_filename)

        # Write header only if there is one
        if header:
            inc_filename = create_inc_filename(signature, cache_params)
            assert os.path.exists(os.path.dirname(inc_filename))
            lockfree_move_file(temp_inc_filename, inc_filename)
        else:
            inc_filename = None

        # Compress or delete source code based on params
        temp_src_filename = compress_source_code(temp_src_filename, cache_params)
        if temp_src_filename:
            src_filename = create_src_filename(signature, cache_params)
            if temp_src_filename.endswith(".gz"):
                src_filename = src_filename + ".gz"
            assert os.path.exists(os.path.dirname(src_filename))
            lockfree_move_file(temp_src_filename, src_filename)
        else:
            src_filename = None

        # Write compiler command and output to log file
        if cache_params["enable_build_log"]:
            # Recreate compiler command without the tempdir
            cmd = make_compile_command(src_basename, lib_basename,
                                       dependencies, build_params, cache_params)

            log_contents = "%s\n\n%s" % (" ".join(cmd), output)
            log_filename = create_log_filename(signature, cache_params)
            assert os.path.exists(os.path.dirname(log_filename))
            store_textfile(log_filename, log_contents)
        else:
            log_filename = None

        files = set((inc_filename, src_filename, lib_filename, log_filename))
        files = files - set((None,))
        files = sorted(files)
        debug("Compilation succeeded. Files written to cache:\n"
              + "\n".join(files))
        err_info = None
    else:
        # Create filenames in a local directory to store files for
        # reproducing failure
        fail_dir = create_fail_dir_path(signature, cache_params)
        make_dirs(fail_dir)

        # Library name is returned below
        lib_filename = None

        # Write header only if there is one
        if header:
            inc_filename = os.path.join(fail_dir, inc_basename)
            lockfree_move_file(temp_inc_filename, inc_filename)

        # Always write source for inspection after compilation failure
        src_filename = os.path.join(fail_dir, src_basename)
        lockfree_move_file(temp_src_filename, src_filename)

        # Write compile command to failure dir, adjusted to use local
        # source file name so it can be rerun
        cmd = make_compile_command(src_basename, lib_basename, dependencies,
                                   build_params, cache_params)
        cmds = " ".join(cmd)
        script = "#!/bin/bash\n# Execute this file to recompile locally\n" + cmds
        cmd_filename = os.path.join(fail_dir, "recompile.sh")
        store_textfile(cmd_filename, script)
        make_executable(cmd_filename)

        # Write readme file with instructions
        readme = "Run or source recompile.sh to compile locally and reproduce the build failure.\n"
        readme_filename = os.path.join(fail_dir, "README")
        store_textfile(readme_filename, readme)

        # Write compiler output to failure dir (will refer to temp paths)
        log_filename = os.path.join(fail_dir, "error.log")
        store_textfile(log_filename, output)

        info("------------------- Start compiler output ------------------------")
        info(output)
        info("-------------------  End compiler output  ------------------------")
        warning("Compilation failed! Sources, command, and "
                "errors have been written to: %s" % (fail_dir,))

        err_info = {'src_filename': src_filename,
                    'cmd_filename': cmd_filename,
                    'readme_filename': readme_filename,
                    'fail_dir': fail_dir,
                    'log_filename': log_filename}

    return status, output, lib_filename, err_info
