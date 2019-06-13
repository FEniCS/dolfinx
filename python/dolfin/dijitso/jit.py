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

"""This module contains the main jit() function and related utilities."""

import ctypes
import numpy

from dolfin.dijitso.log import error
from dolfin.dijitso.params import validate_params
from dolfin.dijitso.str import as_unicode
from dolfin.dijitso.cache import lookup_lib, load_library
from dolfin.dijitso.cache import write_library_binary, read_library_binary
from dolfin.dijitso.build import build_shared_library
from dolfin.dijitso.signatures import hash_params


class DijitsoError(RuntimeError):
    def __init__(self, message, err_info):
        super(DijitsoError, self).__init__(message)
        self.err_info = err_info


def extract_factory_function(lib, name):
    """Extract function from loaded library.

    Assuming signature ``(void *)()``, for anything else use look at
    ctypes documentation.

    Returns the factory function or raises error.
    """
    function = getattr(lib, name)
    function.restype = ctypes.c_void_p
    return function


def jit_signature(name, params):  # TODO: Unused?
    """Compute the signature that jit will use for given name and params."""

    # Validation and completion with defaults for missing parameters
    params = validate_params(params)

    # Extend provided name of jitable with hash of relevant parameters
    signature_params = {
        "generator": params["generator"],
        "build": params["build"]
    }

    signature = "%s_%s" % (name, hash_params(signature_params))
    return signature


# TODO: send, receive, wait functionality is not currently in use,
# decide to use it from dolfin or clean up the code and comments here.
def jit(jitable, name, params, generate=None,
        send=None, receive=None, wait=None):
    """Just-in-time compile and import of a shared library with a cache mechanism.

    A signature is computed from the name, params["generator"],
    and params["build"]. The name should be a unique identifier
    for the jitable, preferrably produced by a good hash function.

    The signature is used to identity if the library has already been
    compiled and cached. A two-level memory and disk cache ensures good
    performance for repeated lookups within a single program as well as
    persistence across program runs.

    If no library has been cached, the passed 'generate' function is
    called to generate the source code:

        header, source, dependencies = \
            generate(jitable, name, signature, params["generator"])

    It is expected to translate the 'jitable' object into
    C or C++ (default) source code which will subsequently be
    compiled as a shared library and stored in the disk cache.
    The returned 'dependencies' should be a tuple of signatures
    returned from other completed dijitso.jit calls, and are
    linked to when building.

    The compiled shared library is then loaded with ctypes and returned.

    For use in a parallel (MPI) context, three functions send, receive,
    and wait can be provided. Each process can take on a different role
    depending on whether generate, or receive, or neither is provided.

      * Every process that gets a generate function is called a 'builder',
        and will generate and compile code as described above on a cache miss.
        If the function send is provided, it will then send the shared library
        binary file as a binary blob by calling send(numpy_array).

      * Every process that gets a receive function is called a 'receiver',
        and will call 'numpy_array = receive()' expecting the binary blob
        with a compiled binary shared library which will subsequently be
        written to file in the local disk cache.

      * The rest of the processes are called 'waiters' and will do nothing.

      * If provided, all processes will call wait() before attempting to
        load the freshly compiled library from disk cache.

    The intention of the above pattern is to be flexible, allowing several
    different strategies for sharing build results. The user of dijitso
    can determine groups of processes that share a disk cache, and assign
    one process per physical disk cache directory to write to that directory,
    avoiding multiple processes writing to the same files.

    This forms the basis for three main strategies:

      * Build on every process.

      * Build on one process per physical cache directory.

      * Build on a single global root node and send a copy of
        the binary to one process per physical cache directory.

    It is highly recommended to avoid have multiple builder processes
    sharing a physical cache directory.
    """
    # TODO: Could simplify interface here and roll
    #   (jitable, name, params["generator"]) into a single jitobject?
    # TODO: send/receive doesn't combine well with generate
    #   triggering additional jit calls for dependencies.
    #   It's possible that dependencies are hard to determine without
    #   generate doing some analysis that we want to avoid.
    #   Drop send/receive? Probably not that useful anyway.

    # Complete params with hardcoded defaults and config file defaults
    params = validate_params(params)

    # 0) Look for library in memory or disk cache
    # FIXME: use only name as signature for now
    # TODO: just remove one of signature or name from API?
    # signature = jit_signature(name, params)
    name = as_unicode(name)
    signature = name
    cache_params = params["cache"]
    lib = lookup_lib(signature, cache_params)
    err_info = None

    if lib is None:
        # Since we didn't find the library in cache, we must build it.

        if receive and generate:
            # We're not supposed to generate if we're receiving
            error("Please provide only one of generate or receive.")

        elif generate:
            # 1) Generate source code
            header, source, dependencies = generate(jitable, name, signature, params["generator"])
            # Ensure we got unicode from generate
            header = as_unicode(header)
            source = as_unicode(source)
            dependencies = [as_unicode(dep) for dep in dependencies]

            # 2) Compile shared library and 3) store in dijitso
            # inc/src/lib dir on success
            # NB! It's important to not raise exception on compilation
            # failure, such that we can reach wait() together with
            # other processes if any.
            status, output, lib_filename, err_info = \
                build_shared_library(signature, header, source, dependencies,
                                     params)

            # 4a) Send library over network if we have a send function
            if send:
                if status == 0:
                    lib_data = read_library_binary(lib_filename)
                else:
                    lib_data = numpy.zeros((1,))
                send(lib_data)

        elif receive:
            # 4b) Get library as binary blob from given receive
            # function and store in cache
            lib_data = receive()
            # Empty if compilation failed
            status = -1 if lib_data.shape == (1,) else 0
            if status == 0:
                write_library_binary(lib_data, signature, cache_params)

        else:
            # Do nothing (we'll be waiting below for other process to
            # build)
            if not wait:
                error("Please provide wait if not providing one of generate or receive.")

        # 5) Notify waiters that we're done / wait for builder to
        # notify us
        if wait:
            wait()

        # Finally load library from disk cache (places in memory
        # cache)
        # NB! This returns None if the file does not exist,
        # i.e. if compilation failed on builder process
        lib = load_library(signature, cache_params)

    if err_info:
        # TODO: Parse output to find error(s) for better error messages
        raise DijitsoError("Dijitso JIT compilation failed, see '%s' for details"
                           % err_info['fail_dir'], err_info)

    # Return built library and its signature
    return lib, signature
