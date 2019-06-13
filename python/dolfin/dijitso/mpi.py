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

"""Utilities for mpi features of dijitso."""

import io
import os
import uuid
from glob import glob

import numpy

from dolfin.dijitso.log import info, error
from dolfin.dijitso.system import try_delete_file


def bcast_uuid(comm):
    "Create a unique id shared across all processes in comm."
    guid = numpy.ndarray((1,), dtype=numpy.uint64)
    if comm.rank == 0:
        # uuid creates a unique 128 bit id, we just pick the low 64 bits
        guid[0] = numpy.uint64(uuid.uuid4().int & ((1 << 64) - 1))
    comm.Bcast(guid, root=0)
    return int(guid[0])


def discover_path_access_ranks(comm, path):
    """Discover which ranks share access to the same directory.

    This cannot be done by comparing paths, because
    a path string can represent a local work directory
    or a network mapped directory, depending on cluster
    configuration.

    Current approach is that each process touches a
    filename with its own rank in their given path.
    By reading in the filelist from the same path,
    we'll find which ranks have access to the same
    directory.

    To avoid problems with leftover files from previous
    program crashes, or collisions between simultaneously
    running programs, we use a random uuid in the filenames
    written.
    """
    # Create a unique basename for rank files of this program
    guid = bcast_uuid(comm)  # TODO: Run this in an init function and store for program duration?
    basename = os.path.join(path, "rank.%d." % guid)

    # Write the rank of this process to a filename
    filename = basename + str(comm.rank)
    with io.open(filename, "wb"):
        pass

    # Wait for all writes to take place. Don't know how robust this is
    # with nfs!!!
    comm.Barrier()

    # Read filelist
    noderanks = sorted([int(fn.replace(basename, "")) for fn in glob(basename + "*")])

    # Wait for everyone to finish reading filelist
    comm.Barrier()

    # Clean up our own rank file. If the process is aborted,
    # this may fail to happen and leave a dangling file!
    # However the file takes no space, and the guid ensures
    # it won't be colliding with other filenames.
    # TODO: Include a gc command in dijitso to clean up this and other stuff.
    try_delete_file(filename)
    return noderanks


def gather_global_partitions(comm, partition):
    """Gather an ordered list of unique partition values within comm."""
    global_partitions = numpy.ndarray((comm.size,), dtype=numpy.uint64)
    local_partition = numpy.ndarray((1,), dtype=numpy.uint64)
    local_partition[0] = partition
    comm.Allgather(local_partition, global_partitions)
    return sorted(set(global_partitions))


def create_subcomm(comm, ranks):
    "Create a communicator for a set of ranks."
    group = comm.Get_group()
    subgroup = group.Incl(ranks)
    subcomm = comm.Create(subgroup)
    subgroup.Free()
    group.Free()
    return subcomm


def create_node_comm(comm, comm_dir):
    """Create comms for communicating within a node."""
    # Find ranks that share this physical comm_dir (physical dir, not same path string)
    node_ranks = discover_path_access_ranks(comm, comm_dir)

    # Partition comm into one communicator for each physical comm_dir
    assert len(node_ranks) >= 1
    node_root = min(node_ranks)
    node_comm = comm.Split(node_root, node_ranks.index(comm.rank))
    return node_comm, node_root


def create_node_roots_comm(comm, node_root):
    """Build comm for communicating among the node roots."""
    unique_global_node_roots = gather_global_partitions(comm, node_root)
    roots_comm = create_subcomm(comm, unique_global_node_roots)
    return roots_comm


def create_comms_and_role_root(comm, node_comm, node_root):
    """Approach: global root builds and sends binary to node roots,
    everyone waits on their node group."""
    copy_comm = create_node_roots_comm(comm, node_root)
    wait_comm = node_comm
    if comm.rank == 0:
        role = "builder"
    elif node_comm.rank == 0:
        assert comm.rank == node_root
        role = "receiver"
    else:
        assert comm.rank != node_root
        role = "waiter"
    return copy_comm, wait_comm, role


def create_comms_and_role_node(comm, node_comm, node_root):
    """Approach: each node root builds, everyone waits on their node group."""
    copy_comm = None
    wait_comm = node_comm
    if node_comm.rank == 0:
        assert comm.rank == node_root
        role = "builder"
    else:
        assert comm.rank != node_root
        role = "waiter"
    return copy_comm, wait_comm, role


def create_comms_and_role_process(comm, node_comm, node_root):
    """Approach: each process builds its own module, no communication.

    To ensure no race conditions in this case independently of cache dir setup,
    we include an error check on the size of the autodetected node_comm.
    This should always be 1, or we provide the user with an informative message.
    TODO: Append program uid and process rank to basedir instead?
    """
    if node_comm.size > 1:
        error("Asking for per-process building but processes share cache dir."
              " Please configure dijitso dirs to be distinct per process.")
    copy_comm = None
    wait_comm = None
    assert node_comm.rank == 0
    assert comm.rank == node_root
    role = "builder"
    return copy_comm, wait_comm, role


def create_comms_and_role(comm, comm_dir, buildon):
    """Determine which role each process should take, and create
    the right copy_comm and wait_comm for the build strategy.

    buildon must be one of "root", "node", or "process".

    Returns (copy_comm, wait_comm, role).
    """
    # Now assign values to the copy_comm, wait_comm, and role,
    # depending on buildon strategy chosen.  If we have no comm,
    # always return the builder role
    if comm is None:
        copy_comm, wait_comm, role = None, None, "builder"
    else:
        node_comm, node_root = create_node_comm(comm, comm_dir)
        if buildon == "root":
            copy_comm, wait_comm, role = create_comms_and_role_root(comm,
                                                                    node_comm,
                                                                    node_root)
        elif buildon == "node":
            copy_comm, wait_comm, role = create_comms_and_role_node(comm,
                                                                    node_comm,
                                                                    node_root)
        elif buildon == "process":
            copy_comm, wait_comm, role = create_comms_and_role_process(comm,
                                                                       node_comm,
                                                                       node_root)
        else:
            error("Invalid parameter buildon=%s" % (buildon,))
    return copy_comm, wait_comm, role


def send_binary(comm, lib_data):
    "Send compiled library as binary blob over MPI."
    # TODO: Test this in parallel locally.
    # TODO: Test this in parallel on clusters.
    # http://mpi4py.scipy.org/docs/usrman/tutorial.html
    # Check that we are the root
    root = 0
    assert comm.rank == root

    # Send file size
    lib_size = numpy.ndarray((1,), dtype=numpy.uint32)
    lib_size[0] = lib_data.shape[0]
    info("rank %d: send size with root=%d." % (comm.rank, root))
    comm.Bcast(lib_size, root=root)

    # Send file contents
    info("rank %d: send data with root=%d." % (comm.rank, root))
    comm.Bcast(lib_data, root=root)


def receive_binary(comm):
    "Store shared library received as a binary blob to cache."
    # Check that we are not the root
    root = 0
    assert comm.rank != root

    # Receive file size
    lib_size = numpy.ndarray((1,), dtype=numpy.uint32)
    info("rank %d: receive size with root=%d." % (comm.rank, root))
    comm.Bcast(lib_size, root=root)

    # Receive file contents
    lib_data = numpy.ndarray(lib_size[0], dtype=numpy.uint8)
    info("rank %d: receive data with root=%d." % (comm.rank, root))
    comm.Bcast(lib_data, root=root)

    return lib_data


"""
def foo():
    # TODO: Should call these once (for each comm at least) globally
    # in dolfin, not on each jit call

    def get_comm_dir(cache_params):
        return os.path.join(cache_params["cache_dir"], cache_params["comm_dir"])

    comm_dir = get_comm_dir()
    copy_comm, wait_comm, role = create_comms_and_role(comm, comm_dir, buildon)

    if wait_comm is not None:
        def wait():
            wait_comm.Barrier()
    else:
        wait = None

    if copy_comm is not None and copy_comm.size > 1:
        def send(lib_data):
            send_binary(copy_comm, lib_data)
    else:
        send = None

    def receive():
        return receive_binary(copy_comm)
"""
