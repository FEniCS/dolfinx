# Copyright (C) 2018-2026 Michal Habera, Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""General tools for timing and configuration."""

import datetime
import functools
import typing
from collections.abc import Callable, Sequence

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx.cpp.common import (
    git_commit_hash,
    hardware_concurrency,
    has_adios2,
    has_complex_ufcx_kernels,
    has_debug,
    has_kahip,
    has_parmetis,
    has_petsc,
    has_petsc4py,
    has_ptscotch,
    has_slepc,
    has_superlu_dist,
    local_range,
    ufcx_signature,
)

__all__ = [
    "IndexMap",
    "Reduction",
    "Scatterer",
    "Timer",
    "create_sub_index_map",
    "git_commit_hash",
    "hardware_concurrency",
    "has_adios2",
    "has_complex_ufcx_kernels",
    "has_debug",
    "has_kahip",
    "has_parmetis",
    "has_petsc",
    "has_petsc4py",
    "has_ptscotch",
    "has_slepc",
    "has_superlu_dist",
    "index_map",
    "list_timings",
    "local_range",
    "scatterer",
    "timed",
    "timing",
    "ufcx_signature",
]

Reduction = _cpp.common.Reduction

# Default MPI tag of the consensus exchange used when building a ghosted
# index map (dolfinx::MPI::tag::consensus_nbx).
_CONSENSUS_NBX_TAG = _cpp.common.consensus_nbx_tag

_ScatterArray: typing.TypeAlias = npt.NDArray[
    np.int64 | np.float32 | np.float64 | np.complex64 | np.complex128
]


class IndexMap:
    """Map indices across processes.

    An index map describes the parallel distribution of a range of
    indices. Each index is owned by exactly one process. A process holds
    the indices it owns, numbered ``[0, size_local)`` locally, followed
    by the 'ghost' indices it holds but does not own, numbered
    ``[size_local, size_local + num_ghosts)``.
    """

    _cpp_object: _cpp.common.IndexMap

    def __init__(self, imap: _cpp.common.IndexMap):
        """Create an index map.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`index_map` to create an index
            map.

        Args:
            imap: C++ IndexMap object.
        """
        self._cpp_object = imap

    def __eq__(self, other: object) -> bool:
        """Check that two wrappers hold the same underlying C++ index map.

        Note:
            This is identity of the wrapped object, not equivalence of
            the distribution it describes. Two separately constructed
            index maps do not compare equal, even if identical.
        """
        if not isinstance(other, IndexMap):
            return NotImplemented
        return self._cpp_object == other._cpp_object

    def __hash__(self) -> int:
        """Hash of the wrapped index map."""
        return hash(self._cpp_object)

    @property
    def comm(self) -> _MPI.Comm:
        """MPI communicator that the index map is distributed over."""
        return self._cpp_object.comm

    @property
    def size_local(self) -> int:
        """Number of indices owned by the calling process."""
        return self._cpp_object.size_local

    @property
    def size_global(self) -> int:
        """Number of indices across all processes."""
        return self._cpp_object.size_global

    @property
    def num_ghosts(self) -> int:
        """Number of ghost indices on the calling process."""
        return self._cpp_object.num_ghosts

    @property
    def local_range(self) -> tuple[int, int]:
        """Range of global indices owned by the calling process."""
        return self._cpp_object.local_range

    @property
    def ghosts(self) -> npt.NDArray[np.int64]:
        """Global index of each ghost index.

        Note:
            The returned array is a read-only view.
        """
        return self._cpp_object.ghosts

    @property
    def owners(self) -> npt.NDArray[np.int32]:
        """Owning rank of each ghost index.

        Note:
            The returned array is a read-only view.
        """
        return self._cpp_object.owners

    def index_to_dest_ranks(self, tag: int) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Ranks that ghost each owned index, as an adjacency list.

        Args:
            tag: MPI tag used by the consensus exchange. Must be the
                same on all ranks, and must not clash with another
                in-flight exchange.

        Returns:
            Ghosting ranks of each owned index, as a (data, offsets)
            pair.
        """
        return self._cpp_object.index_to_dest_ranks(tag)

    def local_to_global(self, local: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
        """Map local indices to global indices.

        Args:
            local: Local indices.

        Returns:
            Global index of each entry of ``local``.
        """
        return self._cpp_object.local_to_global(local)

    def global_to_local(self, global_index: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
        """Map global indices to local indices.

        Args:
            global_index: Global indices.

        Returns:
            Local index of each entry of ``global_index``, with ``-1``
            for indices that are not owned or ghosted by the caller.
        """
        return self._cpp_object.global_to_local(global_index)


class Scatterer:
    """Scatter and gather data with a layout described by an ``IndexMap``.

    A scatterer is stateless: it holds only the communication pattern
    derived from an :class:`IndexMap`, and does not track buffers or
    the status of in-flight MPI requests. Callers of ``scatter_fwd_begin``/
    ``scatter_rev_begin`` are responsible for managing the send/receive
    buffers and the returned request, and can share one scatterer
    between multiple objects (e.g. :class:`dolfinx.la.Vector`) that use
    the same index map.

    A forward scatter sends data associated with owned/local indices
    to the ranks that ghost them; a reverse scatter sends ghost data
    back to the owning ranks, to be accumulated into the owned data.
    Both use the same two-step begin/end pattern, splitting the
    non-blocking exchange from its completion so that unrelated work
    can be done while communication is in flight. A round trip for a
    forward scatter with block size 1, where ``x`` holds the owned
    data and ``x_ghost`` the ghost data::

        local_idx = sc.local_indices_block
        remote_idx = sc.remote_indices_block

        send_buffer = x[local_idx]
        recv_buffer = np.empty(remote_idx.size, dtype=x.dtype)
        request = sc.scatter_fwd_begin(send_buffer, recv_buffer, 1)
        # ... unrelated work can be done here while communication is
        # in flight, but send_buffer/recv_buffer must not be touched ...
        sc.scatter_fwd_end(request)
        x_ghost[remote_idx] = recv_buffer

    A reverse scatter follows the same pattern with the roles of
    ``local_indices_block``/``remote_indices_block`` and of
    ``send_buffer``/``recv_buffer`` swapped, and accumulating (rather
    than assigning) into the destination array; see
    :meth:`scatter_rev_begin` and :meth:`scatter_rev_end`.
    """

    _cpp_object: _cpp.common.Scatterer

    def __init__(self, s: _cpp.common.Scatterer):
        """Create a scatterer.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`scatterer` to create a
            scatterer object.

        Args:
            s: C++ Scatterer object.
        """
        self._cpp_object = s

    def __eq__(self, other: object) -> bool:
        """Check that two wrappers hold the same scatterer."""
        if not isinstance(other, Scatterer):
            return NotImplemented
        return self._cpp_object == other._cpp_object

    def __hash__(self) -> int:
        """Hash of the wrapped scatterer."""
        return hash(self._cpp_object)

    @property
    def local_indices_block(self) -> npt.NDArray[np.int32]:
        """Indices for packing/unpacking owned data in a send/recv buffer.

        For a forward scatter, used to copy owned entries into a send
        buffer. For a reverse scatter, used to accumulate received
        values into the owned entries. Blocked: for block size ``bs``
        a buffer holds ``bs`` values per index and must be
        ``bs * local_indices_block.size`` long.
        """
        return self._cpp_object.local_indices_block

    @property
    def remote_indices_block(self) -> npt.NDArray[np.int32]:
        """Indices for packing/unpacking ghost data in a send/recv buffer.

        For a forward scatter, used to unpack received values into
        ghost entries. For a reverse scatter, used to pack ghost
        entries into a send buffer. Blocked: for block size ``bs`` a
        buffer holds ``bs`` values per index and must be
        ``bs * remote_indices_block.size`` long.
        """
        return self._cpp_object.remote_indices_block

    def scatter_fwd_begin(
        self, send_buffer: _ScatterArray, recv_buffer: _ScatterArray, bs: int = 1
    ) -> _MPI.Request:
        """Start a non-blocking exchange of owned data with ghosting ranks.

        The communication is completed by calling
        :meth:`scatter_fwd_end`. See :attr:`local_indices_block` for
        how to pack ``send_buffer`` and :attr:`remote_indices_block`
        for how to unpack ``recv_buffer``.

        Note:
            Collective. Every rank in the communicator must call this,
            including ranks without neighbours.

        Note:
            ``send_buffer``/``recv_buffer`` must not be changed or
            accessed until after a call to :meth:`scatter_fwd_end`.

        Args:
            send_buffer: Packed owned data, blocked by ``bs``, sized
                ``bs * local_indices_block.size``.
            recv_buffer: Buffer for storing received data, blocked by
                ``bs``, sized ``bs * remote_indices_block.size``.
            bs: Number of values per index map index.

        Returns:
            Request to pass to :meth:`scatter_fwd_end`.
        """
        return self._cpp_object.scatter_fwd_begin(
            send_buffer,
            recv_buffer,
            bs,
        )

    def scatter_fwd_end(self, request: _MPI.Request) -> None:
        """Complete the exchange started by :meth:`scatter_fwd_begin`.

        Note:
            Local completion of the caller's own request, not itself
            collective. Every rank that called
            :meth:`scatter_fwd_begin` must still call this before
            reusing the buffers.

        Args:
            request: Request returned by :meth:`scatter_fwd_begin`.
        """
        self._cpp_object.scatter_fwd_end(request)

    def scatter_rev_begin(
        self, send_buffer: _ScatterArray, recv_buffer: _ScatterArray, bs: int = 1
    ) -> _MPI.Request:
        """Start a non-blocking exchange of ghost data with owning ranks.

        The communication is completed by calling
        :meth:`scatter_rev_end`. See :attr:`remote_indices_block` for
        how to pack ``send_buffer`` and :attr:`local_indices_block`
        for how to unpack (accumulate into) ``recv_buffer``.

        Note:
            Collective. Every rank in the communicator must call this,
            including ranks without neighbours.

        Note:
            ``send_buffer``/``recv_buffer`` must not be changed or
            accessed until after a call to :meth:`scatter_rev_end`.

        Args:
            send_buffer: Data associated with each ghost index,
                blocked by ``bs``, sized ``bs *
                remote_indices_block.size``.
            recv_buffer: Buffer for storing received data, blocked by
                ``bs``, sized ``bs * local_indices_block.size``.
            bs: Number of values per index map index.

        Returns:
            Request to pass to :meth:`scatter_rev_end`.
        """
        return self._cpp_object.scatter_rev_begin(
            send_buffer,
            recv_buffer,
            bs,
        )

    def scatter_rev_end(self, request: _MPI.Request) -> None:
        """Complete the exchange started by :meth:`scatter_rev_begin`.

        Note:
            Local completion of the caller's own request, not itself
            collective. Every rank that called
            :meth:`scatter_rev_begin` must still call this before
            reusing the buffers.

        Args:
            request: Request returned by :meth:`scatter_rev_begin`.
        """
        self._cpp_object.scatter_rev_end(request)


def index_map(
    comm: _MPI.Comm,
    local_size: int,
    ghosts: tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]] | None = None,
    *,
    dest_src: Sequence[npt.NDArray[np.int32]] | None = None,
    tag: int = _CONSENSUS_NBX_TAG,
) -> IndexMap:
    """Create an index map.

    Note:
        Collective. ``ghosts`` must be ``None`` on every process or
        given on every process, and likewise for ``dest_src``. This is
        a precondition and is not checked, since checking it would
        require communication on every call.

    Args:
        comm: MPI communicator to distribute the indices over.
        local_size: Number of indices owned by the calling process.
        ghosts: Tuple ``(ghost_indices, owners)`` of global ghost
            indices and their owning ranks. If ``None``, the index map
            is non-overlapping and ``ghosts`` must be ``None`` on every
            process. For an overlapping map, a process with no ghosts
            must pass empty arrays.
        dest_src: Pair ``(dest, src)`` of destination and source rank
            arrays. ``dest`` lists ranks that ghost caller-owned
            indices; ``src`` lists ranks that own the caller's ghosts
            and must equal the unique values in ``owners``. Both arrays
            must be sorted, unique, and contain valid ranks. Supplying
            them avoids the consensus exchange that otherwise discovers
            which ranks ghost the caller's owned indices.
        tag: MPI tag for the consensus exchange. Ignored if ``dest_src``
            is given. Must be the same on all ranks, and must not clash
            with another in-flight exchange.

    Returns:
        A new index map.
    """
    if ghosts is None:
        if dest_src is not None:
            raise ValueError("'dest_src' given without 'ghosts'.")
        return IndexMap(_cpp.common.IndexMap(comm, local_size))

    ghost_indices, owners = ghosts
    if dest_src is not None:
        return IndexMap(
            _cpp.common.IndexMap(comm, local_size, list(dest_src), ghost_indices, owners)
        )
    return IndexMap(_cpp.common.IndexMap(comm, local_size, ghost_indices, owners, tag))


def create_sub_index_map(
    imap: IndexMap,
    indices: npt.NDArray[np.int32],
) -> tuple[IndexMap, npt.NDArray[np.int32], bool]:
    """Create an index map for a subset of the indices of an index map.

    An index that is included by a process that ghosts it, but not by
    its owner, is re-assigned to one of the including processes.

    Note:
        Collective.

    Args:
        imap: Index map to build a sub-map of.
        indices: Local indices of ``imap``, unique and in range, to
            include in the sub-map.

    Returns:
        The sub-map, the index in ``imap`` of each of its indices, and
        whether any index acquired a new owner.

    Note:
        The owner-change flag is rank-local and is not reduced, so it
        can differ across ranks. Reduce it (e.g. ``comm.allreduce(...,
        op=MPI.LOR)``) before using it in a collective decision;
        branching on the unreduced value can leave some ranks in a
        collective that others have skipped.
    """
    submap, submap_to_map, owners_changed = _cpp.common.create_sub_index_map(
        imap._cpp_object, indices
    )
    return IndexMap(submap), submap_to_map, owners_changed


def scatterer(index_map: IndexMap) -> Scatterer:
    """Create a scatterer for data with a layout described by an index map.

    Args:
        index_map: Index map that describes the parallel layout of
            the data.

    Returns:
        A new scatterer.
    """
    return Scatterer(_cpp.common.Scatterer(index_map._cpp_object))


def timing(task: str) -> tuple[int, datetime.timedelta]:
    """Return the logged elapsed time.

    Timing data is for the calling process.

    Arguments:
        task: The task name using when logging the time.

    Returns:
        (number of times logged, total wall time)
    """
    return _cpp.common.timing(task)


def list_timings(comm: _MPI.Comm, reduction: _cpp.common.Reduction = Reduction.max) -> None:
    """Print out a summary of all Timer measurements.

    When used in parallel, a reduction is applied across all processes.
    By default, the maximum time is shown.
    """
    _cpp.common.list_timings(comm, reduction)


class Timer:
    """A timer for timing section of code.

    The recommended usage is with a context manager.

    Example:
        With a context manager, the timer is started when entering
        and stopped at exit. With a named :class:`Timer`::

            with Timer("Some costly operation"):
                costly_call_1()
                costly_call_2()

            delta = timing("Some costly operation")
            print(delta)

        or with an un-named :class:`Timer`::

            with Timer() as t:
                costly_call_1()
                costly_call_2()
                print(f"Elapsed time: {t.elapsed()}")

    Example:
        It is possible to start and stop a timer explicitly::

            t = Timer("Some costly operation")
            costly_call()
            delta = t.stop()

        and retrieve timing data using::

            delta = t.elapsed()

        To flush the timing data for a named :class:`Timer` to the logger,
        the timer should be stopped and flushed::

            t.stop()
            t.flush()

    Timings are stored globally (if task name is given) and once flushed
    (if used without a context manager) may be printed using functions
    :func:`timing` and :func:`list_timings`, e.g.::

        list_timings(comm)
    """

    _cpp_object: _cpp.common.Timer

    def __init__(self, name: str | None = None):
        """Create timer.

        Args:
            name: Identifier to use when storing elapsed time in logger.
        """
        self._cpp_object = _cpp.common.Timer(name)

    def __enter__(self) -> typing.Self:
        """Start timer."""
        self._cpp_object.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop timer and flush timing data to logger."""
        self._cpp_object.stop()
        self._cpp_object.flush()

    def start(self) -> None:
        """Reset elapsed time and (re-)start timer."""
        self._cpp_object.start()

    def stop(self) -> datetime.timedelta:
        """Stop timer and return elapsed time.

        Returns:
            Elapsed time.
        """
        return self._cpp_object.stop()

    def resume(self) -> None:
        """Resume timer."""
        self._cpp_object.resume()

    def elapsed(self) -> datetime.timedelta:
        """Return elapsed time.

        Returns:
            Elapsed time.
        """
        return self._cpp_object.elapsed()

    def flush(self) -> None:
        """Flush timer duration to the logger.

        Note:
            Timer must have been stopped before flushing.

            Timer can be flushed only once. Subsequent calls will have
            no effect.
        """
        self._cpp_object.flush()


def timed(task: str) -> Callable:
    """Decorator for timing functions."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            with Timer(task):
                return func(*args, **kwargs)

        return wrapper

    return decorator
