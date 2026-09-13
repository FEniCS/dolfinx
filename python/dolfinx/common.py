# Copyright (C) 2018-2026 Michal Habera, Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""General tools for timing and configuration."""

import datetime
import functools
import typing
from collections.abc import Callable

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx.cpp.common import (
    IndexMap,
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
    "list_timings",
    "local_range",
    "scatterer",
    "timed",
    "timing",
    "ufcx_signature",
]

Reduction = _cpp.common.Reduction

_ScatterArray: typing.TypeAlias = npt.NDArray[
    np.int64 | np.float32 | np.float64 | np.complex64 | np.complex128
]


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
            send_buffer,  # type: ignore[arg-type]
            recv_buffer,  # type: ignore[arg-type]
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
            send_buffer,  # type: ignore[arg-type]
            recv_buffer,  # type: ignore[arg-type]
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


def scatterer(index_map: IndexMap) -> Scatterer:
    """Create a scatterer for data with a layout described by an index map.

    Args:
        index_map: Index map that describes the parallel layout of
            the data.

    Returns:
        A new scatterer.
    """
    return Scatterer(_cpp.common.Scatterer(index_map))


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
