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
    "ScatterHandle",
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


class ScatterHandle:
    """Handle for a non-blocking Scatterer exchange.

    Returned by :meth:`Scatterer.scatter_fwd_begin`/
    :meth:`Scatterer.scatter_rev_begin` and passed to the matching
    ``scatter_fwd_end``/``scatter_rev_end`` call to complete the
    exchange. Treat as opaque; do not modify the arrays passed to the
    ``*_begin`` call until the matching ``*_end`` call returns.
    """

    def __init__(
        self,
        request: _MPI.Request,
        buffer: _ScatterArray,
        data: _ScatterArray,
        idx: npt.NDArray[np.int32],
        bs: int,
    ):
        """Create a scatter handle.

        Note:
            This initialiser is intended for internal library use
            only.
        """
        self._request = request
        self._buffer = buffer
        self._data = data
        self._idx = idx
        self._bs = bs


class Scatterer:
    """Scatter and gather data with a layout described by an ``IndexMap``.

    A scatterer is stateless: it holds only the communication pattern
    derived from an :class:`IndexMap`, and can be shared between
    multiple objects (e.g. :class:`dolfinx.la.Vector`) that use the
    same index map.
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

    def scatter_fwd_begin(
        self, local_data: _ScatterArray, remote_data: _ScatterArray, bs: int = 1
    ) -> ScatterHandle:
        """Start scattering owned data to processes that ghost it.

        Complete the exchange by passing the returned handle to
        :meth:`scatter_fwd_end`. Unrelated work can be done between
        the two calls to overlap communication with computation, but
        ``local_data``/``remote_data`` must not be modified until
        :meth:`scatter_fwd_end` returns.

        Args:
            local_data: Array holding the owned data, blocked by
                ``bs``. Must be at least as long as the number of
                owned entries that are ghosted elsewhere.
            remote_data: Array that :meth:`scatter_fwd_end` will fill
                with the ghost values received from owning processes,
                blocked by ``bs``.
            bs: Number of values associated with each index map index.

        Returns:
            Handle to pass to :meth:`scatter_fwd_end`.
        """
        local_idx = self._cpp_object.local_indices_block
        remote_idx = self._cpp_object.remote_indices_block

        local_buffer = local_data.reshape(-1, bs)[local_idx].reshape(-1)
        remote_buffer = np.empty(bs * remote_idx.size, dtype=local_data.dtype)

        request = self._cpp_object.scatter_fwd_begin(
            local_buffer,  # type: ignore[arg-type]
            remote_buffer,  # type: ignore[arg-type]
            bs,
        )
        return ScatterHandle(request, remote_buffer, remote_data, remote_idx, bs)

    def scatter_fwd_end(self, handle: ScatterHandle) -> None:
        """Complete a forward scatter started by :meth:`scatter_fwd_begin`.

        Args:
            handle: Handle returned by :meth:`scatter_fwd_begin`.
        """
        self._cpp_object.scatter_fwd_end(handle._request)
        handle._data.reshape(-1, handle._bs)[handle._idx] = handle._buffer.reshape(-1, handle._bs)

    def scatter_fwd(
        self, local_data: _ScatterArray, remote_data: _ScatterArray, bs: int = 1
    ) -> None:
        """Scatter owned data to processes that ghost it.

        Args:
            local_data: Array holding the owned data, blocked by
                ``bs``. Must be at least as long as the number of
                owned entries that are ghosted elsewhere.
            remote_data: Array to fill with the ghost values received
                from owning processes, blocked by ``bs``.
            bs: Number of values associated with each index map index.
        """
        self.scatter_fwd_end(self.scatter_fwd_begin(local_data, remote_data, bs))

    def scatter_rev_begin(
        self, local_data: _ScatterArray, remote_data: _ScatterArray, bs: int = 1
    ) -> ScatterHandle:
        """Start scattering ghost data to owning processes.

        Complete the exchange by passing the returned handle to
        :meth:`scatter_rev_end`. Unrelated work can be done between
        the two calls to overlap communication with computation, but
        ``local_data``/``remote_data`` must not be modified until
        :meth:`scatter_rev_end` returns.

        Args:
            local_data: Array holding the owned data, blocked by
                ``bs``. :meth:`scatter_rev_end` accumulates values
                received from ``remote_data`` into this array.
            remote_data: Array holding the ghost values to send to
                owning processes, blocked by ``bs``.
            bs: Number of values associated with each index map index.

        Returns:
            Handle to pass to :meth:`scatter_rev_end`.
        """
        local_idx = self._cpp_object.local_indices_block
        remote_idx = self._cpp_object.remote_indices_block

        remote_buffer = remote_data.reshape(-1, bs)[remote_idx].reshape(-1)
        local_buffer = np.empty(bs * local_idx.size, dtype=local_data.dtype)

        request = self._cpp_object.scatter_rev_begin(
            remote_buffer,  # type: ignore[arg-type]
            local_buffer,  # type: ignore[arg-type]
            bs,
        )
        return ScatterHandle(request, local_buffer, local_data, local_idx, bs)

    def scatter_rev_end(self, handle: ScatterHandle) -> None:
        """Complete a reverse scatter started by :meth:`scatter_rev_begin`.

        Args:
            handle: Handle returned by :meth:`scatter_rev_begin`.
        """
        self._cpp_object.scatter_rev_end(handle._request)

        # handle._idx may repeat (an owned entry can be ghosted by more
        # than one rank), so plain `data[idx] += ...` would silently
        # drop all but one contribution per repeated index; np.add.at
        # accumulates unbuffered, handling repeats correctly.
        np.add.at(
            handle._data.reshape(-1, handle._bs),
            handle._idx,
            handle._buffer.reshape(-1, handle._bs),
        )

    def scatter_rev(
        self, local_data: _ScatterArray, remote_data: _ScatterArray, bs: int = 1
    ) -> None:
        """Scatter ghost data to owning processes, accumulating.

        Args:
            local_data: Array holding the owned data, blocked by
                ``bs``. Updated in-place with values accumulated from
                ``remote_data``.
            remote_data: Array holding the ghost values to send to
                owning processes, blocked by ``bs``.
            bs: Number of values associated with each index map index.
        """
        self.scatter_rev_end(self.scatter_rev_begin(local_data, remote_data, bs))


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
