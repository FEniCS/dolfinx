"""Unit tests for timing facilities"""

# Copyright (C) 2017 Jan Blechta, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from time import sleep

import pytest

from dolfinx import common


def test_timer():
    """Test that named Timer works."""
    dt = 0.05
    task = "test_named_str"
    t = common.Timer(task)
    t.start()
    sleep(dt)
    t.stop()
    assert t.elapsed().total_seconds() > 0.9 * dt

    t.resume()
    sleep(dt)
    t.stop()
    assert t.elapsed().total_seconds() > 2 * 0.9 * dt

    t.flush()
    t = common.timing(task)
    assert t[0] == 1
    assert t[1] > 0.045


def xtest_context_manager_named():
    """Test that named Timer works as context manager."""
    task = "test_context_manager_named_str"
    with common.Timer(task):
        sleep(0.05)
    delta = common.timing(task)
    assert delta[1] > 0.045


def xtest_context_manager_anonymous():
    """Test that anonymous Timer works with context manager."""
    timer = common.Timer()
    with timer:
        sleep(0.05)

    assert timer.elapsed().total_seconds() > 0.045


if __name__ == "__main__":
    pytest.main([__file__])
