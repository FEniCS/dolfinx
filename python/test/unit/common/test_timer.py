"""Unit tests for timing facilities"""

# Copyright (C) 2017 Jan Blechta, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from time import sleep

import pytest

from dolfinx import common


def test_context_manager_named():
    """Test that named Timer works as context manager"""
    task = "test_context_manager_named_str"

    # Execute task in the context manager
    t = common.Timer(task)
    t.start()
    sleep(0.05)
    t.stop()
    assert t.elapsed().total_seconds() > 0.035
    del t

    # Check timing
    t = common.timing(task)
    assert t[0] == 1
    assert t[1] > 0.035


def test_context_manager_anonymous():
    """Test that anonymous Timer works as context manager"""
    timer = common.Timer()
    with timer:
        sleep(0.05)

    assert timer.elapsed().total_seconds() > 0.035


if __name__ == "__main__":
    pytest.main([__file__])
