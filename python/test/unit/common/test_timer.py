"""Unit tests for timing facilities"""

# Copyright (C) 2017 Jan Blechta, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from time import sleep

from dolfinx import common


def test_context_manager_named():
    """Test that named Timer works as context manager"""
    task = "test_context_manager_named_str"

    # Execute task in the context manager
    t = common.Timer(task)
    sleep(0.05)
    assert t.elapsed()[0] > 0.035
    del t

    # Check timing
    t = common.timing(task)
    assert t[0] == 1
    assert t[1] > 0.035


def test_context_manager_anonymous():
    """Test that anonymous Timer works as context manager"""
    with common.Timer() as t:
        sleep(0.05)
        assert t.elapsed()[0] > 0.035
