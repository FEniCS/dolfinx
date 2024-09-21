"""Unit tests for timing facilities"""

# Copyright (C) 2017 Jan Blechta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import random
from time import sleep

from dolfinx import common

# Seed random generator for determinism
random.seed(0)

# resolution of the Timer is not precise,
# need to allow some tolerance
timing_resolution = 0.015

def get_random_task_name():
    """Get pseudo-random string"""
    return hex(random.randint(0, 1e32))


def test_context_manager_named():
    """Test that named Timer works as context manager"""
    task = get_random_task_name()

    # Execute task in the context manager
    t = common.Timer(task)
    sleep(0.1)
    assert t.elapsed()[0] >= 0.1 - timing_resolution
    del t

    # Check timing
    t = common.timing(task)
    assert t[0] == 1
    assert t[1] >= 0.05


def test_context_manager_anonymous():
    """Test that anonymous Timer works as context manager"""
    with common.Timer() as t:
        sleep(0.1)
        assert t.elapsed()[0] >= 0.1 - timing_resolution
