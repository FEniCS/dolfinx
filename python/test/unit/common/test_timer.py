"""Unit tests for timing facilities"""

# Copyright (C) 2017 Jan Blechta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

import gc
import random
from time import sleep

from dolfin import timing, TimingClear
from dolfin.timer import Timer


# Seed random generator for determinism
random.seed(0)

def get_random_task_name():
    """Get pseudo-random string"""
    return hex(random.randint(0, 1e32))


def test_context_manager_named():
    """Test that named Timer works as context manager"""
    task = get_random_task_name()

    # Execute task in the context manager
    t = Timer(task)
    sleep(0.05)
    assert t.elapsed()[0] >= 0.05
    del t

    # Check timing
    t = timing(task, TimingClear.clear)
    assert t[0] == 1
    assert t[1] >= 0.05

def test_context_manager_anonymous():
    """Test that anonymous Timer works as context manager"""
    with Timer() as t:
        sleep(0.05)
        assert t.elapsed()[0] >= 0.05
