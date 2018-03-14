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

from dolfin import *


# Seed random generator for determinism
random.seed(0)

def get_random_task_name():
    """Get pseudo-random string"""
    return hex(random.randint(0, 1e32))


def test_context_manager_named():
    """Test that named Timer works as context manager"""
    task = get_random_task_name()

    # Execute task in the context manager
    with Timer(task) as t:
        sleep(0.05)
        assert t.elapsed()[0] >= 0.05

    # Check timing
    t = timing(task, TimingClear.clear)
    assert t[0] == 1
    assert t[1] >= 0.05


def test_context_manager_anonymous():
    """Test that anonymous Timer works as context manager"""
    with Timer() as t:
        sleep(0.05)
        assert t.elapsed()[0] >= 0.05


# Test case for decorated free function
fun2_task = get_random_task_name()
@timed(fun2_task)
def fun2(*args, **kwargs):
    "Foo"
    sleep(0.05)
    return args, kwargs

class C(object):
    # Test case for decorated instancemethod
    task_method2 = get_random_task_name()
    @timed(task_method2)
    def method2(self, *args, **kwargs):
        "Foo"
        sleep(0.05)
        return args, kwargs

    # Test case for decorated staticmethod
    task_method3 = get_random_task_name()
    @staticmethod
    @timed(task_method3)
    def method3(*args, **kwargs):
        "Foo"
        sleep(0.05)
        return args, kwargs

    # Test case for decorated classmethod
    task_method5 = get_random_task_name()
    @classmethod
    @timed(task_method5)
    def method5(cls, *args, **kwargs):
        "Foo"
        sleep(0.05)
        return args, kwargs

# Testing object with decorated methods
o = C()


@pytest.mark.parametrize(("fun", "task"), [
    (fun2, fun2_task),
    (o.method2, o.task_method2),
    (o.method3, o.task_method3),
    (o.method5, o.task_method5),
    (C.method3, C.task_method3),
    (C.method5, C.task_method5),
])
def test_decorator_functionality(fun, task):
    """Test functionality of timed decorator on function,
    instancemethod, classmethod, staticmethod..."""
    # Check that decorator preserves docstring
    assert fun.__doc__ == "Foo"

    # Check that function call works and returns what expected
    assert fun(1, 2, 3, four=5) == ((1, 2, 3), {'four': 5})
    assert fun(1, 2, 3, four=5) == ((1, 2, 3), {'four': 5})

    # Check that correct timing was recorded
    t = timing(task, TimingClear.clear)
    assert t[0] == 2
    assert t[1] >= 0.1


def test_decorator_timer_scope():
    """Check that underlying Timer object is not launched
    by decorator unless function is invoked"""
    task = get_random_task_name()

    # Delete eventual previous timing
    try:
        timing(task, TimingClear.clear)
    except RuntimeError:
        pass

    @timed(task)
    def foo():
        pass

    # Delete function (including eventual timer which should not
    # exist yet) before using it
    del foo
    gc.collect()

    # Check that there's no timing entry
    with pytest.raises(RuntimeError):
        timing(task, TimingClear.clear)


def test_decorator_timing_correctness():
    """Check that timing is measured really when the function
    is executed, not during existence of decorator"""
    task = get_random_task_name()

    @timed(task)
    def foo():
        pass

    sleep(0.05)
    foo()
    sleep(0.05)

    # Check that sleeping above did not influence timer
    assert timing(task, TimingClear.clear)[1] < 0.1
