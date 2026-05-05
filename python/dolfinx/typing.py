# Copyright (C) 2026 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Common typing functionality."""

from typing import TypeVar

import numpy as np

__all__ = ["Index", "Real", "Scalar"]

Index = TypeVar("Index", np.int32, np.int64)

Real = TypeVar("Real", np.float32, np.float64)
Scalar = TypeVar("Scalar", np.float32, np.float64, np.complex64, np.complex128)
