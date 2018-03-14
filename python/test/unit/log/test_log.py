"""Unit tests for the log"""

# Copyright (C) 2017 Tormod Landet
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin

def test_log_level_comparable():
    info = dolfin.LogLevel.INFO
    warning = dolfin.LogLevel.WARNING
    assert info < warning
    assert warning < 1000
