"""Unit tests for the log"""

# Copyright (C) 2017 Tormod Landet
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import log


def test_log():
    info = log.LogLevel.INFO
    warn = log.LogLevel.WARNING
    error = log.LogLevel.ERROR
    print(info, warn, error)

    log.set_log_level(warn)
    log.log(info, "HELLO")
    log.log(warn, "HELLO")
    log.log(error, "HELLO")

    log.set_log_level(info)
    log.log(info, "HELLO")
    log.log(warn, "HELLO")
    log.log(error, "HELLO")
