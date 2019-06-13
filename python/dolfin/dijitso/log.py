# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Martin Sandve Alnæs, Jan Blechta
#
# This file is part of DIJITSO.
#
# DIJITSO is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DIJITSO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DIJITSO. If not, see <http://www.gnu.org/licenses/>.

import logging

__all__ = ['set_log_level', 'get_logger', 'get_log_handler', 'set_log_handler']


_log = logging.getLogger("dijitso")
_loghandler = logging.StreamHandler()
_log.addHandler(_loghandler)
_log.setLevel(logging.INFO)


def get_log_handler():
    return _loghandler


def get_logger():
    return _log


def set_log_handler(handler):
    global _loghandler
    _log.removeHandler(_loghandler)
    _loghandler = handler
    _log.addHandler(_loghandler)


def set_log_level(level):
    """Set verbosity of logging. Argument is int or one of "INFO", "WARNING",
    "ERROR", or "DEBUG".
    """
    if isinstance(level, str):
        level = level.upper()
        assert level in ("INFO", "WARNING", "ERROR", "DEBUG")
        level = getattr(logging, level)
    else:
        assert isinstance(level, int)
    _log.setLevel(level)


# Logging interface for dijitso library

def debug(*message):
    _log.debug(*message)


def info(*message):
    _log.info(*message)


def warning(*message):
    _log.warning(*message)


def error(*message):
    _log.error(*message)
    text = message[0] % message[1:]
    raise RuntimeError(text)


def dijitso_assert(condition, *message):
    if not condition:
        _log.error(*message)
        text = message[0] % message[1:]
        raise AssertionError(text)
