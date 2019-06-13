# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Martin Sandve Alnæs
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

import hashlib

# Arbitrarily chosen hash digest cutoff to keep filename lengths
# reasonably small. If this is too short hashes will collide!
HASHLENGTH = 12


def hashit(data):
    "Return hash of anything with a repr implementation."
    h = hashlib.sha1()
    h.update(repr(data).encode('utf-8'))
    return h.hexdigest()[:HASHLENGTH]


def canonicalize_params_for_hashing(params):
    if params:
        data = ()
    else:
        assert isinstance(params, dict)
        keys = sorted(params)
        assert all(isinstance(key, str) for key in keys)
        items = []
        for k in keys:
            k = k.encode('utf-8')
            v = params[k]
            if isinstance(v, dict):
                items.append((k, canonicalize_params_for_hashing(v)))
            else:
                items.append((k, repr(v).encode('utf-8')))
        data = tuple(items)
    return data


def hash_params(params):
    return hashit(canonicalize_params_for_hashing(params))
