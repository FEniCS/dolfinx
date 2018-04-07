# -*- coding: utf-8 -*-
# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
from dolfin.jit.jit import ffc_jit
import dolfin.cpp


def create_coordinate_map(o):
    """Return a compiled UFC coordinate_mapping object"""

    try:
        # Create a compiled coordinate map from an object with the
        # ufl_mesh attribute
        cmap_ptr = ffc_jit(o.ufl_domain())
    except AttributeError:
        # FIXME: It would be good to avoid the type check, but ffc_jit
        # supports other objects so we could get, e.g., a compiled
        # finite element
        if isinstance(o, ufl.domain.Mesh):
            cmap_ptr = ffc_jit(o)
        else:
            raise TypeError(
                "Cannot create coordinate map from an object of type: {}".format(type(o)))
    except Exception:
        print("Failed to create compiled coordinate map")
        raise

    # Wrap compiled coordinate map and return
    ufc_cmap = dolfin.cpp.fem.make_ufc_coordinate_mapping(cmap_ptr)
    return dolfin.cpp.fem.CoordinateMapping(ufc_cmap)
