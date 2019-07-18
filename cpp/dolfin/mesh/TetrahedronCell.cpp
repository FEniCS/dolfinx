// Copyright (C) 2006-2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TetrahedronCell.h"
#include "Cell.h"
#include "Facet.h"
#include "Geometry.h"
#include "TriangleCell.h"
#include "Vertex.h"
#include <algorithm>
#include <cmath>
#include <dolfin/geometry/utils.h>

using namespace dolfin;
using namespace dolfin::mesh;

