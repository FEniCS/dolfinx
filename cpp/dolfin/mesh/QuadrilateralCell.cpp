// Copyright (C) 2015 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "QuadrilateralCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <cmath>

using namespace dolfin;
using namespace dolfin::mesh;
