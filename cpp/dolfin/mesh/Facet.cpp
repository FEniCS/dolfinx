// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Facet.h"
#include "Cell.h"
#include "IntervalCell.h"
#include "TriangleCell.h"
#include <dolfin/geometry/utils.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
bool Facet::exterior() const
{
  const std::size_t D = _mesh->topology().dim();
  if (this->num_global_entities(D) == 1)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
