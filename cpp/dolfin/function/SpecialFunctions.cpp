// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SpecialFunctions.h"
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>

using namespace dolfin::function;

//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates(std::shared_ptr<const mesh::Mesh> mesh)
    : Expression({mesh->geometry().dim()}), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshCoordinates::eval(Eigen::Ref<EigenRowMatrixXd> values,
                           Eigen::Ref<const EigenRowMatrixXd> x,
                           const ufc::cell& cell) const
{
  dolfin_assert(_mesh);
  dolfin_assert(cell.geometric_dimension == _mesh->geometry().dim());
  dolfin_assert((unsigned int)x.cols() == _mesh->geometry().dim());

  values = x;
}
//-----------------------------------------------------------------------------
FacetArea::FacetArea(std::shared_ptr<const mesh::Mesh> mesh)
    : Expression({}), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(Eigen::Ref<EigenRowMatrixXd> values,
                     Eigen::Ref<const EigenRowMatrixXd> x,
                     const ufc::cell& cell) const
{
  dolfin_assert(_mesh);
  dolfin_assert(cell.geometric_dimension == _mesh->geometry().dim());

  for (std::size_t i = 0; i != x.rows(); ++i)
  {
    if (cell.local_facet >= 0)
    {
      mesh::Cell c(*_mesh, cell.index);
      values(i, 0) = c.facet_area(cell.local_facet);
    }
    else
    {
      // not_on_boundary
      values(i, 0) = 0.0;
    }
  }
}
//-----------------------------------------------------------------------------
