// Copyright (C) 2012 Joachim B Haga
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-09-04
// Last changed: 2012-09-06

#ifdef HAS_VTK

#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/generation/CSGGeometry.h>
#include "VTKPlottableCSGGeometry.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableCSGGeometry::VTKPlottableCSGGeometry(boost::shared_ptr<const CSGGeometry> geometry) :
  VTKPlottableMesh(boost::shared_ptr<const Mesh>(new BoundaryMesh(geometry))),
  _geometry(geometry)
{
  // Do nothing
}
//----------------------------------------------------------------------------
bool VTKPlottableCSGGeometry::is_compatible(const Variable &var) const
{
  return dynamic_cast<const CSGGeometry*>(&var);
}
//----------------------------------------------------------------------------
void VTKPlottableCSGGeometry::update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int framecounter)
{
  boost::shared_ptr<const Mesh> mesh;
  if (var)
  {
    _geometry = boost::dynamic_pointer_cast<const CSGGeometry>(var);
    mesh.reset(new BoundaryMesh(_geometry));
  }
  dolfin_assert(_geometry);

  VTKPlottableMesh::update(mesh, parameters, framecounter);
}
//----------------------------------------------------------------------------
VTKPlottableCSGGeometry *dolfin::CreateVTKPlottable(boost::shared_ptr<const CSGGeometry> geometry)
{
  return new VTKPlottableCSGGeometry(geometry);
}
//----------------------------------------------------------------------------

#endif
