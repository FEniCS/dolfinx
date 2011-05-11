// Copyright (C) 2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2008-11-03
// Last changed: 2009-05-17

#include <dolfin/common/NoDeleter.h>
#include "SubSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, uint component)
  : FunctionSpace(reference_to_no_delete_pointer(V.mesh()), 
                  reference_to_no_delete_pointer(V.element()), 
                  reference_to_no_delete_pointer(V.dofmap()))
{
  // Create array
  std::vector<uint> c;
  c.push_back(component);

  // Extract subspace and assign
  boost::shared_ptr<FunctionSpace> _V(V.extract_sub_space(c));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, uint component, uint sub_component)
  : FunctionSpace(reference_to_no_delete_pointer(V.mesh()), 
                  reference_to_no_delete_pointer(V.element()), 
                  reference_to_no_delete_pointer(V.dofmap()))
{
  // Create array
  std::vector<uint> c;
  c.push_back(component);
  c.push_back(sub_component);

  // Extract subspace and assign
  boost::shared_ptr<FunctionSpace> _V(V.extract_sub_space(c));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, const std::vector<uint>& component)
  : FunctionSpace(reference_to_no_delete_pointer(V.mesh()), 
                  reference_to_no_delete_pointer(V.element()), 
                  reference_to_no_delete_pointer(V.dofmap()))
{
  // Extract subspace and assign
  boost::shared_ptr<FunctionSpace> _V(V.extract_sub_space(component));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
