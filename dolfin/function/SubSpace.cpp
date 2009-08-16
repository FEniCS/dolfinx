// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
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
  boost::shared_ptr<FunctionSpace> _V(V.extract_sub_space(c, true));
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
  boost::shared_ptr<FunctionSpace> _V(V.extract_sub_space(c, true));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, const std::vector<uint>& component)
  : FunctionSpace(reference_to_no_delete_pointer(V.mesh()), 
                  reference_to_no_delete_pointer(V.element()), 
                  reference_to_no_delete_pointer(V.dofmap()))
{
  // Extract subspace and assign
  boost::shared_ptr<FunctionSpace> _V(V.extract_sub_space(component, true));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
