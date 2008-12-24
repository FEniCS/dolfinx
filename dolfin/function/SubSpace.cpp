// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-03
// Last changed: 2008-11-03

#include "SubSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, uint component)
  : FunctionSpace(V.mesh(), V.element(), V.dofmap())
{
  // Create array
  std::vector<uint> c;
  c.push_back(component);
  
  // Extract subspace and assign
  std::tr1::shared_ptr<FunctionSpace> _V(V.extract_sub_space(c));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, uint component, uint sub_component)
  : FunctionSpace(V.mesh(), V.element(), V.dofmap())
{
  // Create array
  std::vector<uint> c;
  c.push_back(component);
  c.push_back(sub_component);

  // Extract subspace and assign
  std::tr1::shared_ptr<FunctionSpace> _V(V.extract_sub_space(c));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
SubSpace::SubSpace(const FunctionSpace& V, const std::vector<uint>& component)
  : FunctionSpace(V.mesh(), V.element(), V.dofmap())
{
  // Extract subspace and assign
  std::tr1::shared_ptr<FunctionSpace> _V(V.extract_sub_space(component));
  *static_cast<FunctionSpace*>(this) = *_V;
}
//-----------------------------------------------------------------------------
