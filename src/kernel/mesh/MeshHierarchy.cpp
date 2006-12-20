// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-12-20
// Last changed: 2006-12-20

#include <dolfin/MeshHierarchy.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshHierarchy::MeshHierarchy() 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshHierarchy::~MeshHierarchy()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh MeshHierarchy::operator[](uint k) const 
{
  dolfin_error("not implemented.");
}
//-----------------------------------------------------------------------------
Mesh MeshHierarchy::operator()(uint k) const 
{
  dolfin_error("not implemented.");
}
//-----------------------------------------------------------------------------

