// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU GPL Version 2.

// First added:  2007-03-01
// Last changed: 2007-03-01

#include <dolfin/DofMap.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(const ufc::dof_map& dof_map, Mesh& mesh) : dof_map(dof_map)
{
  

}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{

}
//-----------------------------------------------------------------------------

