// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-10
// Last changed:

#include <dolfin/Form.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::~Form()
{
  if( local_dof_map_set )
    delete local_dof_map_set;
}
//-----------------------------------------------------------------------------
void Form::updateDofMaps(Mesh& mesh)
{
  if( !dof_map_set )
  {
    // Create dof maps
    dof_map_set = new DofMapSet(form(), mesh);

    // Take ownership of dof maps
    local_dof_map_set = dof_map_set;
  }
}
//-----------------------------------------------------------------------------
void Form::updateDofMaps(Mesh& mesh, MeshFunction<uint>& partitions)
{
  if( !dof_map_set )
  {
    // Create dof maps
    dof_map_set = new DofMapSet(form(), mesh, partitions);

    // Take ownership of dof maps
    local_dof_map_set = dof_map_set;
  }
}
//-----------------------------------------------------------------------------
void Form::setDofMaps(DofMapSet& dof_map_set)
{
  // Delete dof map if locally owned 
  if( local_dof_map_set )
    delete local_dof_map_set;

  // Relinquish ownership of dof maps
  local_dof_map_set = false;

  this->dof_map_set = &dof_map_set;
}
//-----------------------------------------------------------------------------
DofMapSet& Form::dofMaps() const
{
  if( !dof_map_set )
    error("Degree of freedom maps for Form have not been created.");

  return *dof_map_set;
}
//-----------------------------------------------------------------------------

