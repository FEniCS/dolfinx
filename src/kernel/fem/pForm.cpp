// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-10
// Last changed:

#include <dolfin/pForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
pForm::~pForm()
{
  if( local_dof_map_set )
    delete local_dof_map_set;
}
//-----------------------------------------------------------------------------
void pForm::updateDofMaps(Mesh& mesh)
{
  if( !dof_map_set )
  {
    // Create dof maps
    dof_map_set = new pDofMapSet(form(), mesh);

    // Take ownership of dof maps
    local_dof_map_set = dof_map_set;
  }
}
//-----------------------------------------------------------------------------
void pForm::setDofMaps(pDofMapSet& dof_map_set)
{
  // Delete dof map if locally owned 
  if( local_dof_map_set )
    delete local_dof_map_set;

  // Relinquish ownership of dof maps
  local_dof_map_set = false;

  this->dof_map_set = &dof_map_set;
}
//-----------------------------------------------------------------------------
pDofMapSet& pForm::dofMaps() const
{
  if( !dof_map_set )
    error("Degree of freedom maps for pForm have not been created.");

  return *dof_map_set;
}
//-----------------------------------------------------------------------------

