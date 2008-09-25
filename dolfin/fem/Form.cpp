// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-10
// Last changed: 2008-09-25

#include <ufc.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/MeshFunction.h>
#include "Form.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::~Form()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Form::updateDofMaps(Mesh& mesh)
{
  if( !dof_map_set )
  {
    std::tr1::shared_ptr<DofMapSet> _dof_map_set(new DofMapSet(form(), mesh));
    dof_map_set.swap(_dof_map_set);
  }
}
//-----------------------------------------------------------------------------
void Form::updateDofMaps(Mesh& mesh, MeshFunction<uint>& partitions)
{
  if( !dof_map_set )
  {
    // Create dof maps
    std::tr1::shared_ptr<DofMapSet> _dof_map_set(new DofMapSet(form(), mesh, partitions));
    dof_map_set.swap(_dof_map_set);
  }
}
//-----------------------------------------------------------------------------
void Form::setDofMaps(DofMapSet& dof_map_set)
{
  std::tr1::shared_ptr<DofMapSet> _dof_map_set(&dof_map_set, NoDeleter<DofMapSet>());
  this->dof_map_set.swap(_dof_map_set);
}
//-----------------------------------------------------------------------------
DofMapSet& Form::dofMaps() const
{
  if( !dof_map_set )
    error("Degree of freedom maps for Form have not been created.");

  return *dof_map_set;
}
//-----------------------------------------------------------------------------

