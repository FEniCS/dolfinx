// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-10
// Last changed: 2008-09-25

#include <ufc.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/MeshFunction.h>
#include "Form.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::~Form()
{
  std::vector<FiniteElement*>::iterator element;
  for(element = finite_elements.begin(); element != finite_elements.end(); ++element)
    cout << (*element)->value_rank() << endl;
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
void Form::updateFiniteElements()
{
  // Resize Array to hold pointers to finite elements
  const uint num_arguments = form().rank() + form().num_coefficients();
  if( finite_elements.size() != num_arguments)
    finite_elements.resize(num_arguments);

  // Create finite elements
  for(uint i = 0; i < form().rank() + form().num_coefficients(); ++i)
    finite_elements[i] = new FiniteElement(form().create_finite_element(i));
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
FiniteElement& Form::finite_element(uint i)
{
  const uint num_arguments = form().rank() + form().num_coefficients();

  // Check dimensions
  if (i >= num_arguments)
    error("Illegal function index %d. Form only has %d arguments.", i, num_arguments);

  // Create finite elements if needed
  if( finite_elements.size() < num_arguments )
    updateFiniteElements();

  return *(finite_elements[i]);
}
//-----------------------------------------------------------------------------
void Form::check() const
{
  error("Form::check() not implemented.");
}
//-----------------------------------------------------------------------------
