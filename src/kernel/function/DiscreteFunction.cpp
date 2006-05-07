// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Point.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/FEM.h>
#include <dolfin/Mesh.h>
#include <dolfin/Vector.h>
#include <dolfin/AffineMap.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/DiscreteFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x)
  : GenericFunction(), _x(&x), _mesh(0), _element(0),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Mesh and element need to be specified later or are automatically
  // chosen during assembly.
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x, Mesh& mesh)
  : GenericFunction(), _x(&x), _mesh(&mesh), _element(0),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Element needs to be specified later or are automatically
  // chosen during assembly.
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x, Mesh& mesh, FiniteElement& element)
  : GenericFunction(), _x(&x), _mesh(&mesh), _element(&element),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Update vector dimension from element
  updateVectorDimension();
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, FiniteElement& element)
  : GenericFunction(), _x(0), _mesh(&mesh), _element(&element),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Update vector dimension from element
  updateVectorDimension();

  // Allocate local storage
  uint size = FEM::size(mesh, element);
  _x = new Vector(size);
  vector_local = true;
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(const DiscreteFunction& f)
  : GenericFunction(), _x(0), _mesh(f._mesh), _element(f._element),
    _vectordim(f._vectordim), component(f.component),
    mixed_offset(f.mixed_offset), component_offset(f.component_offset),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Create a new vector and copy the values
  dolfin_assert(f._x);
  _x = new Vector();
  *_x =* f._x;
  vector_local = true;
}
//-----------------------------------------------------------------------------
DiscreteFunction::~DiscreteFunction()
{
  // Delete vector if local
  if ( vector_local )
    delete _x;

  // Delete mesh if local
  if ( mesh_local )
    delete _mesh;

  // Delete element if local
  if ( element_local )
    delete _element;
}
//-----------------------------------------------------------------------------
real DiscreteFunction::operator()(const Point& p, uint i)
{
  dolfin_error("Discrete functions cannot be evaluated at arbitrary points.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real DiscreteFunction::operator() (const Vertex& vertex, uint i)
{
  dolfin_assert(_x && _mesh && _element);

  // Initialize local data (if not already initialized correctly)
  local.init(*_element);

  // Get array of values (assumes uniprocessor case)
  real* xx = _x->array();

  // Evaluate all components at given vertex and pick given component
  _element->vertexeval(local.values, vertex.id(), xx + mixed_offset, *_mesh);

  // Restore array
  _x->restore(xx);

  return local.values[component + i];
}
//-----------------------------------------------------------------------------
void DiscreteFunction::sub(uint i)
{
  // Check that we have an element
  if ( !_element )
    dolfin_error("Unable to pick sub function or component of function since no element has been attached.");

  // Check that we have a mesh
  if ( !_mesh )
    dolfin_error("Unable to pick sub function or component of function since no mesh has been attached.");

  // Check if function is mixed
  if ( _element->elementdim() > 1 )
  {
    // Check the dimension
    if ( i >= _element->elementdim() )
      dolfin_error2("Illegal sub function index %d for mixed function with %d sub functions.",
		    i, _element->elementdim());

    // Compute offset for mixed sub function
    mixed_offset = 0;
    for (uint j = 0; j < i; j++)
      mixed_offset += FEM::size(*_mesh, (*_element)[j]);
    
    // Pick sub element and update vector dimension
    _element = &((*_element)[i]);
    updateVectorDimension();

    // Make sure component and component offset are zero
    component = 0;
    component_offset = 0;
  }
  else
  {
    // Check if function is vector-valued
    if ( _vectordim == 1 )
      dolfin_error("Cannot pick component of scalar function.");
    
    // Check the dimension
    if ( i >= _vectordim )
      dolfin_error2("Illegal component index %d for function with %d components.",
		    i, _vectordim);
    
    // Compute offset for component
    component_offset = FEM::size(*_mesh, *_element) / _vectordim;    

    // Save the component and make function scalar
    component = i;
    _vectordim = 1;
  }
}
//-----------------------------------------------------------------------------
void DiscreteFunction::copy(const DiscreteFunction& f)
{
  dolfin_assert(f._x);

  // Initialize local data if not already done
  if ( !vector_local )
  {
    _x = new Vector();
    vector_local = true;
  }
  
  // Copy values to vector
  *_x = *f._x;
  
  // Copy pointers to mesh and element
  _mesh = f._mesh;
  _element = f._element;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real coefficients[], AffineMap& map,
				   FiniteElement& element)
{
  // Save mesh and element (overwriting any previously attached values)
  _mesh = &map.cell().mesh();
  _element = &element;
  
  // Initialize local data (if not already initialized correctly)
  local.init(*_element);
  
  // Get array of values (assumes uniprocessor case)
  real* xx = _x->array();
  
  // Compute mapping to global degrees of freedom
  _element->nodemap(local.dofs, map.cell(), *_mesh);

  // Pick values
  for (uint i = 0; i < _element->spacedim(); i++)
    coefficients[i] = xx[mixed_offset + component_offset + local.dofs[i]];

  // Restore array
  _x->restore(xx);
}
//-----------------------------------------------------------------------------
dolfin::uint DiscreteFunction::vectordim() const
{
  return _vectordim;
}
//-----------------------------------------------------------------------------
Vector& DiscreteFunction::vector()
{
  dolfin_assert(_x);
  return *_x;
}
//-----------------------------------------------------------------------------
Mesh& DiscreteFunction::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& DiscreteFunction::element()
{
  dolfin_assert(_element);
  return *_element;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::attach(Vector& x, bool local)
{
  // Delete old vector if local
  if ( vector_local )
    delete _x;

  // Attach new vector
  _x = &x;
  vector_local = local;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::attach(Mesh& mesh, bool local)
{
  // Delete old mesh if local
  if ( mesh_local )
    delete _mesh;

  // Attach new mesh
  _mesh = &mesh;
  mesh_local = local;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::attach(FiniteElement& element, bool local)
{
  // Delete old mesh if local
  if ( element_local )
    delete _element;

  // Attach new mesh
  _element = &element;
  element_local = local;

  // Recompute vector dimension
  updateVectorDimension();
}
//-----------------------------------------------------------------------------
void DiscreteFunction::init(Mesh& mesh, FiniteElement& element)
{
  cout << "Reinitializing discrete function" << endl;

  // Reset data
  _mesh = &mesh;
  _element = &element;
  component = 0;
  mixed_offset = 0;
  component_offset = 0;
  
  // Update vector dimension from element
  updateVectorDimension();

  // Reinitialize local storage
  uint size = FEM::size(mesh, element);
  if ( !vector_local )
  {
    _x = new Vector(size);
    vector_local = true;
  }
  else
  {
    _x->init(size);
    *_x = 0.0;
  }
}
//-----------------------------------------------------------------------------
void DiscreteFunction::updateVectorDimension()
{
  dolfin_assert(_element);

  if ( _element->rank() == 0 )
  {
    _vectordim = 1;
  }
  else if ( _element->rank() == 1 )
  {
    _vectordim = _element->tensordim(0);
  }
  else
  {
    dolfin_error("Cannot handle tensor-valued functions.");
  }
}
//-----------------------------------------------------------------------------

#endif
