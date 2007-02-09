// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-11-26
// Last changed: 2007-01-29

#include <dolfin/dolfin_log.h>
#include <dolfin/Point.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/FEM.h>
#include <dolfin/Mesh.h>
#include <dolfin/Vector.h>
#include <dolfin/AffineMap.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Function.h>
#include <dolfin/DiscreteFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x)
  : GenericFunction(), _x(&x), _mesh(0), _element(0), _idetector(0),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Mesh and element need to be specified later or are automatically
  // chosen during assembly.
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x, Mesh& mesh)
  : GenericFunction(), _x(&x), _mesh(&mesh), _element(0), _idetector(0),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Element needs to be specified later or are automatically
  // chosen during assembly.
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Vector& x, Mesh& mesh, FiniteElement& element)
  : GenericFunction(), _x(&x), _mesh(&mesh), _element(&element), _idetector(0),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Update vector dimension from element
  updateVectorDimension();
  //constructBasis();
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, FiniteElement& element)
  : GenericFunction(), _x(0), _mesh(&mesh), _element(&element), _idetector(0),
    _vectordim(1), component(0), mixed_offset(0), component_offset(0),
    vector_local(false), mesh_local(false), element_local(false)
{
  // Update vector dimension from element
  updateVectorDimension();
  //constructBasis();

  // Allocate local storage
  uint size = FEM::size(mesh, element);
  _x = new Vector(size);
  vector_local = true;
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(const DiscreteFunction& f)
  : GenericFunction(), _x(0), _mesh(f._mesh), _element(f._element),
    _idetector(0), _vectordim(f._vectordim), component(f.component),
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
//   cout << "DiscreteFunction evaluation" << endl;

  // Note, this is a very expensive operation compared to evaluating at
  // vertices

  if(!_idetector)
  {
    constructBasis();
  }

  if(have_basis)
  {
    dolfin_assert(_x && _mesh && _element);

    // FIXME: create const versions of IntersectionDetector function
    Point pprobe = p;

    // Find cell(s) intersecting p
    Array<uint> cells;
    _idetector->overlap(pprobe, cells);

    // If there are more than one cell, compute the average value of
    // the cells

//     if(cells.size() > 1)
//       cout << "cells.size(): " << cells.size() << endl;

    real sum = 0.0;
    for(uint i = 0; i < cells.size(); i++)
    {
      Cell cell(*_mesh, cells[i]);
      
      // Initialize local data (if not already initialized correctly)
      local.init(*_element);
      
      // Compute mapping to global degrees of freedom
      _element->nodemap(local.dofs, cell, *_mesh);
      
      // Compute positions in global vector by adding offsets
      for (uint j = 0; j < _element->spacedim(); j++)
	local.dofs[j] = local.dofs[j] + mixed_offset + component_offset;
      
      // Get values
      _x->get(local.coefficients, local.dofs, _element->spacedim());
      
      // Compute map
      NewAffineMap map;
      map.update(cell);

      // Compute finite element sum
      real cellsum = 0.0;
      for(uint j = 0; j < _element->spacedim(); j++)
      {
	cellsum += local.coefficients[j] *
	  basis.evalPhysical(*(basis.functions[j]), pprobe, map, i);
      }
      sum += cellsum;
    }

    sum /= cells.size();

    return sum;
  }
  else
  {
    dolfin_error("Discrete functions cannot be evaluated at arbitrary points.");
    return 0.0;
  }
}
//-----------------------------------------------------------------------------
real DiscreteFunction::operator() (const Vertex& vertex, uint i)
{
  dolfin_assert(_x && _mesh && _element);

  // This is a special hack for Lagrange elements, need to compute
  // the L2 projection in general

  // Initialize local data (if not already initialized correctly)
  local.init(*_element);

  // Get vertex nodes for all components
  _element->vertexeval(local.vertex_nodes, vertex.index(), *_mesh);

  // Pick value
  return (*_x)(mixed_offset + local.vertex_nodes[component + i]);
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
    //constructBasis();


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
void DiscreteFunction::interpolate(real coefficients[], Cell& cell,
                                   AffineMap& map, FiniteElement& element)
{
  // Save mesh and element (overwriting any previously attached values)
  _mesh = &cell.mesh();
  _element = &element;
  
  // Initialize local data (if not already initialized correctly)
  local.init(*_element);
  
  // Compute mapping to global degrees of freedom
  _element->nodemap(local.dofs, cell, *_mesh);

  // Compute positions in global vector by adding offsets
  for (uint i = 0; i < _element->spacedim(); i++)
    local.dofs[i] = local.dofs[i] + mixed_offset + component_offset;

  // Get values
  _x->get(coefficients, local.dofs, _element->spacedim());
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(Function& fsource)
{
  FiniteElement& e = element();
  Vector& x = vector();
  Mesh& m = mesh();

  AffineMap map;

  int *nodes = new int[e.spacedim()];
  real *coefficients = new real[e.spacedim()];

  for(CellIterator c(m); !c.end(); ++c)
  {
    Cell& cell = *c;

    // Use DOLFIN's interpolation

    map.update(cell);
    fsource.interpolate(coefficients, cell, map, e);
    e.nodemap(nodes, cell, m);

    for(unsigned int i = 0; i < e.spacedim(); i++)
    {
      x(nodes[i]) = coefficients[i];
    }
  }

  delete [] nodes;
  delete [] coefficients;
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
  // Delete old element if local
  if ( element_local )
    delete _element;

  // Attach new element
  _element = &element;
  element_local = local;

  // Recompute vector dimension
  updateVectorDimension();
  //constructBasis();
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
  //constructBasis();


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
void DiscreteFunction::constructBasis()
{
  dolfin_assert(_element);
  dolfin_assert(_mesh);

  have_basis = basis.construct(*_element);
  //idetector.init(*_mesh);
  _idetector = new IntersectionDetector();
  _idetector->init(*_mesh);
}
//-----------------------------------------------------------------------------
