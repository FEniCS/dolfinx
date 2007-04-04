// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2007-04-02
//
// Note: this breaks the standard envelope-letter idiom slightly,
// since we call the envelope class from one of the letter classes.

#include <dolfin/Vertex.h>
#include <dolfin/Vector.h>
#include <dolfin/P1tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Function.h>
#include <dolfin/NonMatchingFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NonMatchingFunction::NonMatchingFunction(DiscreteFunction& F)
  : GenericFunction(),
    F(&F), _vectordim(F.vectordim()), component(0), _mesh(0),
    mesh_local(false)
{
}
//-----------------------------------------------------------------------------
NonMatchingFunction::NonMatchingFunction(const NonMatchingFunction& f)
  : GenericFunction(),
    F(f.F), _vectordim(f._vectordim), component(0), _mesh(f._mesh),
    mesh_local(false)
{
}
//-----------------------------------------------------------------------------
NonMatchingFunction::~NonMatchingFunction()
{
  // Delete mesh if local
  if ( mesh_local )
    delete _mesh;
}
//-----------------------------------------------------------------------------
real NonMatchingFunction::operator()(const Point& p, uint i)
{
//   cout << "Evaluating NonMatchingFunction at:" << endl;
//   cout << p << endl;

  // Evaluate discrete function
  return F->operator()(p, component + i);
}
//-----------------------------------------------------------------------------
real NonMatchingFunction::operator() (const Vertex& vertex, uint i)
{
  cout << "Evaluating NonMatchingFunction vertex at:" << endl;
  cout << vertex << endl;

  // Evaluate discrete function
  return F->operator()(vertex, component + i);
}
//-----------------------------------------------------------------------------
void NonMatchingFunction::sub(uint i)
{
  // Check if function is vector-valued
  if ( _vectordim == 1 )
    dolfin_error("Cannot pick component of scalar function.");

  // Check the dimension
  if ( i >= _vectordim )
    dolfin_error2("Illegal component index %d for function with %d components.",
		  i, _vectordim);

  // Save the component and make function scalar
  component = i;
  _vectordim = 1;
}
//-----------------------------------------------------------------------------
void NonMatchingFunction::interpolate(real coefficients[], Cell& cell,
                                      AffineMap& map, FiniteElement& element)
{
//   cout << "Interpolating NonMatchingFunction" << endl;

  // Initialize local data (if not already initialized correctly)
  local.init(element);
  
  // Map interpolation points to current cell
  element.pointmap(local.points, local.components, map);

  // FIXME: We can exploit knowledge of the cell to speed this up
  // Evaluate function at interpolation points
  for (uint i = 0; i < element.spacedim(); i++)
  {
    //cout << "point:" << endl;
//     cout << local.points[i] << endl;
    coefficients[i] = F->operator()(local.points[i], component + local.components[i]);
//     cout << "coeff: " << coefficients[i] << endl;
  }
}
//-----------------------------------------------------------------------------
dolfin::uint NonMatchingFunction::vectordim() const
{
  /// Return vector dimension of function
  return _vectordim;
}
//-----------------------------------------------------------------------------
Vector& NonMatchingFunction::vector()
{
  dolfin_error("No vector associated with function (and none can be attached).");
  return *(new Vector()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
Mesh& NonMatchingFunction::mesh()
{
  if ( !_mesh )
    dolfin_error("No mesh associated with function (try attaching one).");
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& NonMatchingFunction::element()
{
  dolfin_error("No finite element associated with function (an none can be attached).");
  return *(new P1tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
void NonMatchingFunction::attach(Vector& x, bool local)
{
  dolfin_error("Cannot attach vectors to non-matching functions.");
}
//-----------------------------------------------------------------------------
void NonMatchingFunction::attach(Mesh& mesh, bool local)
{
  // Delete old mesh if local
  if ( mesh_local )
    delete _mesh;

  // Attach new mesh
  _mesh = &mesh;
  mesh_local = local;
}
//-----------------------------------------------------------------------------
void NonMatchingFunction::attach(FiniteElement& element, bool local)
{
  dolfin_error("Cannot attach finite elements to non-matching functions.");
}
//-----------------------------------------------------------------------------
void NonMatchingFunction::interpolate(real* coefficients,
                                      const ufc::cell& cell,
                                      const ufc::finite_element& finite_element)
{
  dolfin_error("Not implemented");
}
//-----------------------------------------------------------------------------
