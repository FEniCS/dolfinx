#include <dolfin/Quadrature.h>
#include <dolfin/Map.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FunctionList.h>
#include <dolfin/Integral.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
// Integral::Measure
//-----------------------------------------------------------------------------
Integral::Measure::Measure()
{
  // Map and quadrature will be initialised later
  m = 0;
  q = 0;
}
//-----------------------------------------------------------------------------
Integral::Measure::Measure(const Map& map,
			   const Quadrature& quadrature)
{  
  // Save map and quadrature
  m = &map;
  q = &quadrature;
}
//-----------------------------------------------------------------------------
Integral::Measure::~Measure()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Integral::Measure::update(const Map& map,
			       const Quadrature& quadrature)
{
  m = &map;
  q = &quadrature;
}
//-----------------------------------------------------------------------------
real Integral::Measure::operator* (real a) const
{
  // Return zero if the measure is not active
  if ( !active )
    return 0.0;
  
  return a * q->measure() * fabs(det());
}
//-----------------------------------------------------------------------------
real Integral::Measure::operator* (const FunctionSpace::ElementFunction& v)
{
  // Return zero if the measure is not active
  if ( !active )
    return 0.0;

  return v * (*this);
}
//-----------------------------------------------------------------------------
// Integral::InteriorMeasure
//-----------------------------------------------------------------------------
Integral::InteriorMeasure::InteriorMeasure() : Measure()
{
  init();
}
//-----------------------------------------------------------------------------
Integral::InteriorMeasure::InteriorMeasure(Map& m, Quadrature& q)
  : Measure(m, q)
{
  init();
}
//-----------------------------------------------------------------------------
Integral::InteriorMeasure::~InteriorMeasure()
{
  if ( table )
    delete [] table;
  table = 0;
}
//-----------------------------------------------------------------------------
void Integral::InteriorMeasure::update(const Map& map,
				       const Quadrature& quadrature)
{
  // Common update for measures
  Measure::update(map, quadrature);

  // Measure is only active on the interior of the domain
  if ( map.boundary() == -1 )
    active = true;
  else
    active = false;
}
//-----------------------------------------------------------------------------
real Integral::InteriorMeasure::operator* (const FunctionSpace::ShapeFunction& v)
{
  // Return zero if the measure is not active
  if ( !active )
    return 0.0;

  // Get id
  int id = v.id();
  
  // Check if the size of the function list has increased
  if ( FunctionList::size() > n )
    resize(order, FunctionList::size());
  
  // Get value
  Value value = table[0](id);
  
  // Check if integral has already been computed
  if ( value.ok() )
    return value() * fabs(det());
  
  // If the value has not been computed before, we need to compute it
  return integral(v) * fabs(det());
}
//-----------------------------------------------------------------------------
real Integral::InteriorMeasure::operator* (const FunctionSpace::Product& v)
{
  // Return zero if the measure is not active
  if ( !active )
    return 0.0;

  // Get id and number of factors
  int *id = v.id();
  unsigned int size = v.size();

  // Check if the size of the function list has increased
  if ( FunctionList::size() > n )
    resize(order, FunctionList::size());

  // Check if we need to increase the maximum number of factors
  if ( size > order )
    resize(size, n);
  
  // Get value
  Value value = table[size - 1](id);
  
  // Check if integral has already been computed
  if ( value.ok() )
    return value() * fabs(det());
  
  // If the value has not been computed before, we need to compute it
  return integral(v) * fabs(det());
}
//-----------------------------------------------------------------------------
real Integral::InteriorMeasure::integral(const FunctionSpace::ShapeFunction& v)
{
  // Compute integral using the quadrature rule
  real I = 0.0;
  for (int i = 0; i < q->size(); i++)
    I += q->weight(i) * v(q->point(i));
  
  // Set value
  table[0](v.id()).set(I);

  return I;
}
//-----------------------------------------------------------------------------
real Integral::InteriorMeasure::integral(const FunctionSpace::Product& v)
{
  // Compute integral using the quadrature rule
  real I = 0.0;
  for (int i = 0; i < q->size(); i++)
    I += q->weight(i) * v(q->point(i));
  
  // Set value
  table[v.size() - 1](v.id()).set(I);
  
  return I;
}
//-----------------------------------------------------------------------------
real Integral::InteriorMeasure::det() const
{
  // Return determinant
  return m->det();
}
//-----------------------------------------------------------------------------
void Integral::InteriorMeasure::init()
{
  // Assume that we have at most 2 factors (automatically updated)
  order = 2;

  // Check how many different shape functions we need.
  n = FunctionList::size();
  
  // Initialise the table
  table = new Tensor<Value>[order];
  for (unsigned int i = 0; i < order; i++)
    table[i].init(i+1, n);

  // Measure is inactive by default
  active = false;
}
//-----------------------------------------------------------------------------
void Integral::InteriorMeasure::resize(unsigned int new_order, unsigned int new_n)
{
  // Create a new table
  Tensor<Value>* new_table = new Tensor<Value>[new_order];
  for (unsigned int i = 0; i < new_order; i++)
    new_table[i].init(i+1, new_n);
  
  // Delete old table
  delete [] table;
  
  // Use the new table
  table = new_table;
  order = new_order;
  n = new_n;
}
//-----------------------------------------------------------------------------
// Integral::BoundaryMeasure
//-----------------------------------------------------------------------------
Integral::BoundaryMeasure::BoundaryMeasure() : Measure()
{
  init();
}
//-----------------------------------------------------------------------------
Integral::BoundaryMeasure::BoundaryMeasure(Map& m, Quadrature& q)
  : Measure(m, q), boundary(-1)
{
  init();
}
//-----------------------------------------------------------------------------
Integral::BoundaryMeasure::~BoundaryMeasure()
{
  if ( table )
    delete [] table;
  table = 0;
}
//-----------------------------------------------------------------------------
void Integral::BoundaryMeasure::update(const Map& map,
				       const Quadrature& quadrature)
{
  // Common update for measures
  Measure::update(map, quadrature);

  // Save number of boundary
  boundary = map.boundary();
  dolfin_assert(boundary < static_cast<int>(bndmax));

  // Measure is only active on boundaries (edges or faces)
  if ( boundary == -1 )
    active = false;
  else
    active = true;
}
//-----------------------------------------------------------------------------
real Integral::BoundaryMeasure::operator* (const FunctionSpace::ShapeFunction& v)
{
  // Return zero if the measure is not active
  if ( !active )
    return 0.0;

  // Get id
  int id = v.id();
  
  // Check if the size of the function list has increased
  if ( FunctionList::size() > n )
    resize(order, FunctionList::size());
  
  // Get value
  Value value = table[boundary][0](id);
  
  // Check if integral has already been computed
  if ( value.ok() )
    return value() * fabs(det());
  
  // If the value has not been computed before, we need to compute it
  return integral(v) * fabs(det());
}
//-----------------------------------------------------------------------------
real Integral::BoundaryMeasure::operator* (const FunctionSpace::Product& v)
{
  // Return zero if the measure is not active
  if ( !active )
    return 0.0;

  // Get id and number of factors
  int *id = v.id();
  unsigned int size = v.size();

  // Check if the size of the function list has increased
  if ( FunctionList::size() > n )
    resize(order, FunctionList::size());

  // Check if we need to increase the maximum number of factors
  if ( size > order )
    resize(size, n);
  
  // Get value
  Value value = table[boundary][size - 1](id);
  
  // Check if integral has already been computed
  if ( value.ok() )
    return value() * fabs(det());
  
  // If the value has not been computed before, we need to compute it
  return integral(v) * fabs(det());
}
//-----------------------------------------------------------------------------
real Integral::BoundaryMeasure::integral(const FunctionSpace::ShapeFunction& v)
{
  // Compute integral using the quadrature rule
  real I = 0.0;
  for (int i = 0; i < q->size(); i++)
    I += q->weight(i) * v((*m)(q->point(i), boundary));
  
  // Set value
  table[boundary][0](v.id()).set(I);

  return I;
}
//-----------------------------------------------------------------------------
real Integral::BoundaryMeasure::integral(const FunctionSpace::Product& v)
{
  // Compute integral using the quadrature rule
  real I = 0.0;
  for (int i = 0; i < q->size(); i++)
    I += q->weight(i) * v((*m)(q->point(i), boundary));
  
  // Set value
  table[boundary][v.size() - 1](v.id()).set(I);
  
  return I;
}
//-----------------------------------------------------------------------------
real Integral::BoundaryMeasure::det() const
{
  // Return determinant
  return m->bdet();
}
//-----------------------------------------------------------------------------
void Integral::BoundaryMeasure::init()
{
  // Assume that we have at most 2 factors (automatically updated)
  order = 2;

  // Check how many different shape functions we need.
  n = FunctionList::size();
  
  // Assume that there are at most 4 different boundaries. For a triangle,
  // we have 3 boundaries (edges) and for a tetrahedron we have 4
  // boundaries (faces).
  bndmax = 4;
  
  // Initialize the table
  table = new Tensor<Value>* [bndmax];
  for (unsigned int i = 0; i < bndmax; i++)
  {
    table[i] = new Tensor<Value>[order];
    for (unsigned int j = 0; j < order; j++)
      table[i][j].init(j+1, n);
  }

  // Measure is inactive by default
  active = false;
}
//-----------------------------------------------------------------------------
void Integral::BoundaryMeasure::resize(unsigned int new_order, unsigned int new_n)
{
  // Create a new table
  Tensor<Value>** new_table = new Tensor<Value>* [bndmax];
  for (unsigned int i = 0; i < bndmax; i++)
  {
    new_table[i] = new Tensor<Value>[new_order];
    for (unsigned int j = 0; j < new_order; j++)
      new_table[i][j].init(j+1, new_n);
  }
  
  // Delete old table
  for (unsigned int i = 0; i < bndmax; i++)
    delete [] table[i];
  delete [] table,
  
  // Use the new table
  table = new_table;
  order = new_order;
  n = new_n;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
real dolfin::operator* (real a, const Integral::Measure& dm)
{
  return dm * a;
}
//-----------------------------------------------------------------------------
