#include <dolfin/Quadrature.h>
#include <dolfin/Mapping.h>
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
  // Mapping and quadrature will be initialised later
  m = 0;
  q = 0;

  // Initialise
  init();
}
//-----------------------------------------------------------------------------
Integral::Measure::Measure(const Mapping& mapping,
			   const Quadrature& quadrature)
{  
  // Save mapping and quadrature
  m = &mapping;
  q = &quadrature;
  
  // Initialise
  init();
}
//-----------------------------------------------------------------------------
Integral::Measure::~Measure()
{
  if ( table ) {
    for (int i = 0; i < order; i++)
      delete table[i];
    delete [] table;
  }
  table = 0;
}
//-----------------------------------------------------------------------------
void Integral::Measure::update(const Mapping &mapping,
			       const Quadrature &quadrature)
{
  m = &mapping;
  q = &quadrature;
}
//-----------------------------------------------------------------------------
real Integral::Measure::operator* (real a) const
{
  return a * q->measure() * fabs(m->det());
}
//-----------------------------------------------------------------------------
real Integral::Measure::operator* (const FunctionSpace::ShapeFunction &v)
{
  // Get id
  int id = v.id();
  
  // Check if the size of the function list has increased
  if ( FunctionList::size() > n )
    resize(order, FunctionList::size());
  
  // Get value
  Value value = (*(table[0]))(id);
  
  // Check if integral has already been computed
  if ( value.ok() )
    return value() * fabs(m->det());
  
  // If the value has not been computed before, we need to compute it
  return integral(v) * fabs(m->det());
}
//-----------------------------------------------------------------------------
real Integral::Measure::operator* (const FunctionSpace::Product &v)
{
  // Get id and number of factors
  int *id = v.id();
  int size = v.size();

  // Check if the size of the function list has increased
  if ( FunctionList::size() > n )
    resize(order, FunctionList::size());

  // Check if we need to increase the maximum number of factors
  if ( size > order )
    resize(size, n);
  
  // Get value
  Value value = (*(table[size - 1]))(id);
  
  // Check if integral has already been computed
  if ( value.ok() )
    return value() * fabs(m->det());
  
  // If the value has not been computed before, we need to compute it
  return integral(v) * fabs(m->det());
}
//-----------------------------------------------------------------------------
real Integral::Measure::operator* (const FunctionSpace::ElementFunction &v)
{
  return v * (*this);
}
//-----------------------------------------------------------------------------
void Integral::Measure::init()
{
  // Assume that we have at most 2 factors.
  order = 2;

  // Check how many different shape functions we need.
  n = FunctionList::size();
  
  // Initialise the table
  table = new (Tensor<Value> *)[order];
  for (int i = 0; i < order; i++)
    table[i] = new Tensor<Value>(i+1, n);
}
//-----------------------------------------------------------------------------
void Integral::Measure::resize(int new_order, int new_n)
{
  dolfin_debug("Resizing integral table.");
  dolfin_debug1("Number of factors:   %d", new_order);
  dolfin_debug1("Number of functions: %d",  new_n);
  
  // Create a new table
  Tensor<Value> **new_table = new (Tensor<Value> *)[new_order];
  for (int i = 0; i < new_order; i++)
    new_table[i] = new Tensor<Value>(i+1, new_n);
  
  // Copy the old values
  for (int i = 0; i < new_order & i < order; i++)
    new_table[i]->copy(*(table[i]));
  
  // Delete old table
  for (int i = 0; i < order; i++)
    delete table[i];
  delete [] table;
  
  // Use the new table
  table = new_table;
  order = new_order;
  n = new_n;
}
//-----------------------------------------------------------------------------
// Integral::InteriorMeasure
//-----------------------------------------------------------------------------
real Integral::InteriorMeasure::integral(const FunctionSpace::ShapeFunction &v)
{
  // Compute integral using the quadrature rule
  real I = 0.0;
  for (int i = 0; i < q->size(); i++)
    I += q->weight(i) * v(q->point(i));
  
  // Set value
  (*table[0])(v.id()).set(I);

  return I;
}
//-----------------------------------------------------------------------------
real Integral::InteriorMeasure::integral(const FunctionSpace::Product &v)
{
  // Compute integral using the quadrature rule
  real I = 0.0;
  for (int i = 0; i < q->size(); i++)
    I += q->weight(i) * v(q->point(i));
  
  // Set value
  (*table[v.size() - 1])(v.id()).set(I);
  
  return I;
}
//-----------------------------------------------------------------------------
// Integral::BoundaryMeasure
//-----------------------------------------------------------------------------
real Integral::BoundaryMeasure::integral(const FunctionSpace::ShapeFunction &v)
{
  // Not implemented
  return 0.0;
}
//-----------------------------------------------------------------------------
real Integral::BoundaryMeasure::integral(const FunctionSpace::Product &v)
{
  // Not implemented
  return 0.0;
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
real dolfin::operator* (real a, const Integral::Measure &dm)
{
  return dm * a;
}
//-----------------------------------------------------------------------------
