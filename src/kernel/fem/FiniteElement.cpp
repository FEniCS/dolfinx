// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Fredrik Bengzon and Johan Jansson, 2004.

#include <dolfin/dolfin_log.h>
#include <dolfin/FunctionSpace.h>
#include <dolfin/FiniteElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
int FiniteElement::dim() const
{
  return P->dim();
}
//-----------------------------------------------------------------------------
void FiniteElement::update(const Map& map)
{
  // Maybe this can be optimized if we use the same trial and test spaces?
  P->update(map);
  Q->update(map);
}
//-----------------------------------------------------------------------------
// FiniteElement::TrialFunctionIterator
//-----------------------------------------------------------------------------
FiniteElement::TrialFunctionIterator::TrialFunctionIterator
(const FiniteElement& element) : e(&element), v(*(element.P))
{
  
}
//-----------------------------------------------------------------------------
FiniteElement::TrialFunctionIterator::TrialFunctionIterator
(const FiniteElement* element) : e(element), v(*(element->P))
{
  
}
//-----------------------------------------------------------------------------
int FiniteElement::TrialFunctionIterator::dof(const Cell& cell) const
{
  return v.dof(cell);
}
//-----------------------------------------------------------------------------
real FiniteElement::TrialFunctionIterator::dof
(const Cell &cell, const ExpressionFunction& f, real t) const
{
  return v.dof(cell, f, t);
}
//-----------------------------------------------------------------------------
int FiniteElement::TrialFunctionIterator::index() const
{
  return v.index();
}
//-----------------------------------------------------------------------------
bool FiniteElement::TrialFunctionIterator::end() const
{
  return v.end();
}
//-----------------------------------------------------------------------------
void FiniteElement::TrialFunctionIterator::operator++()
{
  ++v;
}
//-----------------------------------------------------------------------------
FiniteElement::TrialFunctionIterator::operator
FunctionSpace::ShapeFunction() const
{
  return *v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction&
FiniteElement::TrialFunctionIterator::operator*() const
{
  return *v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction*
FiniteElement::TrialFunctionIterator::operator->() const
{
  return v.pointer();
}
//-----------------------------------------------------------------------------
// FiniteElement::TestFunctionIterator
//-----------------------------------------------------------------------------
FiniteElement::TestFunctionIterator::TestFunctionIterator
(const FiniteElement& element) : e(&element), v(*(element.Q))
{

}
//-----------------------------------------------------------------------------
FiniteElement::TestFunctionIterator::TestFunctionIterator
(const FiniteElement* element) : e(element), v(*(element->Q))
{

}
//-----------------------------------------------------------------------------
int FiniteElement::TestFunctionIterator::dof(const Cell& cell) const
{
  return v.dof(cell);
}
//-----------------------------------------------------------------------------
real FiniteElement::TestFunctionIterator::dof
(const Cell& cell, const ExpressionFunction& f, real t) const
{
  return v.dof(cell, f, t);
}
//-----------------------------------------------------------------------------
int FiniteElement::TestFunctionIterator::index() const
{
  return v.index();
}
//-----------------------------------------------------------------------------
bool FiniteElement::TestFunctionIterator::end() const
{
  return v.end();
}
//-----------------------------------------------------------------------------
void FiniteElement::TestFunctionIterator::operator++()
{
  ++v;
}
//-----------------------------------------------------------------------------
FiniteElement::TestFunctionIterator::operator
FunctionSpace::ShapeFunction() const
{
  return *v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction&
FiniteElement::TestFunctionIterator::operator*() const
{
  return *v;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction*
FiniteElement::TestFunctionIterator::operator->() const
{
  return v.pointer();
}
//-----------------------------------------------------------------------------
// Vector function space
//-----------------------------------------------------------------------------
FiniteElement::Vector::Vector(unsigned int size)
{
  v = new FiniteElement*[size];
  _size = size;
}
//-----------------------------------------------------------------------------
/*
FiniteElement::Vector::Vector(const Vector& v)
{
  _size = v._size;
  this->v = new (FiniteElement *)[_size];
  for (int i = 0; i < _size; i++)
    this->v[i] = v.v[i];
}
*/
//-----------------------------------------------------------------------------
FiniteElement::Vector::~Vector()
{
  delete [] v;
}
//-----------------------------------------------------------------------------
FiniteElement*&
FiniteElement::Vector::operator() (int i)
{
  return v[i];
}
//-----------------------------------------------------------------------------
unsigned int FiniteElement::Vector::size() const
{
  return _size;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// FiniteElement::Vector::TrialFunctionIterator
//-----------------------------------------------------------------------------
FiniteElement::Vector::TrialFunctionIterator::TrialFunctionIterator
(FiniteElement::Vector& element) : e(element), uiter(e(0)),
				   componentiter(0),
				   shapefunction(e.size())
{
}
//-----------------------------------------------------------------------------
FiniteElement::Vector::TrialFunctionIterator::TrialFunctionIterator
(FiniteElement::Vector* element) : e(*element), uiter(e(0)),
				   componentiter(0),
				   shapefunction(e.size())
{
  
}
//-----------------------------------------------------------------------------
int FiniteElement::Vector::TrialFunctionIterator::dof(const Cell& cell) const
{
  return e.size() * uiter.dof(cell) + componentiter;
}
//-----------------------------------------------------------------------------
/*
real FiniteElement::Vector::TrialFunctionIterator::dof(const Cell &cell, const ExpressionFunction& f, real t) const
{
  return v.dof(cell, f, t);
}
*/
//-----------------------------------------------------------------------------
/*
int FiniteElement::Vector::TrialFunctionIterator::index() const
{
  return v.index();
}
*/
//-----------------------------------------------------------------------------
bool FiniteElement::Vector::TrialFunctionIterator::end() const
{
  //dolfin_debug2("%d %d", uiter.end(), componentiter);

  return uiter.end() && ((componentiter + 1) == e.size());
}
//-----------------------------------------------------------------------------
void FiniteElement::Vector::TrialFunctionIterator::operator++()
{
  ++uiter;

  if(uiter.end() && ((componentiter + 1) < e.size()))
  {
    //dolfin_debug1("stepping to next component: %d", componentiter);

    componentiter++;
    //dolfin_debug1("componentiter: %d", componentiter);
    uiter = FiniteElement::TrialFunctionIterator(e(componentiter));
  }
  else
  {
    //dolfin_debug1("stepping to next basis function: %d", uiter.index());

    //++uiter;
  }
}
//-----------------------------------------------------------------------------
/*
FiniteElement::Vector::TrialFunctionIterator::operator
FunctionSpace::ShapeFunction() const
{
  // Is this necessary?

  return *uiter;
}
*/
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::Vector&
FiniteElement::Vector::TrialFunctionIterator::operator*()
{
  //dolfin_debug2("trialfunction %d on component %d", uiter.index(), componentiter);

  // fill in member shapefunction and return it

  for(int i = 0; i < shapefunction.size(); i++)
  {
    shapefunction(i) = e(componentiter)->zero;
  }

  shapefunction(componentiter) = *uiter;

  return shapefunction;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::Vector*
FiniteElement::Vector::TrialFunctionIterator::operator->()
{
  // fill in member shapefunction and return it

  for(int i = 0; i < shapefunction.size(); i++)
  {
    shapefunction(i) = e(componentiter)->zero;
  }

  shapefunction(componentiter) = *uiter;

  return &shapefunction;
}

//-----------------------------------------------------------------------------
// FiniteElement::Vector::TestFunctionIterator
//-----------------------------------------------------------------------------
FiniteElement::Vector::TestFunctionIterator::TestFunctionIterator
(FiniteElement::Vector& element) : e(element), viter(e(0)),
				   componentiter(0), shapefunction(e.size())
{
  
}
//-----------------------------------------------------------------------------
FiniteElement::Vector::TestFunctionIterator::TestFunctionIterator
(FiniteElement::Vector* element) : e(*element), viter(e(0)),
				   componentiter(0), shapefunction(e.size())
{
  
}
//-----------------------------------------------------------------------------
int FiniteElement::Vector::TestFunctionIterator::dof(const Cell& cell) const
{
  return e.size() * viter.dof(cell) + componentiter;
}
//-----------------------------------------------------------------------------
/*
real FiniteElement::Vector::TestFunctionIterator::dof(const Cell &cell, const ExpressionFunction& f, real t) const
{
  return v.dof(cell, f, t);
}
*/
//-----------------------------------------------------------------------------
/*
int FiniteElement::Vector::TestFunctionIterator::index() const
{
  return v.index();
}
*/
//-----------------------------------------------------------------------------
bool FiniteElement::Vector::TestFunctionIterator::end() const
{
  //dolfin_debug2("%d %d", viter.end(), componentiter);

  return viter.end() && ((componentiter + 1) == e.size());
}
//-----------------------------------------------------------------------------
void FiniteElement::Vector::TestFunctionIterator::operator++()
{
  ++viter;

  if(viter.end() && ((componentiter + 1) < e.size()))
  {
    //dolfin_debug1("stepping to next component: %d", componentiter);

    componentiter++;
    viter = FiniteElement::TestFunctionIterator(e(componentiter));
  }
  else
  {
    //dolfin_debug("stepping to next basis function");

    //++viter;
  }
}
//-----------------------------------------------------------------------------
/*
FiniteElement::Vector::TestFunctionIterator::operator
FunctionSpace::ShapeFunction() const
{
  // Is this necessary?

  return *viter;
}
*/
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::Vector&
FiniteElement::Vector::TestFunctionIterator::operator*()
{
  //dolfin_debug2("testfunction %d on component %d", viter.index(), componentiter);

  // fill in member shapefunction and return it
  
  for(int i = 0; i < shapefunction.size(); i++)
  {
    shapefunction(i) = e(componentiter)->zero;
  }

  shapefunction(componentiter) = *viter;

  return shapefunction;
}
//-----------------------------------------------------------------------------
FunctionSpace::ShapeFunction::Vector*
FiniteElement::Vector::TestFunctionIterator::operator->()
{
  // fill in member shapefunction and return it

  for(int i = 0; i < shapefunction.size(); i++)
  {
    shapefunction(i) = e(componentiter)->zero;
  }

  shapefunction(componentiter) = *viter;

  return &shapefunction;
}
//-----------------------------------------------------------------------------
