// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/FunctionSpace.h>
#include <dolfin/FiniteElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
int FiniteElement::dim() const
{
  return P.dim();
}
//-----------------------------------------------------------------------------
void FiniteElement::update(const Mapping* mapping)
{
  // Maybe this can be optimized if we use the same trial and test spaces?
  P.update(*mapping);
  Q.update(*mapping);
}
//-----------------------------------------------------------------------------
// FiniteElement::TrialFunctionIterator
//-----------------------------------------------------------------------------
FiniteElement::TrialFunctionIterator::TrialFunctionIterator
(const FiniteElement& element) : e(element), v(element.P)
{
  
}
//-----------------------------------------------------------------------------
FiniteElement::TrialFunctionIterator::TrialFunctionIterator
(const FiniteElement* element) : e(*element), v(element->P)
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
(const FiniteElement& element) : e(element), v(element.Q)
{

}
//-----------------------------------------------------------------------------
FiniteElement::TestFunctionIterator::TestFunctionIterator
(const FiniteElement* element) : e(*element), v(element->Q)
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
