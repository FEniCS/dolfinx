// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementBlock.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementBlock::ElementBlock(int N) : 
  components(N), t0(0), t1(0), empty(true), _bytes(0)
{
  _bytes += sizeof(ElementBlock);
  _bytes += N * sizeof(Component);
}
//-----------------------------------------------------------------------------
ElementBlock::~ElementBlock()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Element* ElementBlock::createElement(Element::Type type, real t0, real t1,
				     int q, int index)
{
  update(t0, t1);
  _bytes += sizeof(Element) + q*sizeof(real);
  return components[index].createElement(type, t0, t1, q, index);
}
//-----------------------------------------------------------------------------
Element* ElementBlock::element(unsigned int index, real t)
{
  return components[index].element(t);
}
//-----------------------------------------------------------------------------
Element* ElementBlock::last(unsigned int i)
{
  dolfin_assert(i < components.size());
  return components[i].last();
}
//-----------------------------------------------------------------------------
void ElementBlock::save()
{
  cout << "Saving block" << endl;

}
//-----------------------------------------------------------------------------
unsigned int ElementBlock::size() const
{
  return components.size();
}
//-----------------------------------------------------------------------------
unsigned int ElementBlock::bytes() const
{
  return _bytes;
}
//-----------------------------------------------------------------------------
real ElementBlock::dist(real t0, real t1) const
{
  // Distance is zero if the interval is contained in the block
  if ( (t0 >= this->t0) && (t1 <= this->t1) )
    return 0.0;

  // Interval is to the left
  if ( t0 < this->t0 )
    return this->t0 - t0;

  // Interval is to the right
  return t1 - this->t1;
}
//-----------------------------------------------------------------------------
bool ElementBlock::within(real t) const
{
  return (t0 < t) && (t <= t1);
}
//-----------------------------------------------------------------------------
void ElementBlock::update(real t0, real t1)
{
  // Save interval if this is the first element
  if ( empty )
  {
    this->t0 = t0;
    this->t1 = t1;
    empty = false;
    return;
  }

  // Otherwise, check the boundaries
  if ( t0 < this->t0 )
    this->t0 = t0;
  if ( t1 > this->t1 )
    this->t1 = t1;
}
//-----------------------------------------------------------------------------
