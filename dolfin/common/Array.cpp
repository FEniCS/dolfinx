// Copyright (C) 2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-06
// Last changed:

#include "Array.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Array::Array(uint N): _size(N), x(new double(N))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Array::Array(const Array& x)
{
  error("Not implemented");
}
//-----------------------------------------------------------------------------
Array::Array(uint N, boost::shared_array<double> x) : _size(N), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Array::~Array()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Array::resize(uint N)
{
  error("Not implemented");
}
//-----------------------------------------------------------------------------
dolfin::uint Array::size() const
{
  return _size;
}
//-----------------------------------------------------------------------------
void Array::zero()
{
  error("No implemented");
}
//-----------------------------------------------------------------------------
double Array::min() const
{
  error("No implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
double Array::max() const
{
  error("No implemented");
  return 0.0;
}
//-----------------------------------------------------------------------------
std::string Array::str(bool verbose) const
{
  error("No implemented");
  return "";
}
//-----------------------------------------------------------------------------

