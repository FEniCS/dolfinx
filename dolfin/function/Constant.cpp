// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2009.
//
// First added:  2006-02-09
// Last changed: 2009-09-30

#include <dolfin/log/log.h>
#include "Constant.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Constant::Constant(double value)
  : _size(1), _values(0)
{
  _values = new double[1];
  _values[0] = value;
}
//-----------------------------------------------------------------------------
Constant::Constant(uint size, double value)
  : _size(size), _values(0)
{
  assert(size > 0);

  _values = new double[size];
  for (uint i = 0; i < size; i++)
    _values[i] = value;
}
//-----------------------------------------------------------------------------
Constant::Constant(const std::vector<double>& values)
  : _size(values.size()), _values(0)
{
  assert(values.size() > 0);

  _values = new double[values.size()];
  for (uint i = 0; i < values.size(); i++)
    _values[i] = values[i];
}
//-----------------------------------------------------------------------------
Constant::Constant(const std::vector<uint>& shape,
                   const std::vector<double>& values)
  : _size(0), _values(0)
{
  assert(shape.size() > 0);
  assert(values.size() > 0);

  // Compute size
  _size = 1;
  for (uint i = 0; i < shape.size(); i++)
    _size *= shape[i];

  // Copy values
  assert(values.size() == _size);
  _values = new double[values.size()];
  for (uint i = 0; i < values.size(); i++)
    _values[i] = values[i];
}
//-----------------------------------------------------------------------------
Constant::Constant(const Constant& c)
  : _size(0), _values(0)
{
  *this = c;
}
//-----------------------------------------------------------------------------
Constant::~Constant()
{
  delete [] _values;
}
//-----------------------------------------------------------------------------
const Constant& Constant::operator= (const Constant& c)
{
  assert(c._size > 0);
  assert(c._values);

  delete _values;
  _size = c._size;
  _values = new double[c._size];
  for (uint i = 0; i < c._size; i++)
    _values[i] = c._values[i];

  return *this;
}
//-----------------------------------------------------------------------------
const Constant& Constant::operator= (double c)
{
  if(_size > 1)
    error("Cannot convert non-scalar Constant to a double.");
  _values[0] = c;
  return *this;
}
//-----------------------------------------------------------------------------
void Constant::eval(double* values, const Data& data) const
{
  assert(values);
  assert(_values);

  // Copy values
  for (uint i = 0; i < _size; i++)
    values[i] = _values[i];
}
//-----------------------------------------------------------------------------
