// Copyright (C) 2006-2012 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-07-07
// Last changed: 2012-08-23

#include "uBLASLinearOperator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASLinearOperator::uBLASLinearOperator() : _wrapper(0), M(0), N(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t uBLASLinearOperator::size(std::size_t dim) const
{
  if (dim == 0)
    return M;
  else if (dim == 1)
    return N;
  else
  {
    dolfin_error("uBLASLinearOperator.h",
                 "return size of uBLASLinearOperator",
                 "Illegal dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void uBLASLinearOperator::mult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(_wrapper);
  _wrapper->mult(x, y);
}
//-----------------------------------------------------------------------------
std::string uBLASLinearOperator::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for uBLASLinearOperator not implemented.");
    s << str(false);
  }
  else
  {
    s << "<uBLASLinearOperator>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void uBLASLinearOperator::init(std::size_t M, std::size_t N, GenericLinearOperator* wrapper)
{
  // Store dimensions
  this->M = M;
  this->N = N;

  // Store wrapper
  _wrapper = wrapper;
}
//-----------------------------------------------------------------------------
