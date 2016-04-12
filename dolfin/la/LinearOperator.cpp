// Copyright (C) 2012 Anders Logg
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
// First added:  2012-08-20
// Last changed: 2012-12-12

#include "DefaultFactory.h"
#include "GenericVector.h"
#include "LinearOperator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearOperator::LinearOperator()
{
  // Initialization is postponed until the backend is accessed to
  // enable accessing the member function size() to extract the size.
  // The size would otherwise need to be passed to the constructor of
  // LinearOperator which is often impractical for subclasses.
}
//-----------------------------------------------------------------------------
LinearOperator::LinearOperator(const GenericVector& x,
                               const GenericVector& y)
{
  // Create concrete implementation
  DefaultFactory factory;
  _matA = factory.create_linear_operator(x.mpi_comm());
  dolfin_assert(_matA);

  // Initialize implementation
  _matA->init_layout(x, y, this);
}
//-----------------------------------------------------------------------------
std::string LinearOperator::str(bool verbose) const
{
  return "<User-defined linear operator>";
}
//-----------------------------------------------------------------------------
const GenericLinearOperator* LinearOperator::instance() const
{
  return _matA.get();
}
//-----------------------------------------------------------------------------
GenericLinearOperator* LinearOperator::instance()
{
  return _matA.get();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const LinearAlgebraObject> LinearOperator::shared_instance() const
{
  return _matA;
}
//-----------------------------------------------------------------------------
std::shared_ptr<LinearAlgebraObject> LinearOperator::shared_instance()
{
  return _matA;
}
//-----------------------------------------------------------------------------
