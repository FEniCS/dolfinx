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
#include "LinearOperator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearOperator::LinearOperator(const GenericVector& x,
                               const GenericVector& y)
{
  // Create concrete implementation
  DefaultFactory factory;
  _A = factory.create_linear_operator();
  dolfin_assert(_A);

  // Initialize implementation
  _A->init_layout(x, y, this);
}
//-----------------------------------------------------------------------------
LinearOperator::LinearOperator()
{
  // Initialization is postponed until the backend is accessed to
  // enable accessing the member function size() to extract the size.
  // The size would otherwise need to be passed to the constructor of
  // LinearOperator which is often unpractical for subclasses.
}
//-----------------------------------------------------------------------------
std::string LinearOperator::str(bool verbose) const
{
  return "<User-defined linear operator>";
}
//-----------------------------------------------------------------------------
const GenericLinearOperator* LinearOperator::instance() const
{
  return _A.get();
}
//-----------------------------------------------------------------------------
GenericLinearOperator* LinearOperator::instance()
{
  return _A.get();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const LinearAlgebraObject> LinearOperator::shared_instance() const
{
  return _A;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<LinearAlgebraObject> LinearOperator::shared_instance()
{
  return _A;
}
//-----------------------------------------------------------------------------
