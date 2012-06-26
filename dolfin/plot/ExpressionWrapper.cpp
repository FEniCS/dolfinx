// Copyright (C) 2012 Fredrik Valdmanis
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
// Modified by Anders Logg 2012
//
// First added:  2012-06-02
// Last changed: 2012-06-19

#include "ExpressionWrapper.h"

using namespace dolfin;

//----------------------------------------------------------------------------
ExpressionWrapper::ExpressionWrapper(boost::shared_ptr<const Expression> expression,
                                     boost::shared_ptr<const Mesh> mesh) :
  _mesh(mesh), _expression(expression)
{
  // Do nothing
}
//----------------------------------------------------------------------------
