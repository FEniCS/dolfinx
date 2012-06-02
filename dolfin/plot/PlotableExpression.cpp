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
// First added:  2012-06-02
// Last changed: 2012-06-02

#include "PlotableExpression.h"

using namespace dolfin;
//----------------------------------------------------------------------------
PlotableExpression::PlotableExpression(const Expression& expression, const Mesh& mesh) :
  _mesh(reference_to_no_delete_pointer(mesh)),
  _expression(reference_to_no_delete_pointer(expression))
{
  // Do nothing
}
//----------------------------------------------------------------------------
