// Copyright (C) 2013 Patrick E. Farrell
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
// First added:  2013-05-29
// Last changed: 2013-05-29

#include "VectorSpaceBasis.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
VectorSpaceBasis::VectorSpaceBasis(const std::vector<const GenericVector*> basis, const bool check):
  _basis(&basis)
{
  if (check)
  {
    if (!check_orthonormality())
    {
    dolfin_error("VectorSpaceBasis.cpp",
                 "orthonormality check",
                 "Input vector space basis is not orthonormal");
    }
  }
}
//-----------------------------------------------------------------------------
