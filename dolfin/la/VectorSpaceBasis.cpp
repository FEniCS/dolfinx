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

#include <dolfin/common/constants.h>
#include "VectorSpaceBasis.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
VectorSpaceBasis::VectorSpaceBasis(std::vector<boost::shared_ptr<
                                     const GenericVector> > basis,
                                   const bool check)
  : _basis(basis)
{
  // Check that basis vectors are orthonormal
  if (check)
  {
    if (!is_orthonormal())
    {
      dolfin_error("VectorSpaceBasis.cpp",
                   "verify orthonormality",
                   "Input vector space basis is not orthonormal");
    }
  }
}
//-----------------------------------------------------------------------------
bool VectorSpaceBasis::is_orthonormal() const
{
  for (std::size_t i = 0; i < _basis.size(); i++)
  {
    for (std::size_t j = i; j < _basis.size(); j++)
    {
      dolfin_assert(_basis[i]);
      dolfin_assert(_basis[j]);
      const double delta_ij = (i == j) ? 1.0 : 0.0;
      const double dot_ij = _basis[i]->inner(*_basis[j]);
      if (abs(delta_ij - dot_ij) > DOLFIN_EPS)
        return false;
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
void VectorSpaceBasis::orthogonalize(GenericVector& x) const
{
  for (std::size_t i = 0; i < _basis.size(); i++)
  {
    dolfin_assert(_basis[i]);
    const double dot = _basis[i]->inner(x);
    x.axpy(-dot, *_basis[i]);
  }
}
//-----------------------------------------------------------------------------
std::size_t VectorSpaceBasis::size() const
{
  return _basis.size();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericVector>
VectorSpaceBasis::operator[] (std::size_t i) const
{
  dolfin_assert(i < _basis.size());
  return _basis[i];
}
//-----------------------------------------------------------------------------
