// Copyright (C) 2013 Patrick E. Farrell
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VectorSpaceBasis.h"
#include "PETScVector.h"
#include <cmath>
#include <dolfin/common/constants.h>

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
VectorSpaceBasis::VectorSpaceBasis(
    const std::vector<std::shared_ptr<PETScVector>> basis)
    : _basis(basis)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void VectorSpaceBasis::orthonormalize(double tol)
{
  // Loop over each vector in basis
  for (std::size_t i = 0; i < _basis.size(); ++i)
  {
    // Orthogonalize vector i with respect to previously
    // orthonormalized vectors
    for (std::size_t j = 0; j < i; ++j)
    {
      const double dot_ij = _basis[i]->dot(*_basis[j]);
      _basis[i]->axpy(-dot_ij, *_basis[j]);
    }

    if (_basis[i]->norm("l2") < tol)
    {
      log::dolfin_error("VectorSpaceBasis.cpp", "orthonormalize vector basis",
                   "Vector space has linear dependency");
    }

    // Normalise basis function
    (*_basis[i]) /= _basis[i]->norm("l2");
  }
}
//-----------------------------------------------------------------------------
bool VectorSpaceBasis::is_orthonormal(double tol) const
{
  for (std::size_t i = 0; i < _basis.size(); i++)
  {
    for (std::size_t j = i; j < _basis.size(); j++)
    {
      assert(_basis[i]);
      assert(_basis[j]);
      const double delta_ij = (i == j) ? 1.0 : 0.0;
      const double dot_ij = _basis[i]->dot(*_basis[j]);
      if (std::abs(delta_ij - dot_ij) > tol)
        return false;
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
bool VectorSpaceBasis::is_orthogonal(double tol) const
{
  for (std::size_t i = 0; i < _basis.size(); i++)
  {
    for (std::size_t j = i; j < _basis.size(); j++)
    {
      assert(_basis[i]);
      assert(_basis[j]);
      if (i != j)
      {
        const double dot_ij = _basis[i]->dot(*_basis[j]);
        if (std::abs(dot_ij) > tol)
          return false;
      }
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
void VectorSpaceBasis::orthogonalize(PETScVector& x) const
{
  for (std::size_t i = 0; i < _basis.size(); i++)
  {
    assert(_basis[i]);
    const double dot = _basis[i]->dot(x);
    x.axpy(-dot, *_basis[i]);
  }
}
//-----------------------------------------------------------------------------
std::size_t VectorSpaceBasis::dim() const { return _basis.size(); }
//-----------------------------------------------------------------------------
std::shared_ptr<const PETScVector> VectorSpaceBasis::
operator[](std::size_t i) const
{
  assert(i < _basis.size());
  return _basis[i];
}
//-----------------------------------------------------------------------------
