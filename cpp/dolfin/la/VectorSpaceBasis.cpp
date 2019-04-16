// Copyright (C) 2013-2019 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VectorSpaceBasis.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include <cmath>

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
    assert(_basis[i]);
    // Orthogonalize vector i with respect to previously orthonormalized
    // vectors
    for (std::size_t j = 0; j < i; ++j)
    {
      PetscScalar dot_ij = 0.0;
      VecDot(_basis[i]->vec(), _basis[j]->vec(), &dot_ij);
      VecAXPY(_basis[i]->vec(), -dot_ij, _basis[j]->vec());
    }

    // Normalise basis function
    PetscReal norm = 0.0;
    VecNormalize(_basis[i]->vec(), &norm);
    if (norm < tol)
    {
      throw std::runtime_error(
          "VectorSpaceBasis has linear dependency. Cannot orthogonalize.");
    }
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
      PetscScalar dot_ij = 0.0;
      VecDot(_basis[i]->vec(), _basis[j]->vec(), &dot_ij);

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
        PetscScalar dot_ij = 0.0;
        VecDot(_basis[i]->vec(), _basis[j]->vec(), &dot_ij);
        if (std::abs(dot_ij) > tol)
          return false;
      }
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
bool VectorSpaceBasis::in_nullspace(const PETScMatrix& A, double tol) const
{
  PETScVector y = A.create_vector(0);
  for (auto x : _basis)
  {
    MatMult(A.mat(), x->vec(), y.vec());
    const double norm = y.norm(la::Norm::l2);
    if (norm > tol)
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
void VectorSpaceBasis::orthogonalize(PETScVector& x) const
{
  for (std::size_t i = 0; i < _basis.size(); i++)
  {
    assert(_basis[i]);
    PetscScalar dot = 0.0;
    VecDot(_basis[i]->vec(), x.vec(), &dot);
    VecAXPY(x.vec(), -dot, _basis[i]->vec());
  }
}
//-----------------------------------------------------------------------------
std::size_t VectorSpaceBasis::dim() const { return _basis.size(); }
//-----------------------------------------------------------------------------
std::shared_ptr<const la::PETScVector> VectorSpaceBasis::
operator[](std::size_t i) const
{
  assert(i < _basis.size());
  return _basis[i];
}
//-----------------------------------------------------------------------------
