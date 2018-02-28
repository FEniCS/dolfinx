// Copyright (C) 2007-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "AssemblerBase.h"
#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "SparsityPatternBuilder.h"
#include "utils.h"
#include <array>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <memory>
#include <vector>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
void AssemblerBase::init_global_tensor(PETScVector& x, const Form& a)
{
  fem::init(x, a);
  if (!add_values)
    x.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::init_global_tensor(PETScMatrix& A, const Form& a)
{
  fem::init(A, a);
  if (!add_values)
    A.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::check(const Form& a)
{
  // Extract mesh and coefficients
  dolfin_assert(a.mesh());
  const mesh::Mesh& mesh = *(a.mesh());

  // Check ghost mode for interior facet integrals in parallel
  if (a.integrals().num_interior_facet_integrals() > 0
      && MPI::size(mesh.mpi_comm()) > 1)
  {
    std::string ghost_mode = mesh.ghost_mode();
    if (!(ghost_mode == "shared_vertex" || ghost_mode == "shared_facet"))
    {
      dolfin_error("AssemblerBase.cpp", "assemble form",
                   "Incorrect mesh ghost mode \"%s\" (expected "
                   "\"shared_vertex\" or \"shared_facet\" for "
                   "interior facet integrals in parallel)",
                   ghost_mode.c_str());
    }
  }

  const auto& coefficients = a.coeffs();

  // Check that all coefficients have been set
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!coefficients.get(i))
    {
      dolfin_error("AssemblerBase.cpp", "assemble form",
                   "Coefficient number %d has not been set", i);
    }
  }
}
//-----------------------------------------------------------------------------
std::string AssemblerBase::progress_message(std::size_t rank,
                                            std::string integral_type)
{
  std::stringstream s;
  s << "Assembling ";

  switch (rank)
  {
  case 0:
    s << "scalar value over ";
    break;
  case 1:
    s << "vector over ";
    break;
  case 2:
    s << "matrix over ";
    break;
  default:
    s << "rank " << rank << " tensor over ";
    break;
  }

  s << integral_type;

  return s.str();
}
//-----------------------------------------------------------------------------
