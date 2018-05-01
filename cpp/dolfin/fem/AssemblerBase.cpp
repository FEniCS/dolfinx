// Copyright (C) 2007-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "AssemblerBase.h"
#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "utils.h"
#include <array>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <memory>
#include <vector>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
void AssemblerBase::init_global_tensor(la::PETScVector& b, const Form& L)
{
  fem::init(b, L);
  if (!add_values)
    b.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::init_global_tensor(la::PETScMatrix& A, const Form& a)
{
  fem::init(A, a);
  if (!add_values)
    A.zero();
}
//-----------------------------------------------------------------------------
void AssemblerBase::check(const Form& a)
{
  // Extract mesh and coefficients
  assert(a.mesh());
  const mesh::Mesh& mesh = *(a.mesh());

  // Check ghost mode for interior facet integrals in parallel
  if (a.integrals().num_interior_facet_integrals() > 0
      and MPI::size(mesh.mpi_comm()) > 1)
  {
    mesh::GhostMode ghost_mode = mesh.get_ghost_mode();
    if (!(ghost_mode == mesh::GhostMode::shared_vertex
          or ghost_mode == mesh::GhostMode::shared_facet))
    {
      throw std::runtime_error(
          "Incorrect mesh ghost mode.  Expected \"shared_vertex\" or "
          "\"shared_facet\" for interior facet integrals in parallel");
    }
  }
  const auto& coefficients = a.coeffs();

  // Check that all coefficients have been set
  for (std::size_t i = 0; i < coefficients.size(); ++i)
  {
    if (!coefficients.get(i))
    {
      throw std::runtime_error("Coefficient number " + std::to_string(i)
                               + " has not been set");
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
