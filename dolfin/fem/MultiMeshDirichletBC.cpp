// Copyright (C) 2014 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 4 of the License, or
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
// First added:  2014-05-12
// Last changed: 2014-05-12

#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/log.h>
#include <dolfin/function/MultiMeshFunctionSpace.h>
#include "DirichletBC.h"
#include "MultiMeshDirichletBC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshDirichletBC(const MultiMeshFunctionSpace& V,
                                           const GenericFunction& g,
                                           const SubDomain& sub_domain,
                                           std::string method,
                                           bool check_midpoint)
{
  // Initialize boundary conditions for parts
  init(reference_to_no_delete_pointer(V),
       reference_to_no_delete_pointer(g),
       reference_to_no_delete_pointer(sub_domain),
       method,
       check_midpoint);
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                                           std::shared_ptr<const GenericFunction> g,
                                           std::shared_ptr<const SubDomain> sub_domain,
                                           std::string method,
                                           bool check_midpoint)
{
  init(V, g, sub_domain, method, check_midpoint);
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::~MultiMeshDirichletBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A) const
{
  // Iterate over boundary conditions for parts and apply
  for (auto bc : _bcs)
    bc->apply(A);
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericVector& b) const
{
  // Iterate over boundary conditions for parts and apply
  for (auto bc : _bcs)
    bc->apply(b);
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A,
                                 GenericVector& b) const
{
  // Iterate over boundary conditions for parts and apply
  for (auto bc : _bcs)
    bc->apply(A, b);
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericVector& b,
                                 const GenericVector& x) const
{
  // Iterate over boundary conditions for parts and apply
  for (auto bc : _bcs)
    bc->apply(b, x);
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A,
                                 GenericVector& b,
                                 const GenericVector& x) const
{
  // Iterate over boundary conditions for parts and apply
  for (auto bc : _bcs)
    bc->apply(A, b, x);
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::init(std::shared_ptr<const MultiMeshFunctionSpace> V,
                                std::shared_ptr<const GenericFunction> g,
                                std::shared_ptr<const SubDomain> sub_domain,
                                std::string method,
                                bool check_midpoint)
{
  log(PROGRESS, "Initializing multimesh Dirichlet boundary conditions.");

  // Clear old data if any
  _bcs.clear();

  // Iterate over parts
  for (std::size_t i = 0; i < V->num_parts(); i++)
  {
    // Get view of function space for part
    std::shared_ptr<const FunctionSpace> V_i = V->view(i);

    // Create Dirichlet boundary condition for part
    std::shared_ptr<DirichletBC> bc(new DirichletBC(V_i,
                                                    g,
                                                    sub_domain,
                                                    method,
                                                    check_midpoint));

    // Add to list
    _bcs.push_back(bc);
  }
}
//-----------------------------------------------------------------------------
