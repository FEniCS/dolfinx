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
// Last changed: 2016-03-02

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Array.h>
#include <dolfin/function/MultiMeshFunctionSpace.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include "DirichletBC.h"
#include "MultiMeshDirichletBC.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                                           std::shared_ptr<const GenericFunction> g,
                                           std::shared_ptr<const SubDomain> sub_domain,
                                           std::string method,
                                           bool check_midpoint,
                                           bool exclude_overlapped_boundaries)
  : _sub_domain(0),
    _exclude_overlapped_boundaries(exclude_overlapped_boundaries)
{
  log(PROGRESS, "Initializing multimesh Dirichlet boundary conditions.");

  // Initialize subdomain wrapper
  _sub_domain.reset(new MultiMeshSubDomain(sub_domain,
                                           V->multimesh(),
                                           _exclude_overlapped_boundaries));

  // Iterate over parts
  for (std::size_t part = 0; part < V->num_parts(); part++)
  {
    // Get view of function space for part
    std::shared_ptr<const FunctionSpace> V_part = V->view(part);

    // Create Dirichlet boundary condition for part
    std::shared_ptr<DirichletBC> bc(new DirichletBC(V_part,
                                                    g,
                                                    _sub_domain,
                                                    method,
                                                    check_midpoint));

    // Add to list
    _bcs.push_back(bc);
  }
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshDirichletBC(std::shared_ptr<const MultiMeshFunctionSpace> V,
                                          std::shared_ptr<const GenericFunction> g,
                                          std::shared_ptr<const MeshFunction<std::size_t>> sub_domains,
                                          std::size_t sub_domain,
                                          std::size_t part,
                                          std::string method)
  : _sub_domain(0),
    _exclude_overlapped_boundaries(false)
{
  // Get view of function space for part
  std::shared_ptr<const FunctionSpace> V_part = V->view(part);

  // Create Dirichlet boundary condition for part
  std::shared_ptr<DirichletBC> bc(new DirichletBC(V_part,
                                                  g,
                                                  sub_domains,
                                                  sub_domain,
                                                  method));

  // Add to list. Note that in this case (as opposed to the case when the
  // boundary conditions are specified in terms of a subdomain) we will only
  // have a single boundary condition.
  _bcs.push_back(bc);
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::~MultiMeshDirichletBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A) const
{
  // Check whether we have a list of boundary conditions, one for each
  // part, or if we have a single boundary condition for a single
  // part.

  if (_sub_domain)
  {
    // Iterate over boundary conditions
    for (std::size_t part = 0; part < _bcs.size(); part++)
    {
      // Set current part for subdomain wrapper
      dolfin_assert(_sub_domain);
      _sub_domain->set_current_part(part);

      // Apply boundary condition for current part
      _bcs[part]->apply(A);
    }
  }
  else
  {
    dolfin_assert(_bcs.size() == 1);

    // Apply the single boundary condition
    _bcs[0]->apply(A);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericVector& b) const
{
  // Iterate over boundary conditions
  for (std::size_t part = 0; part < _bcs.size(); part++)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(part);

    // Apply boundary condition
    _bcs[part]->apply(b);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A,
                                 GenericVector& b) const
{
  // Iterate over boundary conditions
  for (std::size_t part = 0; part < _bcs.size(); part++)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(part);

    // Apply boundary condition
    _bcs[part]->apply(A, b);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericVector& b,
                                 const GenericVector& x) const
{
  // Iterate over boundary conditions
  for (std::size_t part = 0; part < _bcs.size(); part++)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(part);

    // Apply boundary condition
    _bcs[part]->apply(b, x);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::apply(GenericMatrix& A,
                                 GenericVector& b,
                                 const GenericVector& x) const
{
  // Iterate over boundary conditions
  for (std::size_t part = 0; part < _bcs.size(); part++)
  {
    // Set current part for subdomain wrapper
    dolfin_assert(_sub_domain);
    _sub_domain->set_current_part(part);

    // Apply boundary condition
    _bcs[part]->apply(A, b, x);
  }
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshSubDomain::MultiMeshSubDomain
(std::shared_ptr<const SubDomain> sub_domain,
 std::shared_ptr<const MultiMesh> multimesh,
 bool exclude_overlapped_boundaries)
  : _user_sub_domain(sub_domain),
    _multimesh(multimesh),
    _current_part(0),
    _exclude_overlapped_boundaries(exclude_overlapped_boundaries)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MultiMeshDirichletBC::MultiMeshSubDomain::~MultiMeshSubDomain()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool MultiMeshDirichletBC::MultiMeshSubDomain::inside(const Array<double>& x,
                                                      bool on_boundary) const
{
  dolfin_assert(_user_sub_domain);

  // If point is on boundary, check that it really is on the boundary,
  // which it may not be if it is contained in some other mesh.
  if (on_boundary && _exclude_overlapped_boundaries)
  {
    for (std::size_t part = 0; part < _multimesh->num_parts(); part++)
    {
      // Skip current part
      if (part == _current_part)
        continue;

      // Check whether point is contained in other mesh
      const Point point(x.size(), x.data());
      if (_multimesh->bounding_box_tree(part)->collides_entity(point))
      {
        on_boundary = false;
        break;
      }
    }
  }

  // Call user-defined function with possibly modified on_boundary
  return _user_sub_domain->inside(x, on_boundary);
}
//-----------------------------------------------------------------------------
void MultiMeshDirichletBC::MultiMeshSubDomain::set_current_part
(std::size_t current_part)
{
  _current_part = current_part;
}
//-----------------------------------------------------------------------------
