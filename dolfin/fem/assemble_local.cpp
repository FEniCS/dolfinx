// Copyright (C) 2015 Tormod Landet
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
// First added:  2015-09-22

#include <Eigen/Dense>

#include <dolfin/fem/LocalAssembler.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/mesh/Cell.h>
#include "Form.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include "assemble_local.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble_local(const Form& a,
                            const Cell& cell,
                            std::vector<double>& tensor)
{
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e;
  UFC ufc(a);
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;

  // Get size of local tensor
  std::size_t N, M;
  if (a.rank() == 0)
  {
    N = 1;
    M = 1;
  }
  else if (a.rank() == 1)
  {
    N = a.function_space(0)->dofmap()->cell_dofs(cell.index()).size();
    M = 1;
  }
  else
  {
    N = a.function_space(0)->dofmap()->cell_dofs(cell.index()).size();
    M = a.function_space(1)->dofmap()->cell_dofs(cell.index()).size();
  }

  // Extract cell_domains etc from the form
  const MeshFunction<std::size_t>* cell_domains =
    a.cell_domains().get();
  const MeshFunction<std::size_t>* exterior_facet_domains =
    a.exterior_facet_domains().get();
  const MeshFunction<std::size_t>* interior_facet_domains =
    a.interior_facet_domains().get();

  // Update to the local cell and assemble
  A_e.resize(N, M);
  cell.get_coordinate_dofs(coordinate_dofs);
  LocalAssembler::assemble(A_e, ufc, coordinate_dofs,
                           ufc_cell, cell, cell_domains,
                           exterior_facet_domains, interior_facet_domains);
  
  // Copy values into the return tensor
  tensor.resize(N*M);
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      tensor[i*M+j] = A_e(i,j);
}
//-----------------------------------------------------------------------------
