// Copyright (C) 2013 Anders Logg
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
// First added:  2013-09-12
// Last changed: 2013-09-18

#include <dolfin/log/log.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/function/CCFEMFunctionSpace.h>

#include "SparsityPatternBuilder.h"
#include "CCFEMForm.h"
#include "CCFEMAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void CCFEMAssembler::assemble(GenericTensor& A, const CCFEMForm& a)
{
  begin(PROGRESS, "Assembling tensor over CCFEM function space.");

  // Initialize global tensor
  init_global_tensor(A, a);

  end();
}
//-----------------------------------------------------------------------------
void CCFEMAssembler::init_global_tensor(GenericTensor& A, const CCFEMForm& a)
{
  // This function initializes the big system matrix corresponding to
  // all dofs (including inactive dofs) on all parts of the CCFEM
  // function space.

  // Create layout for initializing tensor
  boost::shared_ptr<TensorLayout> tensor_layout;
  tensor_layout = A.factory().create_layout(a.rank());
  dolfin_assert(tensor_layout);

  // Get dimensions
  std::vector<std::size_t> global_dimensions;
  std::vector<std::pair<std::size_t, std::size_t> > local_ranges;
  std::vector<std::size_t> block_sizes;
  for (std::size_t i = 0; i < a.rank(); i++)
  {
    boost::shared_ptr<const CCFEMFunctionSpace> V = a.function_space(i);
    dolfin_assert(V);

    global_dimensions.push_back(V->dim());
    local_ranges.push_back(std::make_pair(0, V->dim())); // FIXME: not parallel
  }

  // Set block size
  const std::size_t block_size = 1;

  // Initialise tensor layout
  tensor_layout->init(global_dimensions, block_size, local_ranges);

  // Build sparsity pattern if required
  if (tensor_layout->sparsity_pattern())
  {
    GenericSparsityPattern& pattern = *tensor_layout->sparsity_pattern();
    SparsityPatternBuilder::build_ccfem(pattern, a);
  }
}
//-----------------------------------------------------------------------------
