// Copyright (C) 2013 Garth N. Wells
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
// First added:  2013-02-12
// Last changed:

#include <armadillo>

#include <dolfin/la/GenericVector.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include "GenericDofMap.h"
#include "Form.h"
#include "UFC.h"

#include "LocalSolver.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void LocalSolver::solve(GenericVector& x, const Form& a, const Form& L,
                        bool symmetric) const
{
  UFC ufc_a(a);
  UFC ufc_L(L);

  // Set timer
  Timer timer("Local solver");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Update off-process coefficients
  const std::vector<boost::shared_ptr<const GenericFunction> >
    coefficients_a = a.coefficients();
  for (std::size_t i = 0; i < coefficients_a.size(); ++i)
    coefficients_a[i]->update();
  const std::vector<boost::shared_ptr<const GenericFunction> >
    coefficients_L = L.coefficients();
  for (std::size_t i = 0; i < coefficients_L.size(); ++i)
    coefficients_L[i]->update();

  // Form ranks
  const std::size_t rank_a = ufc_a.form.rank();
  const std::size_t rank_L = ufc_L.form.rank();

  // Check form ranks
  dolfin_assert(rank_a == 2);
  dolfin_assert(rank_L == 1);

  // Collect pointers to dof maps
  boost::shared_ptr<const GenericDofMap> dofmap_a0 = a.function_space(0)->dofmap();
  boost::shared_ptr<const GenericDofMap> dofmap_a1 = a.function_space(1)->dofmap();
  boost::shared_ptr<const GenericDofMap> dofmap_L = a.function_space(0)->dofmap();
  dolfin_assert(dofmap_a0);
  dolfin_assert(dofmap_a1);
  dolfin_assert(dofmap_L);

  // Initialise vector
  std::pair<std::size_t, std::size_t> local_range = dofmap_L->ownership_range();
  x.resize(local_range);

  // Cell integrals
  ufc::cell_integral* integral_a = ufc_a.default_cell_integral.get();
  ufc::cell_integral* integral_L = ufc_L.default_cell_integral.get();

  // Armadillo data structures
  arma::mat A;
  arma::vec b, x_local;

  // Assemble over cells
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_a.update(*cell);
    ufc_L.update(*cell);

    // Get local-to-global dof maps for cell
    const std::vector<dolfin::la_index>& dofs_a0 = dofmap_a0->cell_dofs(cell->index());
    const std::vector<dolfin::la_index>& dofs_a1 = dofmap_a1->cell_dofs(cell->index());
    const std::vector<dolfin::la_index>& dofs_L  = dofmap_L->cell_dofs(cell->index());

    // Check that local problem is square and a and L match
    dolfin_assert(dofs_a0.size() == dofs_a1.size());
    dolfin_assert(dofs_a1.size() == dofs_L.size());

    // Resize A and b
    A.set_size(dofs_a0.size(), dofs_a1.size());
    b.set_size(dofs_L.size());

    // Tabulate A, and b on cell
    integral_a->tabulate_tensor(A.memptr(), ufc_a.w(), ufc_a.cell);
    integral_L->tabulate_tensor(b.memptr(), ufc_L.w(), ufc_L.cell);

    // Solve local problem (Armadillo uses column-major)
    if (symmetric)
      arma::solve(x_local, A, b);
    else
      arma::solve(x_local, A.t(), b);

    // Set solution in global vector
    x.set(x_local.memptr(), dofs_a0.size(), dofs_a0.data());

    p++;
  }

  // Finalise vector
  x.apply("insert");
}
//-----------------------------------------------------------------------------
