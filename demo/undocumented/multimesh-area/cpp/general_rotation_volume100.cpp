// Copyright (C) 2014 August Johansson and Anders Logg
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
// First added:  2014-03-10
// Last changed: 2017-02-09
//


//#define CGAL_HEADER_ONLY 1
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/generation/UnitSquareMesh.h>

#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC

// We need to use epeck here. Quotient<MP_FLOAT> as number type gives overflow
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>
#endif

#include "common.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<MultiMesh> test_volume_2d_rot(std::size_t num_meshes)
{
  // Background mesh
  auto mesh_0  = std::make_shared<UnitSquareMesh>(1, 1);
  mesh_0->scale(10.0);
  mesh_0->translate(Point(-5,-5));

  // List of meshes
  std::vector<std::shared_ptr<const Mesh> > meshes;
  meshes.reserve(num_meshes + 1);
  meshes.push_back(mesh_0);

  for (std::size_t i = 0; i < num_meshes; ++i)
  {
    auto mesh = std::make_shared<UnitSquareMesh>(1, 1);
    const double angle = 2*DOLFIN_PI*i / num_meshes;
    mesh->translate(Point(-0.5, -0.5));
    mesh->scale(2.0);
    mesh->rotate(180.0*angle / DOLFIN_PI);
    mesh->translate(Point(cos(angle), sin(angle)));
    meshes.push_back(mesh);
  }

  // Create multimesh
  const std::size_t quadrature_order = 1;
  std::shared_ptr<MultiMesh> multimesh(new MultiMesh(meshes, quadrature_order));

  return multimesh;
}


//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  // const std::size_t num_meshes = argc == 1 ? 2 : atoi(argv[1]);
  const std::size_t num_meshes = 8;

  const std::vector<std::size_t> all_num_meshes = { num_meshes };

  for (const std::size_t num_meshes: all_num_meshes)
  {
    std::cout << "\n\nnum_meshes = " << num_meshes << std::endl;

    std::shared_ptr<MultiMesh> m = test_volume_2d_rot(num_meshes);
    const MultiMesh& multimesh = *m;

    // Compute volume of each cell using dolfin::MultiMesh
    std::vector<std::vector<std::pair<CELL_STATUS, double> > > cell_status_multimesh;
    std::vector<std::vector<std::size_t> > num_qr = compute_volume(multimesh, cell_status_multimesh);
    std::cout << "Done computing volumes with multimesh" << std::endl;
    double multimesh_volume = 0.;

#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
    // Compute volume of each cell using cgal
    std::vector<std::vector<std::pair<CELL_STATUS, FT>>> cell_status_cgal;
    get_cells_status_cgal(multimesh, cell_status_cgal);
    std::cout << "Done computing volumes with cgal" << std::endl;
    FT cgal_volume = 0.;
    dolfin_assert(cell_status_cgal.size() == cell_status_multimesh.size());
    dolfin_assert(cell_status_cgal.size() == num_qr.size());
#endif

    dolfin_assert(cell_status_multimesh.size() == num_qr.size());

    for (std::size_t i = 0; i < cell_status_multimesh.size(); i++)
    {
      const std::vector<std::pair<CELL_STATUS, double> >& current_multimesh = cell_status_multimesh[i];
      dolfin_assert(current_multimesh.size() == num_qr[i].size());

#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
      const std::vector<std::pair<CELL_STATUS, FT> >& current_cgal = cell_status_cgal[i];
      dolfin_assert(current_cgal.size() == current_multimesh.size());
#endif

      std::cout << "Cells in part " << i << ": " << std::endl;
      for (std::size_t j = 0; j < current_multimesh.size(); j++)
      {
	std::cout << "  Cell " << j << std::endl;
	std::cout << "    Multimesh: " << cell_status_str(current_multimesh[j].first) << " (" << current_multimesh[j].second << ")" << std::endl;

#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
	const FT error = current_cgal[j].second - current_multimesh[j].second;
	const double relative_error = std::abs(CGAL::to_double(error / current_cgal[j].second));
	const double tol = std::max(3*num_qr[i][j]*DOLFIN_EPS, DOLFIN_EPS_LARGE);
	std::cout << "    CGAL:      " << cell_status_str(current_cgal[j].first) << " (" << current_cgal[j].second << ")" << std::endl;
	std::cout << "      Diff:    " << error <<" (relative error " << relative_error << ", tol " << tol << ")" << std::endl;
	cgal_volume += current_cgal[j].second;
	dolfin_assert(std::abs(CGAL::to_double(error)) < DOLFIN_EPS_LARGE or
		      relative_error < tol);
	dolfin_assert(current_cgal[j].first == current_multimesh[j].first);
#endif

	multimesh_volume += current_multimesh[j].second;
	dolfin_assert((current_multimesh[j].first == CUT) ? num_qr[i][j] > 0 : true);
      }
      std::cout << std::endl;
    }

    // Exact volume is known
    const double exact_volume = 100;

    std::cout << "Total volume" << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << "Multimesh: " << multimesh_volume << ", error: " << (exact_volume-multimesh_volume) << std::endl;
#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
    std::cout << "CGAL:      " << cgal_volume << ", error: " << (exact_volume-cgal_volume) << std::endl;
#endif
    const double relative_error = std::abs(exact_volume - multimesh_volume) / exact_volume;

    dolfin_assert(relative_error < 1e-10);
  }


}
