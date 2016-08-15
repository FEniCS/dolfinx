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
// Last changed: 2015-06-04
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include <gtest/gtest.h>
#include "MultiMeshAssemble.h"

using namespace dolfin;

// This test was commented out in the original file
// TEST(MultiMeshes, test_multiple_meshes_quadrature) {
//     set_log_level(DBG);

//     // Create multimesh from three triangle meshes of the unit square

//     // Many meshes, but not more than three overlap => this works
//     UnitCubeMesh mesh_0(11, 12, 13);
//     BoxMesh mesh_1(Point(0.1, 0.1, 0.1),    Point(0.9, 0.9, 0.9),    13, 11, 12);
//     BoxMesh mesh_2(Point(0.2, 0.2, 0.2),    Point(0.95, 0.95, 0.8),  11, 13, 11);
//     BoxMesh mesh_3(Point(0.94, 0.01, 0.01), Point(0.98, 0.99, 0.99), 1, 11, 11);
//     BoxMesh mesh_4(Point(0.01, 0.01, 0.01), Point(0.02, 0.02, 0.02), 1, 1, 1);

//     // // Completely nested 2D: can't do no more than three meshes
//     // UnitSquareMesh mesh_0(1, 1);
//     // RectangleMesh mesh_1(Point(0.1, 0.1), Point(0.9, 0.9, 1, 1);
//     // RectangleMesh mesh_2(Point(0.2, 0.2), Point(0.8, 0.8, 1, 1);
//     // RectangleMesh mesh_3(Point(0.3, 0.3), Point(0.7, 0.7, 1, 1);
//     // RectangleMesh mesh_4(Point(0.4, 0.4), Point(0.6, 0.6, 1, 1);

//     // // Completely nested 3D: can't do no more than three meshes
//     // UnitCubeMesh mesh_0(2, 3, 4);
//     // BoxMesh mesh_1(Point(0.1, 0.1, 0.1),    Point(0.9, 0.9, 0.9),    4, 3, 2);
//     // BoxMesh mesh_2(Point(0.2, 0.2, 0.2),    Point(0.8, 0.8, 0.8),    3, 4, 3);
//     // BoxMesh mesh_3(Point(0.8, 0.01, 0.01),  Point(0.9, 0.99, 0.99),  4, 2, 3);
//     // BoxMesh mesh_4(Point(0.01, 0.01, 0.01), Point(0.02, 0.02, 0.02), 1, 1, 1);

//     // Build the multimesh
//     MultiMesh multimesh;
//     multimesh.add(mesh_0);
//     multimesh.add(mesh_1);
//     multimesh.add(mesh_2);
//     multimesh.add(mesh_3);
//     multimesh.add(mesh_4);
//     multimesh.build();

//     // Exact volume is known
//     const double exact_volume = 1;
//     double volume = 0;

//     // Sum contribution from all parts
//     std::cout << "Sum contributions\n";
//     for (std::size_t part = 0; part < multimesh.num_parts(); part++)
//     {
//       std::cout << "% part " << part;
//       double part_volume = 0;

//       // Uncut cell volume given by function volume
//       const auto uncut_cells = multimesh.uncut_cells(part);
//       for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
//       {
//         const Cell cell(*multimesh.part(part), *it);
//         volume += cell.volume();
//         part_volume += cell.volume();
//       }

//       std::cout << "\t uncut volume "<< part_volume<<' ';

//       // Cut cell volume given by quadrature rule
//       const auto& cut_cells = multimesh.cut_cells(part);
//       for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
//       {
//         const auto& qr = multimesh.quadrature_rule_cut_cell(part, *it);
//         for (std::size_t i = 0; i < qr.second.size(); ++i)
//         {
//           volume += qr.second[i];
//           part_volume += qr.second[i];
//         }
//       }
//       std::cout << "\ttotal volume " << part_volume << std::endl;
//     }

//     std::cout<<std::setprecision(13) << "exact volume " << exact_volume<<'\n'
//               << "volume " << volume<<std::endl;
//     ASSERT_NEAR(exact_volume, volume, DOLFIN_EPS_LARGE);
// }

//-----------------------------------------------------------------------------
TEST(MultiMeshes, test_multiple_meshes_interface_quadrature)
{
  // // These three meshes are ok
  // UnitSquareMesh mesh_0(1, 1);
  // RectangleMesh mesh_1(Point(0.1, 0.1), Point(0.9, 0.9), 1, 1);
  // RectangleMesh mesh_2(Point(0.2, 0.2), Point(0.8, 0.8), 1, 1);
  // double exact_volume = 4*(0.9-0.1); // mesh0 and mesh1
  // exact_volume += 4*(0.8-0.2); // mesh1 and mesh2
  
  // UnitCubeMesh mesh_0(1, 2, 3);
  // BoxMesh mesh_1(Point(0.1, 0.1, 0.1),    Point(0.9, 0.9, 0.9),  2,3,4); //2, 3, 4);
  // BoxMesh mesh_2(Point(-0.1, -0.1, -0.1), Point(0.7, 0.7, 0.7),  4, 3, 2);
  // BoxMesh mesh_3(Point(0.51, 0.51, 0.51), Point( 0.7, 0.7, 0.7), 1, 1, 1); //4, 3, 2);
  // BoxMesh mesh_4(Point(0.3, 0.3, 0.3),    Point(0.7, 0.7, 0.7),  1, 1, 1);
  // double exact_volume = 0.8*0.8*6; // for mesh_0 and mesh_1
  // exact_volume += 0.4*0.4*6; // for mesh_1 and mesh_4
  
  auto mesh_0 = std::make_shared<UnitCubeMesh>(1, 1, 1);
  auto mesh_1 = std::make_shared<BoxMesh>(Point(0.1, 0.1, 0.1), Point(0.9, 0.9, 0.9), 1, 1, 1);
  auto mesh_2 = std::make_shared<BoxMesh>(Point(0.2, 0.2, 0.2), Point(0.8, 0.8, 0.8), 1, 1, 1);
  // BoxMesh mesh_3(Point(0.51, 0.51, 0.51), Point(0.7, 0.7, 0.7), 1, 1, 1); //4, 3, 2);
  // BoxMesh mesh_4(Point(0.3, 0.3, 0.3),    Point(0.7, 0.7, 0.7), 1, 1, 1);
  double exact_volume = (0.9 - 0.1)*(0.9 - 0.1)*6; // for mesh_0 and mesh_1
  exact_volume += (0.8 - 0.2)*(0.8 - 0.2)*6; // mesh_1 and mesh_2
  
  // UnitCubeMesh mesh_0(1, 1, 1);
  // MeshEditor editor;
  // Mesh mesh_1;
  // editor.open(mesh_1, 3, 3);
  // editor.init_vertices(4);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(0.7, 0.1, -0.1));
  // editor.add_vertex(1, Point(0.7, 0.3, -0.1));
  // editor.add_vertex(2, Point(0.5, 0.1, -0.1));
  // editor.add_vertex(3, Point(0.7, 0.1, 0.1));
  // editor.add_cell(0, 0,1,2,3);
  // editor.close();
  
  // Mesh mesh_2;
  // editor.open(mesh_2, 3,3);
  // editor.init_vertices(4);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(0.7, 0.1, -0.2));
  // editor.add_vertex(1, Point(0.7, 0.3, -0.2));
  // editor.add_vertex(2, Point(0.5, 0.1, -0.2));
  // editor.add_vertex(3, Point(0.7, 0.1, 0.05));
  // editor.add_cell(0, 0,1,2,3);
  // editor.close();
  
  //double exact_volume = 0.8*0.8*6; // for mesh_0 and mesh_1
  //exact_volume += 0.4*0.4*6; // for mesh_1 and mesh_4
  
  // MeshEditor editor;
  // Mesh mesh_0;
  // editor.open(mesh_0, 2, 2);
  // editor.init_vertices(3);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(0.,0.));
  // editor.add_vertex(1, Point(2.,0.));
  // editor.add_vertex(2, Point(1.,2.));
  // editor.add_cell(0, 0,1,2);
  // editor.close();
  
  // Mesh mesh_1;
  // editor.open(mesh_1, 2, 2);
  // editor.init_vertices(3);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(0.,-0.5));
  // editor.add_vertex(1, Point(2.,-0.5));
  // editor.add_vertex(2, Point(1.,1.5));
  // editor.add_cell(0, 0,1,2);
  // editor.close();
  
  // Mesh mesh_2;
  // editor.open(mesh_2, 2, 2);
  // editor.init_vertices(3);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(0.,-1.));
  // editor.add_vertex(1, Point(2.,-1.));
  // editor.add_vertex(2, Point(1.,1.));
  // editor.add_cell(0, 0,1,2);
  // editor.close();
  
  // double exact_volume = 2*std::sqrt(0.75*0.75 + 1.5*1.5); // mesh_0 and mesh_1
  // exact_volume += 2*std::sqrt(0.5*0.5 + 1*1); // mesh_0 and mesh_2
  // exact_volume += 2*std::sqrt(0.75*0.75 + 1.5*1.5); // mesh_1and mesh_2
  // double volume = 0;
  
  // // These three meshes are ok.
  // MeshEditor editor;
  // Mesh mesh_0;
  // editor.open(mesh_0, 2, 2);
  // editor.init_vertices(3);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(0.,0.));
  // editor.add_vertex(1, Point(2.,0.));
  // editor.add_vertex(2, Point(1.,2.));
  // editor.add_cell(0, 0,1,2);
  // editor.close();
  
  // Mesh mesh_1;
  // editor.open(mesh_1, 2, 2);
  // editor.init_vertices(3);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(1.5,-2.));
  // editor.add_vertex(1, Point(4.,0.));
  // editor.add_vertex(2, Point(1.5,2));
  // editor.add_cell(0, 0,1,2);
  // editor.close();
  
  // Mesh mesh_2;
  // editor.open(mesh_2, 2, 2);
  // editor.init_vertices(3);
  // editor.init_cells(1);
  // editor.add_vertex(0, Point(3.,0.5));
  // editor.add_vertex(1, Point(-1.,0.5));
  // editor.add_vertex(2, Point(1.,-1.5));
  // editor.add_cell(0, 0,1,2);
  // editor.close();
  
  // double exact_volume = (1.5-0.25) + (1-0.5); // mesh_0, mesh_1 and mesh_2
  // exact_volume += (3-1.5) + std::sqrt(1.5*1.5 + 1.5*1.5); // mesh_1 and mesh_2
  
  File("mesh_0.xml") << *mesh_0;
  File("mesh_1.xml") << *mesh_1;
  File("mesh_2.xml") << *mesh_2;

  // Build the multimesh
  MultiMesh multimesh;
  multimesh.add(mesh_0);
  multimesh.add(mesh_1);
  multimesh.add(mesh_2);
  //multimesh.add(mesh_3);
  //multimesh.add(mesh_4);
  multimesh.build();

  // Sum contribution from all parts
  std::cout << "\n\n Sum up\n\n";
  double volume = 0;
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    std::cout << "% part " << part << '\n';
    double part_volume = 0;
    
    const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);
    
    // Get collision map
    const auto& cmap = multimesh.collision_map_cut_cells(part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      const unsigned int cut_cell_index = it->first;
      
      // Iterate over cutting cells
      const auto& cutting_cells = it->second;
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
        //const std::size_t cutting_part = jt->first;
        //const std::size_t cutting_cell_index = jt->second;
        
        // Get quadrature rule for interface part defined by
        // intersection of the cut and cutting cells
        const std::size_t k = jt - cutting_cells.begin();
        dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
        const auto& qr = quadrature_rules.at(cut_cell_index)[k];
        
        for (std::size_t j = 0; j < qr.second.size(); ++j)
        {
          volume += qr.second[j];
          part_volume += qr.second[j];
        }
        
      }
    }
    
    std::cout<<"part volume " << part_volume<<std::endl;
  }
  
  std::cout << "exact volume " << exact_volume<<'\n'
            << "volume " << volume<<std::endl;
  ASSERT_NEAR(exact_volume, volume, 10*DOLFIN_EPS_LARGE);
}
//-----------------------------------------------------------------------------
TEST(MultiMeshes, test_assembly)
{
 // FIXME: Reimplement when functionals are in place again
}

TEST(MultiMeshes, test_assemble_expression)
{
  auto mesh_0 = std::make_shared<RectangleMesh>(Point(0.,0.),
						Point(2., 2.), 16, 16);
  auto mesh_1 = std::make_shared<RectangleMesh>(Point(1., 1.),
						Point(3., 3.), 8, 8);
  auto mesh_2 = std::make_shared<RectangleMesh>(Point(-1., -1.),
						Point(0., 0.), 8, 8);

  // Build multimesh
  auto multimesh = std::make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  multimesh->add(mesh_2);
  multimesh->build();

  // The function v(x)=1
  class MyFunction : public Expression
  {
  public:
    void eval(Array<double>& values, const Array<double>& x) const
    { values[0] = 1; }
  };

  auto v = std::make_shared<MyFunction>();
  MultiMeshAssemble::MultiMeshFunctional M(multimesh);
  M.v=v;

  // Equvialent to computing area of multimesh
  double funcarea = assemble_multimesh(M);

  // Alternative computation of area
  double volume = 0;
  std::vector<double> all_volumes;
  // Sum contribution from all parts
  for (std::size_t part = 0; part < multimesh->num_parts(); part++)
    {
      double part_volume = 0;
      std::vector<double> status(multimesh->part(part)->num_cells(), 0);
      // Uncut cell volume given by function volume
      double uncut_volume = 0;
      const auto uncut_cells = multimesh->uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
	const Cell cell(*multimesh->part(part), *it);
	volume += cell.volume();
	part_volume += cell.volume();
	uncut_volume += cell.volume();
	status[*it] = 1;
      }
      // Cut cell volume given by quadrature rule
      double cut_volume = 0;
      const auto& cut_cells = multimesh->cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
	const auto& qr = multimesh->quadrature_rule_cut_cell(part, *it);
	for (std::size_t i = 0; i < qr.second.size(); ++i)
	{
	  volume += qr.second[i];
	  part_volume += qr.second[i];
	  cut_volume += qr.second[i];
	}
	status[*it] = 2;
      }
      all_volumes.push_back(part_volume);
    }
  ASSERT_NEAR(volume, funcarea, 10*DOLFIN_EPS_LARGE);
}

// Create test for assemble MultiMeshFunction with zero and non-zero inits

// Test all
int MultiMesh_main(int argc, char **argv) {
  // Test not working in parallel
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Skipping unit test in parallel.");
    info("OK");
    return 0;
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
//-----------------------------------------------------------------------------
