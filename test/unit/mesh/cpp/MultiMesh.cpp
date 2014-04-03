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
// Last changed: 2014-04-03
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>

#include <dolfin/geometry/SimplexQuadrature.h>

#include </home/august/Projects/fenicsBB/dolfin/dolfin/mesh/plotstuff.h>


using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  //CPPUNIT_TEST(test_multiple_meshes_quadrature);
  CPPUNIT_TEST(test_multiple_meshes_interface_quadrature);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_multiple_meshes_quadrature()
  {

    // Create multimesh from three triangle meshes of the unit square

    // const std::size_t gdim = 2;
    // const std::size_t tdim = 2;
    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 2, 2);
    // RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 1, 1);
    // RectangleMesh mesh_3(0.8, 0.01, 0.9, 0.99, 3, 55);
    // RectangleMesh mesh_4(0.01, 0.01, 0.02, 0.02, 1, 1);

    // const std::size_t gdim = 2;
    // const std::size_t tdim = 2;
    // UnitSquareMesh mesh_0(31, 17);
    // RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 21, 12);
    // RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 11, 31);
    // RectangleMesh mesh_3(0.8, 0.01, 0.9, 0.99, 3, 55);
    // RectangleMesh mesh_4(0.01, 0.01, 0.02, 0.02, 1, 1);

    const std::size_t gdim = 3;
    const std::size_t tdim = 3;
    UnitCubeMesh mesh_0(2, 3, 4);
    BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,   4, 3, 2);
    BoxMesh mesh_2(0.2, 0.2, 0.2,    0.8, 0.8, 0.8,   3, 4, 3);
    BoxMesh mesh_3(0.8, 0.01, 0.01,  0.9, 0.99, 0.99,  4, 2, 3);
    BoxMesh mesh_4(0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 1, 1, 1);

    // Build the multimesh
    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.add(mesh_2);
    multimesh.add(mesh_3);
    multimesh.add(mesh_4);
    multimesh.build();

    // Exact volume is known
    const double exact_volume = 1;
    double volume = 0;

    // Sum contribution from all parts
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      //std::cout << "% part " << part << '\n';

      // Uncut cell volume given by function volume
      const auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();
      }

      // Cut cell volume given by quadrature rule
      const auto cut_cells = multimesh.cut_cells(part);
      auto qr = multimesh.quadrature_rule_cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
          volume += qr[*it].first[i];
      }
    }

    std::cout << "exact volume " << exact_volume<<'\n'
              << "volume " << volume<<std::endl;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }


  void test_multiple_meshes_interface_quadrature()
  {
    // const std::size_t gdim = 2;
    // const std::size_t tdim = 2;
    // UnitSquareMesh mesh_0(31, 17);
    // RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 21, 12);
    // RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 11, 31);
    // RectangleMesh mesh_3(0.8, 0.01, 0.9, 0.99, 3, 55);
    // RectangleMesh mesh_4(0.01, 0.01, 0.02, 0.02, 1, 1);

    // UnitCubeMesh mesh_0(1, 2, 3);
    // BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,   2,3,4);//2, 3, 4);
    // BoxMesh mesh_2(-0.1, -0.1, -0.1,    0.7, 0.7, 0.7,   4, 3, 2);
    // BoxMesh mesh_3(0.51, 0.51, 0.51,    0.7, 0.7, 0.7,   1,1,1);//4, 3, 2);
    // BoxMesh mesh_4(0.3, 0.3, 0.3,    0.7, 0.7, 0.7,   1,1,1);

    // double exact_volume = 0.8*0.8*6; // for mesh_0 and mesh_1
    // exact_volume += 0.4*0.4*6; // for mesh_1 and mesh_4
    // double volume = 0;


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

    // double exact_volume = 0.8*0.8*6; // for mesh_0 and mesh_1
    // exact_volume += 0.4*0.4*6; // for mesh_1 and mesh_4
    // double volume = 0;


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


    MeshEditor editor;
    Mesh mesh_0;
    editor.open(mesh_0, 2, 2);
    editor.init_vertices(3);
    editor.init_cells(1);
    editor.add_vertex(0, Point(0.,0.));
    editor.add_vertex(1, Point(2.,0.));
    editor.add_vertex(2, Point(1.,2.));
    editor.add_cell(0, 0,1,2);
    editor.close();

    Mesh mesh_1;
    editor.open(mesh_1, 2, 2);
    editor.init_vertices(3);
    editor.init_cells(1);
    editor.add_vertex(0, Point(1.5,-2.));
    editor.add_vertex(1, Point(4.,0.));
    editor.add_vertex(2, Point(1.5,2));
    editor.add_cell(0, 0,1,2);
    editor.close();

    Mesh mesh_2;
    editor.open(mesh_2, 2, 2);
    editor.init_vertices(3);
    editor.init_cells(1);
    editor.add_vertex(0, Point(3.,0.5));
    editor.add_vertex(1, Point(-1.,0.5));
    editor.add_vertex(2, Point(1.,-1.5));
    editor.add_cell(0, 0,1,2);
    editor.close();

    double exact_volume = (1.5-0.25) + (1-0.5); // mesh_0, mesh_1 and mesh_2
    exact_volume += 1.5 + std::sqrt(1.5*1.5 + 1.5*1.5); // mesh_1 and mesh_2

    File("mesh_0.xml") << mesh_0;
    File("mesh_1.xml") << mesh_1;

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
      double partvolume = 0;

      // Cut cell
      const std::size_t gdim = mesh_0.geometry().dim();
      const auto& cut_cells = multimesh.cut_cells(part);
      auto quadrature_rule = multimesh.quadrature_rule_cut_cells_interface(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        std::cout << "% cut cell " << (*it)<<'\n';
        for (std::size_t i = 0; i < quadrature_rule[*it].first.size(); ++i)
	{
          volume += quadrature_rule[*it].first[i];
          partvolume += quadrature_rule[*it].first[i];
          //std::cout << drawqr(quadrature_rule[*it]);
          for (std::size_t d = 0; d < gdim; ++d)
            std::cout << quadrature_rule[*it].second[i*gdim+d]<<' ';
          std::cout << "    "<<quadrature_rule[*it].first[i]<<'\n';
	}
      }

      std::cout<<"part volume " << partvolume<<std::endl;
    }

    std::cout << "exact volume " << exact_volume<<'\n'
              << "volume " << volume<<std::endl;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }


};

int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshes);
  DOLFIN_TEST;
}
