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
// Last changed: 2014-03-13
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  CPPUNIT_TEST(test_integrate_triangles);
  CPPUNIT_TEST(test_integrate_triangles_three_meshes);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_integrate_triangles()
  {
    // Create MultiMesh from two triangle meshes of the unit square
    UnitSquareMesh mesh_0(3, 3), mesh_1(4, 4);

    // Translate some random distance
    Point pt(0.632350, 0.278498);
    mesh_1.translate(pt);

    // Exact volume is known
    const double exact_volume = 2 - (1-pt[0]) * (1-pt[1]);

    // Build the multimesh
    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.build();

    // For part 0, compute area of uncut, cut and covered cells
    double volume = 0;

    for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
    {
      // Uncut cell volume given by function volume
      auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();
      }

      // Cut cell volume given by quadrature rule
      auto cut_cells = multimesh.cut_cells(part);
      auto qr = multimesh.quadrature_rule_cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();

        // Loop over weights
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
          volume += qr[*it].first[i];
      }
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }



  void test_integrate_triangles_three_meshes()
  {
    // Create MultiMesh from three triangle meshes of the unit square
    UnitSquareMesh mesh_0(3, 3), mesh_1(4, 4), mesh_2(5, 5);

    // Translate some random distance
    const Point a(0.1, 0.1);
    const Point b(0.9, 0.9);
    mesh_1.translate(a);
    mesh_2.translate(b);

    // Exact volume is known
    const double exact_volume = 2.15;

    // Build the multimesh
    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.add(mesh_2);
    multimesh.build();

    // For part 0, compute area of uncut, cut and covered cells
    double volume = 0;

    for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
    {
      // Uncut cell volume given by function volume
      auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();
     }

      // Cut cell volume given by quadrature rule
      auto cut_cells = multimesh.cut_cells(part);
      auto qr = multimesh.quadrature_rule_cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();

        // Loop over weights
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
          volume += qr[*it].first[i];
      }
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }

};





int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshes);
  DOLFIN_TEST;
}

