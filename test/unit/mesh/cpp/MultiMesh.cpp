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
// Last changed: 2014-03-10
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  CPPUNIT_TEST(test_integrate_triangles);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_integrate_triangles()
  {
    // Create MultiMesh from two triangle meshes of the unit square
    UnitSquareMesh mesh_0(3, 3), mesh_1(4, 4);

    // Translate some random distance
    Point pt(0.632350, 0.278498);
    //Point pt(0.50001, 0);
    //Point pt(0.1,0.1);
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
      std::cout<<"\n\n part "<<part<<"\n\n";

      // Uncut cell volume given by function volume
      auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();

        const MeshGeometry& geometry = cell.mesh().geometry();
        const unsigned int* vertices = cell.entities(0);
        const Point a = geometry.point(vertices[0]);
        const Point b = geometry.point(vertices[1]);
        const Point c = geometry.point(vertices[2]);
        Point av = (a+b+c)/3.;
        //std::cout << av[0]<<' '<<av[1]<<' '<<av[2]<<std::endl;
        std::cout << "drawtriangle("
                  << "["<<a[0]<<' '<<a[1]<<"],"
                  << "["<<b[0]<<' '<<b[1]<<"],"
                  << "["<<c[0]<<' '<<c[1]<<"]);\n";
      }

      // Cut cell volume given by quadrature rule
      auto cut_cells = multimesh.cut_cells(part);
      auto qr = multimesh.quadrature_rule_cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        const MeshGeometry& geometry = cell.mesh().geometry();
        const unsigned int* vertices = cell.entities(0);
        const Point a = geometry.point(vertices[0]);
        const Point b = geometry.point(vertices[1]);
        const Point c = geometry.point(vertices[2]);
        Point av = (a+b+c)/3.;
        std::cout << av[0]<<' '<<av[1]<<' '<<av[2]<<std::endl;

        volume += cell.volume();

        // Loop over weights
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
        {
          volume += qr[*it].first[i];
          //std::cout<<qr[*it].first[i]<<' ';
        }
      }

      // // Other part's cut cells (none)
      // {
      //   const int part = 1;
      //   auto cut_cells = multimesh.cut_cells(part);
      //   auto qr = multimesh.quadrature_rule_cut_cells(part);
      //   for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      //   {
      //     // const Cell cell(*multimesh.part(part), *it);
      //     // const MeshGeometry& geometry = cell.mesh().geometry();
      //     // const unsigned int* vertices = cell.entities(0);
      //     // const Point a = geometry.point(vertices[0]);
      //     // const Point b = geometry.point(vertices[1]);
      //     // const Point c = geometry.point(vertices[2]);
      //     // Point av = (a+b+c)/3.;
      //     // std::cout << av[0]<<' '<<av[1]<<' '<<av[2]<<std::endl;

      //     // Loop over weights
      //     for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
      //       volume += qr[*it].first[i];
      //   }
      // }




      // // Covered cells
      // std::cout<<"\n\n covered\n\n";
      // auto covered_cells = multimesh.covered_cells(part);
      // for (auto it = covered_cells.begin(); it != covered_cells.end(); ++it)
      // {
      //   const Cell cell(*multimesh.part(part), *it);
      //   volume += cell.volume();


      //   const MeshGeometry& geometry = cell.mesh().geometry();
      //   const unsigned int* vertices = cell.entities(0);
      //   const Point a = geometry.point(vertices[0]);
      //   const Point b = geometry.point(vertices[1]);
      //   const Point c = geometry.point(vertices[2]);
      //   Point av = (a+b+c)/3.;
      //   //std::cout << av[0]<<' '<<av[1]<<' '<<av[2]<<std::endl;
      //   std::cout << "drawtriangle("
      //             << "["<<a[0]<<' '<<a[1]<<"],"
      //             << "["<<b[0]<<' '<<b[1]<<"],"
      //             << "["<<c[0]<<' '<<c[1]<<"]);\n";

      // }

    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }
};



int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshes);
  DOLFIN_TEST;
}



// class MeshFunctions : public CppUnit::TestFixture
// {
//   CPPUNIT_TEST_SUITE(MeshFunctions);
//   CPPUNIT_TEST(test_create_from_domains);
//   CPPUNIT_TEST_SUITE_END();

// public:

//   void test_create_from_domains()
//   {
//     // Create mesh
//     std::shared_ptr<Mesh> mesh(new UnitSquareMesh(3, 3));
//     dolfin_assert(mesh);

//     const std::size_t D = mesh->topology().dim();

//     // Test setting all values
//     for (std::size_t d = 0; d <= D; ++d)
//     {
//       // Create MeshDomains object
//       MeshDomains mesh_domains;
//       mesh_domains.init(D);

//       mesh->init(d);

//       // Build mesh domain
//       std::map<std::size_t, std::size_t>& domain = mesh_domains.markers(d);
//       for (std::size_t i = 0; i < mesh->num_entities(d); ++i)
//         domain.insert(std::make_pair(i, i));

//       // Create MeshFunction and test values
//       MeshFunction<std::size_t> mf(mesh, d, mesh_domains);
//       for (std::size_t i = 0; i < mf.size(); ++i)
//         CPPUNIT_ASSERT(mf[i] == i);
//     }

//     // Test setting some values only
//     for (std::size_t d = 0; d <= D; ++d)
//     {
//       // Create MeshDomains object
//       MeshDomains mesh_domains;
//       mesh_domains.init(D);

//       mesh->init(d);

//       // Build mesh domain
//       std::map<std::size_t, std::size_t>& domain = mesh_domains.markers(d);
//       const std::size_t num_entities = mesh->num_entities(d);
//       for (std::size_t i = num_entities/2; i < num_entities; ++i)
//         domain.insert(std::make_pair(i, i));

//       // Create MeshFunction and test values
//       MeshFunction<std::size_t> mf(mesh, d, mesh_domains);
//       for (std::size_t i = 0; i < num_entities/2; ++i)
//         CPPUNIT_ASSERT(mf[i] == std::numeric_limits<std::size_t>::max());
//       for (std::size_t i = num_entities/2; i < mf.size(); ++i)
//         CPPUNIT_ASSERT(mf[i] == i);
//     }


//   }

// };


// int main()
// {
//   CPPUNIT_TEST_SUITE_REGISTRATION(MeshFunctions);
//   DOLFIN_TEST;
// }
