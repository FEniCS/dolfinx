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
#include <dolfin/common/unittest.h>
#include <dolfin/geometry/SimplexQuadrature.h>

#include "MultiMeshStokes2D.h"

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  //CPPUNIT_TEST(test_multiple_meshes_quadrature);
  CPPUNIT_TEST(test_multiple_meshes_interface_quadrature);
  CPPUNIT_TEST(test_assembly);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_multiple_meshes_quadrature()
  {
    set_log_level(DBG);

    // Create multimesh from three triangle meshes of the unit square

    // Many meshes, but not more than three overlap => this works
    UnitCubeMesh mesh_0(11, 12, 13);
    BoxMesh mesh_1(Point(0.1, 0.1, 0.1),    Point(0.9, 0.9, 0.9),    13, 11, 12);
    BoxMesh mesh_2(Point(0.2, 0.2, 0.2),    Point(0.95, 0.95, 0.8),  11, 13, 11);
    BoxMesh mesh_3(Point(0.94, 0.01, 0.01), Point(0.98, 0.99, 0.99), 1, 11, 11);
    BoxMesh mesh_4(Point(0.01, 0.01, 0.01), Point(0.02, 0.02, 0.02), 1, 1, 1);

    // // Completely nested 2D: can't do no more than three meshes
    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(Point(0.1, 0.1), Point(0.9, 0.9, 1, 1);
    // RectangleMesh mesh_2(Point(0.2, 0.2), Point(0.8, 0.8, 1, 1);
    // RectangleMesh mesh_3(Point(0.3, 0.3), Point(0.7, 0.7, 1, 1);
    // RectangleMesh mesh_4(Point(0.4, 0.4), Point(0.6, 0.6, 1, 1);

    // // Completely nested 3D: can't do no more than three meshes
    // UnitCubeMesh mesh_0(2, 3, 4);
    // BoxMesh mesh_1(Point(0.1, 0.1, 0.1),    Point(0.9, 0.9, 0.9),    4, 3, 2);
    // BoxMesh mesh_2(Point(0.2, 0.2, 0.2),    Point(0.8, 0.8, 0.8),    3, 4, 3);
    // BoxMesh mesh_3(Point(0.8, 0.01, 0.01),  Point(0.9, 0.99, 0.99),  4, 2, 3);
    // BoxMesh mesh_4(Point(0.01, 0.01, 0.01), Point(0.02, 0.02, 0.02), 1, 1, 1);

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
    std::cout << "Sum contributions\n";
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      std::cout << "% part " << part;
      double part_volume = 0;

      // Uncut cell volume given by function volume
      const auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();
        part_volume += cell.volume();
      }

      std::cout << "\t uncut volume "<< part_volume<<' ';

      // Cut cell volume given by quadrature rule
      const auto& cut_cells = multimesh.cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        const auto& qr = multimesh.quadrature_rule_cut_cell(part, *it);
        for (std::size_t i = 0; i < qr.second.size(); ++i)
        {
          volume += qr.second[i];
          part_volume += qr.second[i];
        }
      }
      std::cout << "\ttotal volume " << part_volume<< std::endl;
    }

    std::cout<<std::setprecision(13) << "exact volume " << exact_volume<<'\n'
              << "volume " << volume<<std::endl;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }

  void test_multiple_meshes_interface_quadrature()
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

    UnitCubeMesh mesh_0(1, 1, 1);
    BoxMesh mesh_1(Point(0.1, 0.1, 0.1), Point(0.9, 0.9, 0.9), 1, 1, 1);
    BoxMesh mesh_2(Point(0.2, 0.2, 0.2), Point(0.8, 0.8, 0.8), 1, 1, 1);
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

    File("mesh_0.xml") << mesh_0;
    File("mesh_1.xml") << mesh_1;
    File("mesh_2.xml") << mesh_2;

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
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }

  void test_assembly()
  {
    // Set some parameters
    parameters["reorder_dofs_serial"] = false;

    // Some parameters
    const std::size_t N = 4;
    const std::size_t n = 2;
    const double b1 = 1.0;
    const double b2 = 1.0;

    // Right-hand side
    class Source : public Expression
    {
    public:
      Source() : Expression(2) {}

      void eval(Array<double>& values, const Array<double>& x) const
      {
        values[0] = 2*DOLFIN_PI*sin(2*DOLFIN_PI*x[1])*
          (cos(2*DOLFIN_PI*x[0]) -
           2*DOLFIN_PI*DOLFIN_PI*cos(2*DOLFIN_PI*x[0]) +
           DOLFIN_PI*DOLFIN_PI);
        values[1] = 2*DOLFIN_PI*sin(2*DOLFIN_PI*x[0])*
          (cos(2*DOLFIN_PI*x[1]) +
           2*DOLFIN_PI*DOLFIN_PI*cos(2*DOLFIN_PI*x[1]) -
           DOLFIN_PI*DOLFIN_PI);
      }
    } source;

    // Subdomain for no-slip boundary
    class DirichletBoundary : public SubDomain
    {
      bool inside(const Array<double>& x, bool on_boundary) const
      {
        return on_boundary and
          (near(x[0], 0) || near(x[0], 1) ||
           near(x[1], 0) || near(x[1], 1));
      }
    } dirichlet_boundary;

    // Create meshes
    UnitSquareMesh mesh_0(N, N);
    const double c = 0.123123;
    RectangleMesh mesh_1(Point(0.5 - c, 0.5 - c), Point(0.5 + c, 0.5 + c), n, n);
    mesh_1.rotate(37, 2);

    // Create function spaces
    MultiMeshStokes2D::FunctionSpace W0(mesh_0);
    MultiMeshStokes2D::FunctionSpace W1(mesh_1);

    // Create forms
    MultiMeshStokes2D::BilinearForm a0(W0, W0);
    MultiMeshStokes2D::BilinearForm a1(W1, W1);
    MultiMeshStokes2D::LinearForm L0(W0);
    MultiMeshStokes2D::LinearForm L1(W1);
    MultiMeshStokes2D::Functional M0(mesh_0);
    MultiMeshStokes2D::Functional M1(mesh_1);

    // Build multimesh function space
    MultiMeshFunctionSpace W;
    W.parameters("multimesh")["quadrature_order"] = 3;
    W.add(W0);
    W.add(W1);
    W.build();

    // Create constants
    Constant beta_1(b1);
    Constant beta_2(b2);

    // Create solution function
    MultiMeshFunction w(W);

    // Set coefficients
    a0.w0 = beta_1;
    a1.w0 = beta_1;
    a0.w1 = beta_2;
    a1.w1 = beta_2;
    L0.w0 = source;
    L1.w0 = source;
    L0.w1 = beta_2;
    L1.w1 = beta_2;
    M0.w0 = *w.part(0);
    M1.w0 = *w.part(1);

    // Build multimesh forms
    MultiMeshForm a(W, W);
    MultiMeshForm L(W);
    MultiMeshForm M(W);
    a.add(a0);
    a.add(a1);
    L.add(L0);
    L.add(L1);
    M.add(M0);
    M.add(M1);
    a.build();
    L.build();
    M.build();

    // Create subspaces for boundary conditions
    MultiMeshSubSpace V(W, 0);
    MultiMeshSubSpace Q(W, 1);

    // Create boundary condition
    Constant zero(0, 0);
    MultiMeshDirichletBC bc(V, zero, dirichlet_boundary);

    // Assemble system matrix and right-hand side
    Matrix A;
    Vector b;
    MultiMeshAssembler assembler;
    assembler.assemble(A, a);
    assembler.assemble(b, L);

    // Apply boundary condition
    bc.apply(A, b);

    // Compute solutipon
    solve(A, *w.vector(), b);

    // Compute squared L2 norm of solution
    Scalar m;
    assembler.assemble(m, M);

    // Check value
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.217133856286212, m.get_scalar_value(), 1e-12);
  }

};

int main()
{
  // Test not working in parallel
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Skipping unit test in parallel.");
    info("OK");
    return 0;
  }

  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshes);
  DOLFIN_TEST;
}
