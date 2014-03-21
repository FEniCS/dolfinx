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
// Last changed: 2014-03-21
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>

#include </home/august/Projects/fenicsBB/dolfin/dolfin/geometry/IntersectionTriangulation.h>
#include </home/august/Projects/fenicsBB/dolfin/dolfin/geometry/SimplexQuadrature.h>

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  CPPUNIT_TEST(test_multiple_meshes);
  //CPPUNIT_TEST(test_integrate_triangles);
  // CPPUNIT_TEST(test_integrate_triangles_three_meshes);
  // CPPUNIT_TEST(test_integrate_covered_meshes);
  CPPUNIT_TEST_SUITE_END();

public:

  std::string drawtriangle(const std::vector<Point> &tri)
  {
    std::stringstream ss;
    ss << "drawtriangle("
       << "["<<tri[0][0]<<' '<<tri[0][1]<<"],"
       << "["<<tri[1][0]<<' '<<tri[1][1]<<"],"
       << "["<<tri[2][0]<<' '<<tri[2][1]<<"]);";
    return ss.str();
  }

  std::string drawtriangle(const Cell &cell)
  {
    std::vector<Point> tri(3);
    for (int i = 0; i < 3; ++i)
      tri[i] = cell.mesh().geometry().point(cell.entities(0)[i]);
    return drawtriangle(tri);
  }

  std::string plot(const Point& p,const std::string m="'.'")
  {
    std::stringstream ss;
    ss<<"plot("<<p[0]<<','<<p[1]<<','<<m<<");";
    return ss.str();
  }

  std::string drawtriangulation(const std::vector<double> &triangles)
  {
    std::string str;
    std::vector<Point> tri(3);
    for (std::size_t i = 0; i < triangles.size()/6; ++i)
    {
      tri[0] = Point(triangles[6*i], triangles[6*i+1]);
      tri[1] = Point(triangles[6*i+2], triangles[6*i+3]);
      tri[2] = Point(triangles[6*i+4], triangles[6*i+5]);
      str += drawtriangle(tri);
    }
    return str;
  }

#define Pause {char dummycharXohs5su8='a';std::cout<<"\n Pause: "<<__FILE__<<" line "<<__LINE__<<" function "<<__FUNCTION__<<std::endl;std::cin>>dummycharXohs5su8;}





  void add_triangles(const std::vector<double> &a,
                     std::vector<double> &b)
  {
    // Add the triangles in a to the ones in b
    b.insert(b.end(), a.begin(), a.end());
  }

  double add_qr(const std::vector<double> &triangles,
                double factor,
                std::pair<std::vector<double>, std::vector<double> > &qr)
  {
    if (triangles.size())
    {
      // Get quadrature rules of flat triangles
      const std::size_t gdim = 2;
      const std::size_t tdim = 2;
      const std::size_t order = 1;

      double sum = 0;
      for (std::size_t tri = 0; tri < triangles.size()/6; ++tri)
      {
        auto local_qr =
          SimplexQuadrature::compute_quadrature_rule(&triangles[0]+6*tri,
                                                     tdim,gdim,order);

        // Add the quadrature rules in a with modified weight factor to
        // the ones in b
        const std::size_t num_points = local_qr.first.size();
        for (std::size_t i = 0; i < num_points; i++)
        {
          qr.first.push_back(factor*local_qr.first[i]);
          sum += factor*local_qr.first[i];
          for (std::size_t j = 0; j < gdim; j++)
            qr.second.push_back(local_qr.second[i*gdim + j]);
        }
      }

      return sum;
    }

    return 0;
  }

  double
  qrsum(const std::pair<std::vector<double>,std::vector<double> > &qr)
  {
    double sum = 0;
    for (std::size_t i = 0; i < qr.first.size(); ++i)
      sum += qr.first[i];
    return sum;
  }


  std::vector<double>
  triangulate_intersection(const Cell& cell,
                           const std::vector<double> &triangles)
  {
    std::vector<double> net_triangulation;

    std::vector<Point> tri_cell(3), tri(3);
    const MeshGeometry& geometry = cell.mesh().geometry();
    const unsigned int* vertices = cell.entities(0);

    // dimension 2
    // three nodes per triangle

    for (std::size_t i = 0; i < triangles.size()/6; ++i)
    {
      for (std::size_t j = 0; j < 3; ++j)
        tri_cell[j] = geometry.point(vertices[j]);

      tri[0] = Point(triangles[6*i], triangles[6*i+1]);
      tri[1] = Point(triangles[6*i+2], triangles[6*i+3]);
      tri[2] = Point(triangles[6*i+4], triangles[6*i+5]);

      // Compute intersection triangulation
      auto local_tris = IntersectionTriangulation::triangulate_intersection_triangle_triangle(tri_cell, tri);

      // Add these to the net triangulation
      add_triangles(local_tris, net_triangulation);
    }

    // if (net_triangulation.size()) {
    //   std::cout << drawtriangulation(net_triangulation);
    //   Pause;
    // }

    return net_triangulation;
  }

  void test_multiple_meshes()
  {
    // Create multimesh from three triangle meshes of the unit square
    UnitSquareMesh mesh_0(31, 17);
    RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 21, 12);
    RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 11, 31);
    RectangleMesh mesh_3(0.8, 0.01, 0.9, 0.99, 3, 55);

    // Exact volume is known
    const double exact_volume = 1;

    // Build the multimesh
    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.add(mesh_2);
    multimesh.add(mesh_3);
    multimesh.build();

    // Take a cut cell
    // Loop over all cutting cells
    // compute tri_cut_cut = intersection_triangulation{cut, cutting}
    // compute tri_prev_cut = intersection_triangulation{prev_triangulation, cutting}
    // new_triangulation = prev_triangulation + tri_cut_cut + tri_prev_cut
    // new_qr weights are new_qr = prev_qr + qr{cut, cutting} - qr{prev_triangulation, cutting}
    // update: prev_qr = new_qr and prev_triangulation = new_triangulation




    double cut_volume = 0;

    for (std::size_t cut_part = 0; cut_part < multimesh.num_parts(); cut_part++)
    {
      std::cout << "% part " << cut_part << '\n';

      // Iterate over cut cells for current part
      const auto cmap = multimesh.collision_map_cut_cells(cut_part);
      for (auto it = cmap.begin(); it != cmap.end(); ++it)
      {
        // Get cut cell
        const unsigned int cut_cell_index = it->first;
        const Cell cut_cell(*multimesh.part(cut_part), cut_cell_index);

        std::cout << drawtriangle(cut_cell)<<std::endl;

        // Data structures for storing triangulations
        std::vector<double> net_triangulation;

        // Data structures for storing quadrature rules
        std::pair<std::vector<double>, std::vector<double> > net_qr;

        // Loop over cutting cells
        const auto cutting_cells = it->second;
        for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
        {
          // Get cutting part and cutting cell
          const std::size_t cutting_part = jt->first;
          const Cell cutting_cell(*multimesh.part(cutting_part), jt->second);
          std::cout<<drawtriangle(cutting_cell)<<' ';


          // Compute triangulation of intersection of cut and cutting cells
          auto intersection_cut_cutting = cut_cell.triangulate_intersection(cutting_cell);

          // Compute triangulation of intersection of cutting cell and
          // the previous triangulation
          auto intersection_cutting_old = triangulate_intersection(cutting_cell, net_triangulation);

          // Add new triangulations to previous to form new_triangulation
          add_triangles(intersection_cut_cutting, net_triangulation);
          add_triangles(intersection_cutting_old, net_triangulation);

          // Add qr with modified weights
          double vcc = add_qr(intersection_cut_cutting, -1., net_qr);
          double vco = add_qr(intersection_cutting_old, 1., net_qr);

          std::cout <<" % "<< vcc<<' '<<vco<<' '<<qrsum(net_qr)<<'\n';
        }

        double localvol = cut_cell.volume();
        cut_volume += cut_cell.volume();
        for (std::size_t i = 0; i < net_qr.first.size(); ++i)
        {
          localvol += net_qr.first[i];
          cut_volume += net_qr.first[i];
        }

        std::cout<<"\n% local volume "<<localvol<< std::endl;
        std::cout << "% qr sum " << qrsum(net_qr)<<std::endl;
        //Pause;
      }
    }

    std::cout << "\ncut volume " << cut_volume << std::endl;



    // Sum contributions from all parts
    double uncut_volume = 0;
    for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
    {
      // Uncut cell volume given by function volume
      auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        uncut_volume += cell.volume();
      }
    }
    std::cout << "uncut volume " << uncut_volume<<std::endl;

    double total_volume = cut_volume + uncut_volume;
    std::cout << "total volume " << total_volume << std::endl;

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, total_volume, DOLFIN_EPS_LARGE);
  }

  void test_integrate_triangles()
  {
    // Create multimesh from two triangle meshes of the unit square
    UnitSquareMesh mesh_0(3, 3);
    UnitSquareMesh mesh_1(4, 4);

    // Translate some random distance
    Point point(0.632350, 0.278498);
    mesh_1.translate(point);

    // Exact volume is known
    const double exact_volume = 2 - (1 - point[0]) * (1 - point[1]);

    // Build the multimesh
    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.build();

    // Sum contributions from all parts
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
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
          volume += qr[*it].first[i];
      }
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }

  void test_integrate_triangles_three_meshes()
  {
    // Create multimesh from three triangle meshes of the unit square
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

    // Sum contributions from all parts
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
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
          volume += qr[*it].first[i];
      }
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }

  void test_integrate_covered_meshes()
  {
    // Create multimesh from three triangle meshes of the unit square
    UnitSquareMesh mesh_0(1, 1);
    RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 1, 1);
    RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 1, 1);

    // Exact volume is known
    const double exact_volume = 1;

    // Build the multimesh
    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.add(mesh_2); // works if this line is commented out!
    multimesh.build();

    // Sum contributions from all parts
    double volume = 0;
    double v_uncut = 0;
    for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
    {
      std::cout<<"part "<<part<<'\n';

      cout << "uncut:";

      // Uncut cell volume given by function volume
      auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        cout << " " << *it;
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();
        v_uncut += cell.volume();
      }
      cout << " V = " << v_uncut << endl;

      cout << "cut:";

      // Cut cell volume given by quadrature rule
      auto cut_cells = multimesh.cut_cells(part);
      auto qr = multimesh.quadrature_rule_cut_cells(part);
      double v_cut = 0;
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        cout << " " << *it;
        //const Cell cell(*multimesh.part(part), *it);
        //volume += cell.volume();
        //std::cout<<drawtriangle(cell);

        // Loop over weights
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
        {
          volume += qr[*it].first[i];
          v_cut += qr[*it].first[i];
        }
      }
      cout << " V = " << v_cut << endl << endl;


    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }

};

int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshes);
  DOLFIN_TEST;
}
