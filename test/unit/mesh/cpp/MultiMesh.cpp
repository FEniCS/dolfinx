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
// Last changed: 2014-03-31
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>

#include <dolfin/geometry/SimplexQuadrature.h>

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  //CPPUNIT_TEST(test_multiple_meshes_quadrature);
  CPPUNIT_TEST(test_multiple_meshes_interface_quadrature);
  CPPUNIT_TEST_SUITE_END();

public:

#define Pause {char dummycharXohs5su8='a';std::cout<<"\n Pause: "<<__FILE__<<" line "<<__LINE__<<" function "<<__FUNCTION__<<std::endl;std::cin>>dummycharXohs5su8;}

  std::string drawsimplex(const std::vector<Point>& simplex,
                          const std::string& color = "'b'")
  {
    std::stringstream ss;
    switch(simplex.size())
    {
    case 2:
      ss << "drawline(";
      break;
    case 3:
      ss << "drawtriangle(";
      break;
    case 4:
      ss << "drawtet(";
      break;
    default: Pause;
    }

    for (std::size_t i = 0; i < simplex.size(); ++i)
    {
      ss << "[";
      for (int d = 0; d < 3; ++d)
        ss << simplex[i][d] <<' ';
      ss << "],";
    }
    ss << color<< ");";
    return ss.str();
  }

  std::string drawcell(const Cell &cell,
                       const std::string color = "'b'")
  {
    const std::size_t nnodes = cell.mesh().topology().dim()+1;
    std::vector<Point> simplex(nnodes);
    for (std::size_t i = 0; i < nnodes; ++i)
      simplex[i] = cell.mesh().geometry().point(cell.entities(0)[i]);
    return drawsimplex(simplex, color);
  }

  std::string plot(const Point& p,const std::string m="'.'")
  {
    std::stringstream ss;
    ss<<"plot3("<<p[0]<<','<<p[1]<<','<<p[2]<<','<<m<<");";
    return ss.str();
  }

  std::string drawtriangulation(const std::vector<double> &triangles,
                                const std::size_t gdim)
  {
    std::string str;
    std::vector<Point> tri(3);

    if (gdim == 2)
    {
      for (std::size_t i = 0; i < triangles.size()/6; ++i)
      {
        tri[0] = Point(triangles[6*i], triangles[6*i+1]);
        tri[1] = Point(triangles[6*i+2], triangles[6*i+3]);
        tri[2] = Point(triangles[6*i+4], triangles[6*i+5]);
        str += drawsimplex(tri);
      }
      return str;
    }

    if (gdim == 3)
    {
      for (std::size_t i = 0; i < triangles.size()/9; ++i)
      {
        tri[0]= Point(triangles[9*i], triangles[9*i+1], triangles[9*i+2]);
        tri[1]= Point(triangles[9*i+3], triangles[9*i+4], triangles[9*i+5]);
        tri[2]= Point(triangles[9*i+6], triangles[9*i+7], triangles[9*i+8]);

        str += drawsimplex(tri);
      }
      return str;
    }
  }

  std::string drawtriangulation3D(const std::vector<double> &triangles)
  {
    std::string str;
    std::vector<Point> tri(3);
    for (std::size_t i = 0; i < triangles.size()/6; ++i)
    {
      tri[0] = Point(triangles[6*i], triangles[6*i+1]);
      tri[1] = Point(triangles[6*i+2], triangles[6*i+3]);
      tri[2] = Point(triangles[6*i+4], triangles[6*i+5]);
      str += drawsimplex(tri);
    }
    return str;
  }



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

    const std::size_t gdim = 2;
    const std::size_t tdim = 2;
    UnitSquareMesh mesh_0(31, 17);
    RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 21, 12);
    RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 11, 31);
    RectangleMesh mesh_3(0.8, 0.01, 0.9, 0.99, 3, 55);
    RectangleMesh mesh_4(0.01, 0.01, 0.02, 0.02, 1, 1);

    // const std::size_t gdim = 3;
    // const std::size_t tdim = 3;
    // UnitCubeMesh mesh_0(2, 3, 4);
    // BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,   4, 3, 2);
    // BoxMesh mesh_2(0.2, 0.2, 0.2,    0.8, 0.8, 0.8,   3, 4, 3);
    // BoxMesh mesh_3(0.8, 0.01, 0.01,  0.9, 0.99, 0.99,  4, 2, 3);
    // BoxMesh mesh_4(0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 1, 1, 1);

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

    UnitCubeMesh mesh_0(2, 4, 4);
    //BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,   4, 3, 2);
    BoxMesh mesh_1(-0.1, -0.1, -0.1,    1.1, 1.1, 0.1,   4, 4, 2);


    // Build the multimesh
    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    //multimesh.add(mesh_2);
    //multimesh.add(mesh_3);
    //multimesh.add(mesh_4);
    multimesh.build();

    // Exact volume of the interface is known
    const double exact_volume = 1;//0.8*0.8*6; // for mesh_0 and mesh_1
    double volume = 0;


    // Sum contribution from all parts
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      std::cout << "% part " << part << '\n';

      // Cut cell
      const auto cut_cells = multimesh.cut_cells(part);
      auto quadrature_rule = multimesh.quadrature_rule_cut_cells_interface(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        for (std::size_t i = 0; i < quadrature_rule[*it].first.size(); ++i)
	{
          volume += quadrature_rule[*it].first[i];
          const int gdim = 3;
          Point pt(quadrature_rule[*it].second[i*gdim],
                   quadrature_rule[*it].second[i*gdim+1],
                   quadrature_rule[*it].second[i*gdim+2]);
	  std::cout << "plot3("<<pt[0]<<','<<pt[1]<<','<<pt[2]<<",'.');";
	}
      }
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
