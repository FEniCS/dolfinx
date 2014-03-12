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
// Last changed: 2014-03-12
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  //CPPUNIT_TEST(test_integrate_triangles);
  CPPUNIT_TEST(test_integrate_triangles_three_meshes);
  CPPUNIT_TEST_SUITE_END();

public:

#include <string>
#include <sstream>
#include <fstream>

#define Pause { std::cout<<__FUNCTION__<<' '<<__LINE__; char fdajfds; std::cin>>fdajfds; }

  void dolfin_write_medit_triangles(const std::string &filename,
                                    const Mesh& mesh,
                                    const int t = 0)
  {
    std::stringstream ss;
    ss<<filename<<"."<<t<<".mesh";
    std::ofstream file(ss.str().c_str());
    if (!file.good()) { std::cout << ss.str()<<'\n'; Pause; }
    file.precision(13);
    // write vertices
    const std::size_t nno = mesh.num_vertices();
    file << "MeshVersionFormatted 1\nDimension\n2\nVertices\n"
         << nno<<'\n';
    const std::vector<double>& coords = mesh.coordinates();
    for (std::size_t i = 0; i < nno; ++i)
      file << coords[2*i]<<' '<<coords[2*i+1]<<" 1\n";
    // write connectivity
    const std::size_t nel = mesh.num_cells();
    file << "Triangles\n"
         << nel <<'\n';
    const std::vector<unsigned int>& cells = mesh.cells();
    for (std::size_t e = 0; e < nel; ++e)
      file << cells[3*e]+1<<' '<<cells[3*e+1]+1<<' '<<cells[3*e+2]+1<<" 1\n";
    file.close();
  }

  std::string drawtriangle(const Cell &cell)
  {
    const MeshGeometry& geometry = cell.mesh().geometry();
    const unsigned int* vertices = cell.entities(0);
    const Point a = geometry.point(vertices[0]);
    const Point b = geometry.point(vertices[1]);
    const Point c = geometry.point(vertices[2]);
    //Point av = (a+b+c)/3.;
    //std::cout << av[0]<<' '<<av[1]<<' '<<av[2]<<std::endl;
    std::stringstream ss;
    ss << "drawtriangle("
       << "["<<a[0]<<' '<<a[1]<<"],"
       << "["<<b[0]<<' '<<b[1]<<"],"
       << "["<<c[0]<<' '<<c[1]<<"]);";
    return ss.str();
  }



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
    std::cout<<"\n\n\n"<<__FUNCTION__<<"\n\n";

    set_log_level(PROGRESS);

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


    std::cout << std::abs(a[0]-b[0])*std::abs(a[1]-b[1])<<'\n';

    // For part 0, compute area of uncut, cut and covered cells
    double volume = 0;

    for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
    {
      std::cout<<"\npart "<<part<<"\n\n";


      // Uncut cell volume given by function volume
      std::cout<<"\nuncut\n";
      auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();

        //std::cout << (*it)<<' ';
        //std::cout<<drawtriangle(cell);
        //std::cout<<(*it)<<' '<<drawtriangle(cell)<<'\n';
       }

      // Cut cell volume given by quadrature rule
      std::cout<<"\ncut\n";
      auto cut_cells = multimesh.cut_cells(part);
      auto qr = multimesh.quadrature_rule_cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();

        //std::cout << (*it)<<' ';
        //std::cout<<drawtriangle(cell);
        std::cout<<(*it)<<' '<<drawtriangle(cell)<<'\n';


        // Loop over weights
        for (std::size_t i = 0; i < qr[*it].first.size(); ++i)
          volume += qr[*it].first[i];
      }

      //auto covered_cells = multimesh.covered_cells(part);


    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }

};





int main()
{
  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshes);
  DOLFIN_TEST;
}

