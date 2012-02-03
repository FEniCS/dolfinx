// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-02-02
// Last changed:
//
// This program generates a mesh for a polygonal domain that is
// represented by a list of its vertices.

#include <vector>
#include <dolfin.h>

using namespace dolfin;

#ifdef HAS_CGAL

int main()
{
  // Create empty mesh
  Mesh mesh;

  // Polygonal domain vertices
  std::vector<Point> domain_vertices;
  domain_vertices.push_back(Point(0.0, 0.0));
  domain_vertices.push_back(Point(1.0, 0.0));
  domain_vertices.push_back(Point(1.5, 1.5));
  domain_vertices.push_back(Point(1.0, 2.0));
  domain_vertices.push_back(Point(0.0, 1.0));
  domain_vertices.push_back(domain_vertices[0]);

  // Generate 2D mesh and plot
  PolygonalMeshGenerator::generate(mesh, domain_vertices, 0.025);
  plot(mesh);


  // Polyhedron face vertices
  std::vector<Point> face_vertices;
  face_vertices.push_back(Point(0.0, 0.0, 0.0));
  face_vertices.push_back(Point(0.0, 0.0, 1.0));
  face_vertices.push_back(Point(0.0, 1.0, 0.0));
  face_vertices.push_back(Point(1.0, 0.0, 0.0));

  // Polyhedron faces (must be triangular) for a tetrahedron
  std::vector<std::vector<unsigned int> > faces(4, std::vector<unsigned int>(3));
  faces[0][0] = 3;
  faces[0][1] = 2;
  faces[0][2] = 1;

  faces[1][0] = 0;
  faces[1][1] = 3;
  faces[1][2] = 1;

  faces[2][0] = 0;
  faces[2][1] = 2;
  faces[2][2] = 3;

  faces[3][0] = 0;
  faces[3][1] = 1;
  faces[3][2] = 2;

  // Generate 3D mesh and plot
  PolyhedralMeshGenerator::generate(mesh, face_vertices, faces, 0.05);
  plot(mesh);

  // Generate 3D mesh from OFF file input (distorted cube)
  PolyhedralMeshGenerator::generate(mesh, "cube.off", 0.2);
  plot(mesh);
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
}

#endif
