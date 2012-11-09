// Copyright (C) 2012 Benjamin Kehlet
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
// First added:  2012-10-31
// Last changed: 2012-11-07

#include "SurfaceFileReader.h"
#include "self_intersect.h"
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/Point.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Min_sphere_of_spheres_d_traits_3.h>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

using namespace dolfin;

static inline double strToDouble(const std::string& s, bool print=false)
{
  std::istringstream is(s);
  double val;
  is >> val;

  if (print)
    cout << "to_double " << s << " : " << val << endl;

  return val;
}

dolfin::LogStream& operator << (dolfin::LogStream& stream, const boost::tuple<double, double, double>& obj)
{
  stream << obj.get<0>() << " " << obj.get<1>() << " " << obj.get<2>();
  return stream;
}

template <class HDS>
class BuildFromSTL : public CGAL::Modifier_base<HDS> {
public:
  BuildFromSTL(std::string filename) : filename(filename){}
  void operator()( HDS& hds) 
  {
    cout << "Reading surface from " << filename << endl;

    CGAL::Polyhedron_incremental_builder_3<HDS> builder( hds, true);
    builder.begin_surface(100000, 100000);


    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

    std::ifstream file(filename.c_str());
    if (!file.is_open())
    {
      dolfin_error("SurfaceFileReader.cpp",
                   "open .stl file to read 3D surface",
                   "Failed to open file");
    }

    std::size_t num_vertices = 0;
    std::map<boost::tuple<double, double, double>, std::size_t> vertex_map;
    std::vector<std::vector<std::size_t> > facets;
    std::string line;
    const boost::char_separator<char> sep(" ");

    // Read the first line and trim away whitespaces
    std::getline(file, line);
    boost::algorithm::trim(line);
  
    if (line.substr(0, 5) != "solid")
      dolfin_error("SurfaceFileReader.cpp",
                   "open .stl file to read 3D surface",
                   "File does not start with \"solid\"");

    // TODO: Read name of solid

    std::getline(file, line);
    boost::algorithm::trim(line);
    
    while (file.good())
    {

      //bool has_normal = false;
      //Point normal;

      // Read the line "facet normal n1 n2 n3"    
      {
        tokenizer tokens(line, sep);
        tokenizer::iterator tok_iter = tokens.begin();

        if (*tok_iter != "facet")
          dolfin_error("SurfaceFileReader.cpp",
                       "open .stl file to read 3D surface",
                       "Expected keyword \"facet\"");
        ++tok_iter;

        // Check if a normal different from zero is given
        if (tok_iter != tokens.end())
        {
          //cout << "Expecting normal" << endl;

          if  (*tok_iter != "normal")
            dolfin_error("SurfaceFileReader.cpp",
                         "open .stl file to read 3D surface",
                         "Expected keyword \"normal\"");
          ++tok_iter;

          //cout << "Read line: " << line << endl;
          
          // for (uint i = 0; i < 3; ++i)
          // {
          //   normal[i] = strToDouble(*tok_iter);
          //   ++tok_iter;
          // }


          //cout << "Normal: " << normal << endl;
          // if (normal.norm() > DOLFIN_EPS)
          //   has_normal = true;
          
          // if (tok_iter != tokens.end())
          //   dolfin_error("SurfaceFileReader.cpp",
          //                "open .stl file to read 3D surface",
          //                "Expected end of line");
        }
      }

      // Read "outer loop" line
      std::getline(file, line);
      boost::algorithm::trim(line);
        
      if (line != "outer loop")
        dolfin_error("SurfaceFileReader.cpp",
                     "open .stl file to read 3D surface",
                     "Expected key word outer loop");

      std::vector<std::size_t> v_indices(3);

      // Read lines with vertices
      for (uint i = 0; i < 3; ++i)
      {
        std::getline(file, line);
        boost::algorithm::trim(line);

        //cout << "read line: " << line << endl;

        tokenizer tokens(line, sep);
        tokenizer::iterator tok_iter = tokens.begin();

        if (*tok_iter != "vertex")
          dolfin_error("SurfaceFileReader.cpp",
                       "open .stl file to read 3D surface",
                       "Expected key word vertex");


        ++tok_iter;

        const double x = strToDouble(*tok_iter); ++tok_iter;
        const double y = strToDouble(*tok_iter); ++tok_iter;
        const double z = strToDouble(*tok_iter); ++tok_iter;

        boost::tuple<double, double, double> v(x, y, z);
      
        if (vertex_map.count(v) > 0)
        {
          v_indices[i] = vertex_map[v];
        }
        else
        {
          vertex_map[v] = num_vertices;
          v_indices[i] = num_vertices;
          //cout << "Adding vertex " << num_vertices << " : " << x << " " << y << " " << z << endl;
          builder.add_vertex(csg::Exact_Point_3(x, y, z));
          //cout << "Done adding vertex" << endl;
          num_vertices++;
        }
      }

      // TODO
      // if (has_normal)
      // {
      //   cout << "Has normal" << endl;
      // }
      
      //cout << "Adding facet : " << v_indices[0] << " " << v_indices[1] << " " << v_indices[2] << endl;
      builder.add_facet(v_indices.begin(), v_indices.end());
      //facets.push_back(v_indices);

      // Read 'endloop' line
      std::getline(file, line);
      boost::algorithm::trim(line);
      if (line != "endloop")
        dolfin_error("SurfaceFileReader.cpp",
                     "open .stl file to read 3D surface",
                     "Expected key word endloop");

      std::getline(file, line);
      boost::algorithm::trim(line);
      if (line != "endfacet")
        dolfin_error("SurfaceFileReader.cpp",
                     "open .stl file to read 3D surface",
                     "Expected key word endfacet");

      std::getline(file, line);
      boost::algorithm::trim(line);

      if (line.substr(0, 5) != "facet")
        break;
    }

    // Read the 'endsolid' line
    tokenizer tokens(line, sep);
    tokenizer::iterator tok_iter = tokens.begin();
  
    if (*tok_iter != "endsolid")
      dolfin_error("SurfaceFileReader.cpp",
                   "open .stl file to read 3D surface",
                   "Expected key word endsolid");

    ++tok_iter;

    // Add all the facets
    //cout << "Inputting facets" << endl;
    // for (std::vector<std::vector<std::size_t> >::iterator it = facets.begin();
    //      it != facets.end(); it++)
    // {
    //   builder.add_facet(it->begin(), it->end());
    // }

    builder.end_surface();
    
    // TODO: Check name of solid
    
    cout << "Done reading surface" << endl;
  }
    std::string filename;
};
//-----------------------------------------------------------------------------
void csg::SurfaceFileReader::readSurfaceFile(std::string filename, Exact_Polyhedron_3& p)
{
  boost::filesystem::path fpath(filename);
  if (fpath.extension() == ".stl")
  {
    readSTLFile(filename, p);
  } 
  else if(fpath.extension() == ".off")
  {
    // TODO: Let cgal parse the file
  } 
  else
  {
    dolfin_error("SurfaceFileReader.cpp",
                 "open file to read 3D surface",
                 "Unknown file type");
  }
}
//-----------------------------------------------------------------------------
void csg::SurfaceFileReader::readSTLFile(std::string filename, Exact_Polyhedron_3& p)
{
  BuildFromSTL<csg::Exact_HalfedgeDS> stl_builder(filename);
  p.delegate(stl_builder);
}
//-----------------------------------------------------------------------------
bool csg::SurfaceFileReader::has_self_intersections(csg::Exact_Polyhedron_3& p)
{
  // compute self-intersections
  typedef std::list<csg::Exact_Triangle_3>::iterator Iterator;
  typedef CGAL::Box_intersection_d::Box_with_handle_d<double,3,Iterator> Box;
  typedef std::back_insert_iterator<std::list<csg::Exact_Triangle_3> > OutputIterator;

  std::list<csg::Exact_Triangle_3> triangles; // intersecting triangles
  ::self_intersect<csg::Exact_Polyhedron_3, csg::Exact_Kernel,OutputIterator>(p, std::back_inserter(triangles));

  if(triangles.size() != 0)
    cout << "Found " << triangles.size() << " found." << endl;
  else 
    cout << "The polyhedron does not self-intersect." << endl;

  return triangles.size() > 0;
}
//-----------------------------------------------------------------------------
CGAL::Bbox_3 csg::SurfaceFileReader::getBoundingBox(csg::Polyhedron_3& polyhedron)
{
  csg::Polyhedron_3::Vertex_iterator it=polyhedron.vertices_begin();

  // Initialize bounding box with the first point
  csg::Polyhedron_3::Point_3 p = it->point();
  CGAL::Bbox_3 b(p[0],p[1],p[2],p[0],p[1],p[2]);
  ++it;

  for (; it != polyhedron.vertices_end(); ++it)
  {
    csg::Polyhedron_3::Point_3 p = it->point();
    b = b + CGAL::Bbox_3(p[0],p[1],p[2],p[0],p[1],p[2]);
  }

  return b;
}
//-----------------------------------------------------------------------------
double csg::SurfaceFileReader::getBoundingSphereRadius(csg::Polyhedron_3& polyhedron)
{
  typedef CGAL::Min_sphere_of_spheres_d_traits_3<csg::Polyhedron_3::Traits, double> Traits;
  typedef Traits::Sphere Sphere;
  typedef CGAL::Min_sphere_of_spheres_d<Traits> Min_sphere;

  std::vector<Sphere> s(polyhedron.size_of_vertices());

  for (csg::Polyhedron_3::Vertex_iterator it=polyhedron.vertices_begin(); 
       it != polyhedron.vertices_end(); ++it)
  {
    const csg::Polyhedron_3::Point_3 p = it->point();
    s.push_back(Sphere(p, 0.0));
  }

  Min_sphere ms(s.begin(),s.end());

  dolfin_assert(ms.is_valid());
  
  return CGAL::to_double(ms.radius());
}
