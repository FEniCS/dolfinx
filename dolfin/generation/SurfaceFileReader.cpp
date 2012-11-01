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
// Last changed: 2012-10-31

#include "SurfaceFileReader.h"
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <CGAL/Polyhedron_incremental_builder_3.h>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

using namespace dolfin;

static inline double strToDouble(const std::string& s)
{
  std::istringstream is(s);
  double val;
  is >> val;

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

    CGAL::Polyhedron_incremental_builder_3<HDS> B( hds, true);
    B.begin_surface(0, 0);


    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

    std::ifstream file(filename.c_str());
    if (!file.is_open())
    {
      dolfin_error("SurfaceFileReader.cpp",
                   "open .stl file to read 3D surface",
                   "Failed to open file");
    }

    int num_vertices = 0;
    std::map<boost::tuple<double, double, double>, uint > vertex_map;
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
      // Read the line "facet normal n1 n2 n3"
    
      cout << "Read line: " << line << endl;

      {
        tokenizer tokens(line, sep);
        tokenizer::iterator tok_iter = tokens.begin();

        if (*tok_iter != "facet")
          dolfin_error("SurfaceFileReader.cpp",
                       "open .stl file to read 3D surface",
                       "Expected keyword \"facet\"");
        ++tok_iter;

        bool has_normal = false;
        csg::Exact_Point_3 normal;

        if (tok_iter != tokens.end())
        {
          if  (*tok_iter != "normal")
            dolfin_error("SurfaceFileReader.cpp",
                         "open .stl file to read 3D surface",
                         "Expected keyword \"normal\"");
          ++tok_iter;

          has_normal = true;
          for (uint i = 0; i < 3; ++i)
          {
            normal[i] = strToDouble(*tok_iter);
            ++tok_iter;
          }
          if (tok_iter != tokens.end())
            dolfin_error("SurfaceFileReader.cpp",
                         "open .stl file to read 3D surface",
                         "Expected end of line");
        }

        if (has_normal)
          cout << "Has normal" << endl;
      }

      // Read "outer loop" line
      {
        std::getline(file, line);
        boost::algorithm::trim(line);

        cout << "Read line: " << line << endl;
        
        if (line != "outer loop")
          dolfin_error("SurfaceFileReader.cpp",
                       "open .stl file to read 3D surface",
                       "Expected key word outer loop");
      }

      uint v_indices[3];

      // Read lines with vertices
      for (uint i = 0; i < 3; ++i)
      {
        std::getline(file, line);
        boost::algorithm::trim(line);

        cout << "vertex line : " << line << endl;

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

        cout << "x = " << x << ", y = " << y << ", z = " << z << endl;

        boost::tuple<double, double, double> v(x, y, z);
      
        cout << "Read vertex: " << v << endl;

        if (vertex_map.count(v) > 0)
        {
          v_indices[i] = vertex_map[v];
          cout << i << ": Found vertex: " << v << " : " << v_indices[i] << endl;
        }
        else
        {
          vertex_map[v] = num_vertices;
          v_indices[i] = num_vertices;
          cout << i << ": Adding vertex (" << num_vertices << ") = " << v << endl;
          B.add_vertex(csg::Exact_Point_3(x, y, z));
          num_vertices++;
        }
      }

      // TODO: Register
      cout << "Found triangle: "
           << v_indices[0] << " " 
           << v_indices[1] << " " 
           << v_indices[2] << endl;

      B.add_facet ( v_indices, &v_indices[3]);


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

    B.end_surface();

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
