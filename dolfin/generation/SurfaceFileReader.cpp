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


void csg::SurfaceFileReader::readSurfaceFile(std::string filename, Exact_Polyhedron_3& p)
{
  boost::filesystem::path fpath(filename);
  if (fpath.extension() == ".stl")
  {
    readSTLFile(filename, p);
  } 
  else if(fpath.extension() == ".off")
  {

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
  cout << "Reading surface from " << filename << endl;

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
  std::vector<uint[3]> triangles;
  std::string line;


  // Read the first line and trim away whitespaces
  std::getline(file, line);
  boost::algorithm::trim(line);
  
  if (line.substr(0, 5) != "solid")
    dolfin_error("SurfaceFileReader.cpp",
                 "open .stl file to read 3D surface",
                 "File does not start with \"solid\"");

  // TODO: Read name of solid

  while (file.good())
  {
    // Read the line "facet normal n1 n2 n3"
    std::getline(file, line);
    boost::algorithm::trim(line);
    
    {
      std::getline(file, line);

      boost::char_separator<char> sep(" ");
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

      boost::char_separator<char> sep(" ");
      tokenizer tokens(line, sep);

      tokenizer::iterator tok_iter = tokens.begin();

      if (*tok_iter != "vertex")
        dolfin_error("SurfaceFileReader.cpp",
                     "open .stl file to read 3D surface",
                     "Expected key word vertex");

      boost::tuple<double, double, double> v(strToDouble(*(tok_iter++)), 
                                             strToDouble(*(tok_iter++)), 
                                             strToDouble(*(tok_iter++)));
      
      if (vertex_map.count(v))
        v_indices[i] = vertex_map[v];
      else
      {
        vertex_map[v] = num_vertices;
        v_indices[i] = num_vertices;
        num_vertices++;
      }
    }

    // TODO: Register
    cout << "Found triangle: "
         << v_indices[0] << " " 
         << v_indices[1] << " " 
         << v_indices[2] << endl;

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
  boost::char_separator<char> sep(" ");
  tokenizer tokens(line, sep);

  tokenizer::iterator tok_iter = tokens.begin();
  
  if (*tok_iter != "endsolid")
    dolfin_error("SurfaceFileReader.cpp",
                 "open .stl file to read 3D surface",
                 "Expected key word endsolid");

  ++tok_iter;

  // TODO: Check name of solid

  cout << "Done reading surface" << endl;
}
