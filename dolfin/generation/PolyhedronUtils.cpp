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
// Last changed: 2012-11-14

#ifdef HAS_CGAL

#include "PolyhedronUtils.h"
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
class BuildFromSTL : public CGAL::Modifier_base<HDS>
{
public:
  BuildFromSTL(std::string filename) : _filename(filename){}
  void operator()(HDS& hds)
  {
    cout << "Reading surface from " << _filename << endl;

    CGAL::Polyhedron_incremental_builder_3<HDS> builder( hds, true);
    builder.begin_surface(100000, 100000);

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

    std::ifstream file(_filename.c_str());
    if (!file.is_open())
    {
      dolfin_error("PolyhedronUtils.cpp",
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
    {
      dolfin_error("PolyhedronUtils.cpp",
                   "open .stl file to read 3D surface",
                   "File does not start with \"solid\"");
    }

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
          dolfin_error("PolyhedronUtils.cpp",
                       "open .stl file to read 3D surface",
                       "Expected keyword \"facet\"");
        ++tok_iter;

        // Check if a normal different from zero is given
        if (tok_iter != tokens.end())
        {
          //cout << "Expecting normal" << endl;

          if  (*tok_iter != "normal")
            dolfin_error("PolyhedronUtils.cpp",
                         "open .stl file to read 3D surface",
                         "Expected keyword \"normal\"");
          ++tok_iter;

          //cout << "Read line: " << line << endl;

          // for (std::size_t i = 0; i < 3; ++i)
          // {
          //   normal[i] = strToDouble(*tok_iter);
          //   ++tok_iter;
          // }


          //cout << "Normal: " << normal << endl;
          // if (normal.norm() > DOLFIN_EPS)
          //   has_normal = true;

          // if (tok_iter != tokens.end())
          //   dolfin_error("PolyhedronUtils.cpp",
          //                "open .stl file to read 3D surface",
          //                "Expected end of line");
        }
      }

      // Read "outer loop" line
      std::getline(file, line);
      boost::algorithm::trim(line);

      if (line != "outer loop")
        dolfin_error("PolyhedronUtils.cpp",
                     "open .stl file to read 3D surface",
                     "Expected key word outer loop");

      std::vector<std::size_t> v_indices(3);

      // Read lines with vertices
      for (std::size_t i = 0; i < 3; ++i)
      {
        std::getline(file, line);
        boost::algorithm::trim(line);

        //cout << "read line: " << line << endl;

        tokenizer tokens(line, sep);
        tokenizer::iterator tok_iter = tokens.begin();

        if (*tok_iter != "vertex")
        {
          dolfin_error("PolyhedronUtils.cpp",
                       "open .stl file to read 3D surface",
                       "Expected key word vertex");
        }
        ++tok_iter;

        const double x = strToDouble(*tok_iter); ++tok_iter;
        const double y = strToDouble(*tok_iter); ++tok_iter;
        const double z = strToDouble(*tok_iter); ++tok_iter;

        boost::tuple<double, double, double> v(x, y, z);

        if (vertex_map.count(v) > 0)
          v_indices[i] = vertex_map[v];
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
      {
        dolfin_error("PolyhedronUtils.cpp",
                     "open .stl file to read 3D surface",
                     "Expected key word endloop");
      }

      std::getline(file, line);
      boost::algorithm::trim(line);
      if (line != "endfacet")
      {
        dolfin_error("PolyhedronUtils.cpp",
                     "open .stl file to read 3D surface",
                     "Expected key word endfacet");
      }

      std::getline(file, line);
      boost::algorithm::trim(line);

      if (line.substr(0, 5) != "facet")
        break;
    }

    // Read the 'endsolid' line
    tokenizer tokens(line, sep);
    tokenizer::iterator tok_iter = tokens.begin();

    if (*tok_iter != "endsolid")
    {
      dolfin_error("PolyhedronUtils.cpp",
                   "open .stl file to read 3D surface",
                   "Expected key word endsolid");
    }
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
    const std::string _filename;
};
//-----------------------------------------------------------------------------
void PolyhedronUtils::readSurfaceFile(std::string filename, csg::Exact_Polyhedron_3& p)
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
    dolfin_error("PolyhedronUtils.cpp",
                 "open file to read 3D surface",
                 "Unknown file type");
  }
}
//-----------------------------------------------------------------------------
void PolyhedronUtils::readSTLFile(std::string filename, csg::Exact_Polyhedron_3& p)
{
  BuildFromSTL<csg::Exact_HalfedgeDS> stl_builder(filename);
  p.delegate(stl_builder);
}
//-----------------------------------------------------------------------------
CGAL::Bbox_3 PolyhedronUtils::getBoundingBox(csg::Polyhedron_3& polyhedron)
{
  csg::Polyhedron_3::Vertex_iterator it = polyhedron.vertices_begin();

  // Initialize bounding box with the first point
  csg::Polyhedron_3::Point_3 p0 = it->point();
  CGAL::Bbox_3 b(p0[0], p0[1], p0[2], p0[0], p0[1], p0[2]);
  ++it;

  for (; it != polyhedron.vertices_end(); ++it)
  {
    csg::Polyhedron_3::Point_3 p1 = it->point();
    b = b + CGAL::Bbox_3(p1[0], p1[1], p1[2], p1[0], p1[1], p1[2]);
  }

  return b;
}
//-----------------------------------------------------------------------------
double PolyhedronUtils::getBoundingSphereRadius(csg::Polyhedron_3& polyhedron)
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
//-----------------------------------------------------------------------------
template<typename Polyhedron>
static inline double get_edge_length(typename Polyhedron::Halfedge::Halfedge_handle halfedge)
{
  return CGAL::to_double((halfedge->vertex()->point() -
    halfedge->opposite()->vertex()->point()).squared_length());
}
//-----------------------------------------------------------------------------
template <typename Polyhedron>
static inline double get_triangle_area(typename Polyhedron::Facet_handle facet)
{
  const typename Polyhedron::Halfedge_handle edge = facet->halfedge();
  const typename Polyhedron::Point_3 a = edge->vertex()->point();
  const typename Polyhedron::Point_3 b = edge->next()->vertex()->point();
  const typename Polyhedron::Point_3 c = edge->next()->next()->vertex()->point();
  return CGAL::to_double(CGAL::cross_product(b-a, c-a).squared_length());
}
//-----------------------------------------------------------------------------
template<typename Polyhedron>
static inline double get_min_edge_length(typename Polyhedron::Facet_handle facet)
{
  typename Polyhedron::Facet::Halfedge_around_facet_circulator half_edge = facet->facet_begin();
  double min_length = CGAL::to_double((half_edge->vertex()->point()
      - half_edge->opposite()->vertex()->point()).squared_length());

  half_edge++;
  min_length = std::min(min_length, get_edge_length<Polyhedron>(half_edge));

  half_edge++;
  min_length = std::min(min_length, get_edge_length<Polyhedron>(half_edge));

  return min_length;
}
//-----------------------------------------------------------------------------
template<typename Polyhedron>
bool facet_is_degenerate(typename Polyhedron::Facet_handle facet, const double threshold)
{
  // print_facet(facet);
  // if ( get_min_edge_length(facet) < threshold || get_triangle_area(facet) < threshold )
  //   cout << "Is degenerate" << endl;
  // else
  //   cout << "Not degenerate" << endl;

  return get_min_edge_length<Polyhedron>(facet) < threshold
      || get_triangle_area<Polyhedron>(facet) < threshold;
  //return get_min_edge_length(facet) < threshold;
}
//-----------------------------------------------------------------------------
template<typename Polyhedron>
static int number_of_degenerate_facets(Polyhedron& p, const double threshold)
{
  int count = 0;
  for (typename Polyhedron::Facet_iterator facet = p.facets_begin();
       facet != p.facets_end(); facet++)
  {
    dolfin_assert(facet->is_triangle());
    if ( facet_is_degenerate<Polyhedron>(facet, threshold) )
      count++;
  }
  return count;
}
//-----------------------------------------------------------------------------
template <typename Polyhedron>
static typename Polyhedron::Halfedge_handle get_longest_edge(typename Polyhedron::Facet_handle facet)
{
  typename Polyhedron::Halfedge_handle edge = facet->halfedge();
  double length = get_edge_length<Polyhedron>(edge);

  {
    typename Polyhedron::Halfedge_handle e_tmp = edge->next();
    if (get_edge_length<Polyhedron>(e_tmp) > length)
    {
      length = get_edge_length<Polyhedron>(e_tmp);
      edge = e_tmp;
    }
  }

  {
    typename Polyhedron::Halfedge_handle e_tmp = edge->next()->next();
    if ( get_edge_length<Polyhedron>(e_tmp) > length )
    {
      length = get_edge_length<Polyhedron>(e_tmp);
      edge = e_tmp;
    }
  }

  return edge;
}
//-----------------------------------------------------------------------------
template <typename Polyhedron>
double shortest_edge(Polyhedron& p)
{
  double shortest = std::numeric_limits<double>::max();
  for (typename Polyhedron::Halfedge_iterator halfedge = p.halfedges_begin();
       halfedge != p.halfedges_end(); halfedge++)
  {
    const double length = get_edge_length<Polyhedron>(halfedge);
    shortest = std::min(shortest, length);
  }

  return shortest;
}
//-----------------------------------------------------------------------------
template <typename Polyhedron>
static void remove_edge(Polyhedron& p, typename Polyhedron::Halfedge_handle& edge)
{

  // // FIXME: Is it possible to do this in a smarter way than a linear scan
  // for (csg::Polyhedron_3::Facet_iterator facet = p.facets_begin();
  //      facet != p.facets_end(); facet++)
  // {
  //   if ( facet_is_degenerate<csg::Polyhedron_3>(facet, threshold) )
  //   {
  //     //print_facet(facet);

  //     // Find a short edge
  //     csg::Polyhedron_3::Halfedge::Halfedge_handle shortest_edge = facet->facet_begin();
  //     csg::Polyhedron_3::Facet::Halfedge_around_facet_circulator current_edge = facet->facet_begin();
  //     double min_length = get_edge_length(current_edge);

  //     for (int i = 0; i < 2; i++)
  //     {
  // 	current_edge++;
  // 	if (get_edge_length(current_edge) < min_length)
  // 	{
  // 	  shortest_edge = current_edge;
  // 	  min_length = get_edge_length(current_edge);
  // 	}
  //     }

  // Join small triangles with neighbor facets
  edge = p.join_facet(edge->next());
  p.join_facet(edge->opposite()->prev());

  // The joined facets are now quads
  // Join the two close vertices
  p.join_vertex(edge);
}
//-----------------------------------------------------------------------------
template <typename Polyhedron>
static void remove_short_edges(Polyhedron& p, const double threshold)
{
  while (true)
  {
    bool removed = false;
    for (typename Polyhedron::Halfedge_iterator halfedge = p.halfedges_begin();
	 halfedge != p.halfedges_end(); halfedge++)
    {
      if (get_edge_length<Polyhedron>(halfedge) < threshold)
      {
	remove_edge<Polyhedron>(p, halfedge);
	removed = true;
	break;
      }
    }

    if (!removed)
      break;
  }
}
//-----------------------------------------------------------------------------
template <typename Polyhedron>
static typename Polyhedron::Point_3 facet_midpoint(typename Polyhedron::Facet_handle facet)
{
  typename Polyhedron::Point_3 p(CGAL::ORIGIN);

  typename Polyhedron::Facet::Halfedge_around_facet_circulator half_edge = facet->facet_begin();

  for (std::size_t i = 0; i < facet->facet_degree(); i++)
  {
    p = p + (half_edge->vertex()->point() - CGAL::ORIGIN);
    half_edge++;
  }

  p = CGAL::ORIGIN + (p - CGAL::ORIGIN)/static_cast<double>(facet->facet_degree());

  // std::cout << "Center coordinates computed: " << p << std::endl;

  // half_edge = facet->facet_begin();
  // for (std::size_t i = 0; i < facet->facet_degree(); i++)
  // {
  //   std::cout << "Distance to point << " << half_edge->vertex()->point() << " = " << (half_edge->vertex()->point() - p).squared_length() << std::endl;
  //   half_edge++;
  // }

  return p;
}
//-----------------------------------------------------------------------------
template <typename Polyhedron>
static void remove_triangle(Polyhedron& p, typename Polyhedron::Facet_handle facet)
{
  dolfin_assert(facet->is_triangle());

  // cout << "Removing triangle" << endl;
  // print_facet<Polyhedron>(facet);

  // Find the longest edge
  typename Polyhedron::Halfedge_handle edge = get_longest_edge<Polyhedron>(facet);

  // cout << "Longest edge" << endl;
  // print_halfedge<Polyhedron>(edge);

  // cout << "Opposite triangle" << endl;
  // print_facet<Polyhedron>(edge->opposite()->facet());

  edge = p.join_facet(edge);
  // cout << "Edge after join: " << endl;
  // print_halfedge<Polyhedron>(edge);

  // cout << "Facet after join" << endl;
  // print_facet<Polyhedron>(edge->facet());

  typename Polyhedron::Point_3 new_center = facet_midpoint<Polyhedron>(edge->facet());

  edge = p.create_center_vertex(edge);

  edge->vertex()->point() = new_center;

  // std::cout << "Center vertex: " << edge->vertex()->point() << std::endl;

  // for (std::size_t i=0; i < 4; i++)
  // {
  //   print_facet<Polyhedron>(edge->facet());
  //   edge = edge->next()->opposite();
  // }
}
//-----------------------------------------------------------------------------
template<typename Polyhedron>
static void remove_small_triangles(Polyhedron& p, const double threshold)
{
  int n = number_of_degenerate_facets(p, threshold);

  while (n > 0)
  {
    for (typename Polyhedron::Facet_iterator facet = p.facets_begin();
	 facet != p.facets_end(); facet++)
    {
      dolfin_assert(facet->is_triangle());

      if (get_triangle_area<Polyhedron>(facet) < threshold)
      {
	// cout << "Small triangle detected" << endl;
	// print_facet<Polyhedron>(facet);
	remove_triangle<Polyhedron>(p, facet);
	n = number_of_degenerate_facets<Polyhedron>(p, threshold);
	break;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void PolyhedronUtils::remove_degenerate_facets(csg::Exact_Polyhedron_3& p, const double threshold)
{
  int degenerate_facets = number_of_degenerate_facets(p, threshold);

  cout << "Number of degenerate facets: " << degenerate_facets << endl;
  // FIXME: Use has_degenerate_facets() when debugging is done
  if (degenerate_facets > 0)
  {
    dolfin_assert(p.is_pure_triangle());

    shortest_edge(p);

    cout << "Removing triangles with short edges" << endl;
    remove_short_edges(p, threshold);

    cout << "Number of degenerate facets: " << number_of_degenerate_facets(p, threshold) << endl;

    cout << "Removing small triangles" << endl;
    remove_small_triangles(p, threshold);

    cout << "Number of degenerate facets: " << number_of_degenerate_facets(p, threshold) << endl;

    // Removal of triangles should preserve the triangular structure of the polyhedron
    dolfin_assert(p.is_pure_triangle());
  }
}
//-----------------------------------------------------------------------------
bool PolyhedronUtils::has_degenerate_facets(csg::Exact_Polyhedron_3& p, double threshold)
{
  for (csg::Exact_Polyhedron_3::Facet_iterator facet = p.facets_begin();
       facet != p.facets_end(); facet++)
  {
    dolfin_assert(facet->is_triangle());

    if ( facet_is_degenerate<csg::Exact_Polyhedron_3>(facet, threshold) )
      return true;
  }
  return false;
}

#endif
