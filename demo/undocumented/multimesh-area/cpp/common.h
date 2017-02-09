// Utilities for debugging multimesh

#include <fstream>
#include <dolfin/generation/UnitSquareMesh.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/geometry/CGALExactArithmetic.h>

#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
// Note that Qoutient<MP_FLOAT> as number type gives overflow
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>

// typedef CGAL::Quotient<CGAL::MP_Float>            ExactNumber;
// typedef CGAL::Cartesian<ExactNumber>              ExactKernel;

typedef CGAL::Epeck                               ExactKernel;
typedef ExactKernel::FT                           FT;

typedef ExactKernel::Point_2                      Point_2;
typedef ExactKernel::Vector_2                     Vector_2;
typedef ExactKernel::Triangle_2                   Triangle_2;
typedef ExactKernel::Segment_2                    Segment_2;
typedef ExactKernel::Line_2                       Line_2;
typedef CGAL::Polygon_2<ExactKernel>              Polygon_2;
typedef Polygon_2::Vertex_const_iterator          Vertex_const_iterator;
typedef CGAL::Polygon_with_holes_2<ExactKernel>   Polygon_with_holes_2;
typedef Polygon_with_holes_2::Hole_const_iterator Hole_const_iterator;
typedef CGAL::Polygon_set_2<ExactKernel>          Polygon_set_2;

typedef std::vector<std::pair<Point_2, FT>> cgal_QR;
typedef std::vector<Triangle_2> Polygon;
#endif

namespace dolfin
{
  //-----------------------------------------------------------------------------
  enum CELL_STATUS
    {
      UNKNOWN,
      COVERED,
      CUT,
      UNCUT
    };

  //-----------------------------------------------------------------------------
  std::string cell_status_str(CELL_STATUS cs)
  {
    switch(cs)
    {
    case UNKNOWN :
      return "UNKNOWN";
    case COVERED :
      return "COVERED";
    case CUT :
      return "CUT    ";
    case UNCUT :
      return "UNCUT  ";
    }
  }

  //------------------------------------------------------------------------------
#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
  std::pair<Point_2, FT> cgal_compute_quadrature_rule(Triangle_2 t, FT factor)
  {
    const Vector_2 a = t[1]-t[0];
    const Vector_2 b = t[2]-t[0];

    // Compute double the area of the triangle
    const FT det = CGAL::abs(a.x()*b.y() - a.y()*b.x());

    // qr.push_back(std::make_pair( CGAL::ORIGIN + (t[0]-CGAL::ORIGIN)/3 + (t[1]-CGAL::ORIGIN)/3 + (t[2]-CGAL::ORIGIN)/3,
    return std::make_pair( CGAL::centroid(t), factor*det/2 );
  }
#endif
  //------------------------------------------------------------------------------
  inline double compute_area_using_quadrature(const MultiMesh& multimesh)
  {
    //std::cout  << __FUNCTION__ << std::endl;

    double area = 0;
    std::vector<double> all_areas;

    std::ofstream file("quadrature_interface.txt");
    if (!file.good()) { std::cout << "file not good"<<std::endl; exit(0); }
    file.precision(20);

    // Sum contribution from all parts
    // std::cout << "Sum contributions"<<std::endl;
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      //std::cout << "% part " << part << '\n';
      double part_area = 0;
      const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);

      // // Uncut cell area given by function area
      // const auto uncut_cells = multimesh.uncut_cells(part);
      // for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      // {
      //   const Cell cell(*multimesh.part(part), *it);
      //   area += cell.area();
      // 	//std::cout << std::setprecision(20) << cell.area() <<std::endl;
      //   part_area += cell.area();
      // 	status[*it] = 1;
      // 	//file << "0 0 "<< cell.area() << std::endl;
      // }

      // std::cout << "\t uncut area "<< part_area << ' ';


      // Get collision map
      const auto& cmap = multimesh.collision_map_cut_cells(part);
      for (auto it = cmap.begin(); it != cmap.end(); ++it)
      {
	const unsigned int cut_cell_index = it->first;
	const auto& cutting_cells = it->second;

	// Iterate over cutting cells
	for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
	{
	  // Get quadrature rule for interface part defined by
	  // intersection of the cut and cutting cells
	  const std::size_t k = jt - cutting_cells.begin();
	  // std::cout << cut_cell_index << ' ' << k <<' ' << std::flush
	  // 	    << quadrature_rules.size() << ' '
	  // 	    << quadrature_rules.at(cut_cell_index).size() << "   " << std::flush;
	  dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
	  const auto& qr = quadrature_rules.at(cut_cell_index)[k];
	  std::stringstream ss;
	  for (std::size_t i = 0; i < qr.second.size(); ++i)
	  {
	    file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<std::endl;
	    //std::cout << qr.second[i]<<' ';
	    area += qr.second[i];
	    part_area += qr.second[i];
	    // std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<std::endl;
	  }
	  //tools::cout_qr(qr);
	  //std::cout << std::endl;
	}
      }
      //std::cout << "% total area " << part_area << std::endl;
      all_areas.push_back(part_area);
      // {char apa; std::cout << "paused at " << __FUNCTION__<<' '<<__LINE__<<std::endl; std::cin>>apa;}
    }
    file.close();

    return area;
  }
  //------------------------------------------------------------------------------
  // Compute volume contributions from each cell
  // Return number of quadrature points in each cut cell for each part
  std::vector<std::vector<std::size_t> > compute_volume(const MultiMesh& multimesh,
							std::vector<std::vector<std::pair<CELL_STATUS, double> > >& cells_status)
  {
    cells_status.reserve(multimesh.num_parts());
    std::ofstream file("quadrature_volume.txt");
    if (!file.good()) { std::cout << "file not good"<<std::endl; exit(0); }
    file.precision(20);

    std::vector<std::vector<std::size_t> > num_qr(multimesh.num_parts());
    cells_status.resize(multimesh.num_parts());

    // Compute contribution from all parts
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      // std::cout << "Testing part " << part << std::endl;
      // cells_status.push_back(std::vector<std::pair<CELL_STATUS, double> >());
      //std::vector<std::pair<CELL_STATUS, double> >& current_cells_status = cells_status.back();
      std::vector<std::pair<CELL_STATUS, double> >& current_cells_status = cells_status[part];

      std::shared_ptr<const Mesh> current_mesh = multimesh.part(part);
      current_cells_status.resize(current_mesh->num_cells());
      // std::cout << "Number of cells: " << current_cells_status.size() << std::endl;
      num_qr[part] = std::vector<std::size_t>(current_mesh->num_cells(), 0);

      // Uncut cell volume given by function volume
      {
	const std::vector<unsigned int>& uncut_cells = multimesh.uncut_cells(part);
	for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
	{
	  const Cell cell(*multimesh.part(part), *it);
	  const double vol = cell.volume();

	  // // orient2d volume from CGALExactArithmetic
	  // const MeshGeometry& geometry = cell.mesh().geometry();
	  // const unsigned int* vertices = cell.entities(0);
	  // const Point x0 = geometry.point(vertices[0]);
	  // const Point x1 = geometry.point(vertices[1]);
	  // const Point x2 = geometry.point(vertices[2]);
	  // const double vol = std::abs(volume({x0, x1, x2}));

	  current_cells_status[*it] = std::make_pair(UNCUT, vol);
	}
      }

      // Cut cell volume given by quadrature rule
      {
	const std::vector<unsigned int>& cut_cells = multimesh.cut_cells(part);

	for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
	{
	  //std::cout << "Cut cell in part " << part << ": " << *it << std::endl;
	  file << "% Cut cell in part " << part << ": " << *it << std::endl;
	  double volume = 0;
	  const quadrature_rule& qr = multimesh.quadrature_rule_cut_cell(part, *it);
	  //std::cout << "QR: " << qr.first.size() << ", " << qr.second.size() << std::endl;
	  num_qr[part][*it] = qr.second.size();
	  for (std::size_t i = 0; i < qr.second.size(); ++i)
	  {
	    volume += qr.second[i];
	    file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<std::endl;
	  }
	  current_cells_status[*it] = std::make_pair(CUT, volume);
	}
      }

      {
	const std::vector<unsigned int>& covered_cells = multimesh.covered_cells(part);
	for (auto it = covered_cells.begin(); it != covered_cells.end(); ++it)
	{
	  current_cells_status[*it] = std::make_pair(COVERED, 0.);
	}
      }

      // std::cout << "part "<< part << " qr volume " << volume << std::endl;
    }
    return num_qr;
  }
#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
  //-----------------------------------------------------------------------------
  Triangle_2 convert_to_triangle_2(const Cell& cell)
  {
    const std::size_t tdim = cell.mesh().topology().dim();
    dolfin_assert(tdim == 2);
    const MeshGeometry& geometry = cell.mesh().geometry();
    const unsigned int* vertices = cell.entities(0);
    Triangle_2 tri(Point_2(geometry.x(vertices[0],0),
			   geometry.x(vertices[0],1)),
		   Point_2(geometry.x(vertices[1],0),
			   geometry.x(vertices[1],1)),
		   Point_2(geometry.x(vertices[2],0),
			   geometry.x(vertices[2],1)));
    if (tri.orientation() == CGAL::CLOCKWISE)
      tri = tri.opposite();
    return tri;
  }
  //-----------------------------------------------------------------------------
  Triangle_2 convert_to_triangle_2(const std::vector<Point>& pts)
  {
    dolfin_assert(pts.size() == 3);

    Triangle_2 tri(Point_2(pts[0][0], pts[0][1]),
		   Point_2(pts[1][0], pts[1][1]),
		   Point_2(pts[2][0], pts[2][1]));

    if (tri.orientation() == CGAL::CLOCKWISE)
      tri = tri.opposite();

    return tri;
  }

  //-----------------------------------------------------------------------------
  Polygon_2 convert_to_polygon_2(const Triangle_2& tri)
  {
    std::vector<Point_2> vertices;
    vertices.push_back(tri[0]);
    vertices.push_back(tri[1]);
    vertices.push_back(tri[2]);

    Polygon_2 p(vertices.begin(), vertices.end());
    return p;
  }
#endif
  //-----------------------------------------------------------------------------
  inline std::vector<Point> convert_to_points(const Cell& cell)
  {
    const std::size_t tdim = cell.mesh().topology().dim();
    std::vector<Point> simplex(tdim + 1);
    const MeshGeometry& geometry = cell.mesh().geometry();
    const unsigned int* vertices = cell.entities(0);
    for (std::size_t j = 0; j < tdim + 1; ++j)
      simplex[j] = geometry.point(vertices[j]);
    return simplex;
  }

#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
  //-----------------------------------------------------------------------------
  double cgal_intersection_volume(const std::vector<Point>& A,
				  const std::vector<Point>& B)
  {
    Triangle_2 A_tri = convert_to_triangle_2(A);
    Triangle_2 B_tri = convert_to_triangle_2(B);
    Polygon_2 A_pol = convert_to_polygon_2(A_tri);
    Polygon_2 B_pol = convert_to_polygon_2(B_tri);

    Polygon_set_2 polygon_set;
    polygon_set.insert(A_pol);
    polygon_set.join(B_pol); // intersection

    std::vector<Polygon_with_holes_2> result;
    polygon_set.polygons_with_holes(std::back_inserter(result));

    FT vol = 0;

    for(auto pit = result.begin(); pit != result.end(); pit++)
    {
      const Polygon_2& outerboundary = pit->outer_boundary();
      vol += outerboundary.area();
    }
    return CGAL::to_double(vol);
  }

  //-----------------------------------------------------------------------------
  double cgal_intersection_volume(const Cell& A,
				  const Cell& B)
  {
    const std::vector<Point> pA = convert_to_points(A), pB = convert_to_points(B);
    return cgal_intersection_volume(pA, pB);
  }

  //------------------------------------------------------------------------------
  void get_cells_status_cgal(const MultiMesh& multimesh,
			     std::vector<std::vector<std::pair<CELL_STATUS, FT> > >& cells_status)
  {
    cells_status.resize(multimesh.num_parts());

    FT volume = 0;

    for (std::size_t i = 0; i < multimesh.num_parts(); i++)
    {
      // std::cout << "Testing part " << i << std::endl;
      //cells_status.push_back(std::vector<std::pair<CELL_STATUS, FT> >());
      //std::vector<std::pair<CELL_STATUS, FT> >& current_cells_status = cells_status.back();
      std::vector<std::pair<CELL_STATUS, FT> >& current_cells_status = cells_status[i];

      std::shared_ptr<const Mesh> current_mesh = multimesh.part(i);
      const MeshGeometry& current_geometry = current_mesh->geometry();

      for (CellIterator cit(*current_mesh); !cit.end(); ++cit)
      {
	// Test every cell against every cell in overlaying meshes
	Triangle_2 current_cell(Point_2(current_geometry.x(cit->entities(0)[0], 0),
					current_geometry.x(cit->entities(0)[0], 1)),
				Point_2(current_geometry.x(cit->entities(0)[1], 0),
					current_geometry.x(cit->entities(0)[1], 1)),
				Point_2(current_geometry.x(cit->entities(0)[2], 0),
					current_geometry.x(cit->entities(0)[2], 1)));
	if (current_cell.orientation() == CGAL::CLOCKWISE)
	{
	  //std::cout << "Orig: " << current_cell << std::endl;
	  current_cell = current_cell.opposite();
	  //std::cout << "Opposite: " << current_cell << std::endl;
	}
	Polygon_set_2 polygon_set;
	{
	  std::vector<Point_2> vertices;
	  vertices.push_back(current_cell[0]);
	  vertices.push_back(current_cell[1]);
	  vertices.push_back(current_cell[2]);

	  Polygon_2 p(vertices.begin(), vertices.end());
	  polygon_set.insert(p);
	}

	//std::cout << "  Testing part " << i << " cell: " << current_cell << std::endl;
	bool is_uncut = true;
	for (std::size_t j = i+1; j < multimesh.num_parts(); j++)
	{
	  // std::cout << "    Testing against part " << j << std::endl;
	  std::shared_ptr<const Mesh> other_mesh = multimesh.part(j);
	  const MeshGeometry& other_geometry = other_mesh->geometry();
	  for (CellIterator cit_other(*other_mesh); !cit_other.end(); ++cit_other)
	  {
	    std::vector<Point_2> vertices;
	    Point_2 p0(other_geometry.x(cit_other->entities(0)[0], 0),
		       other_geometry.x(cit_other->entities(0)[0], 1));
	    Point_2 p1(other_geometry.x(cit_other->entities(0)[1], 0),
		       other_geometry.x(cit_other->entities(0)[1], 1));
	    Point_2 p2(other_geometry.x(cit_other->entities(0)[2], 0),
		       other_geometry.x(cit_other->entities(0)[2], 1));

	    vertices.push_back(p0);
	    if (Line_2(p0, p1).has_on_positive_side(p2))
	    {
	      vertices.push_back(p1);
	      vertices.push_back(p2);
	    }
	    else
	    {
	      vertices.push_back(p2);
	      vertices.push_back(p1);
	    }
	    Polygon_2 p(vertices.begin(), vertices.end());
	    polygon_set.difference(p);
	  }
	}

	std::vector<Polygon_with_holes_2> result;
	polygon_set.polygons_with_holes(std::back_inserter(result));

	if (result.size() == 0)
	{
	  current_cells_status.push_back(std::make_pair(COVERED, 0.0));
	  //std::cout << "    Covered" << std::endl;
	}
	else
	{
	  // if (result.size() > 1)
	  //   std::cout << "!!!!!!!! Several polygons !!!!!!!" << std::endl;

	  Polygon_2::Vertex_const_iterator v = result[0].outer_boundary().vertices_begin();
	  Polygon_2::Vertex_const_iterator v_end = result[0].outer_boundary().vertices_end();
	  const std::size_t num_vertices = std::distance(v, v_end);
	  const Point_2& v0 = *v; ++v;
	  const Point_2& v1 = *v; ++v;
	  const Point_2& v2 = *v;

	  if (result.size() == 1 &&
	      result[0].holes_begin() == result[0].holes_end() &&
	      num_vertices == 3 &&
	      Triangle_2(v0, v1, v2) == current_cell)
	  {
	    current_cells_status.push_back(std::make_pair(UNCUT,
							  result[0].outer_boundary().area()));
	    // std::cout << "    Uncut" << std::endl;
	  }
	  else
	  {
	    FT current_volume = 0;

	    for(auto pit = result.begin(); pit != result.end(); pit++)
	    {
	      const Polygon_2& outerboundary = pit->outer_boundary();

	      // std::cout << "    Polygon ";
	      // for (auto it = outerboundary.vertices_begin(); it != outerboundary.vertices_end(); it++) std::cout << *it << ", ";
	      // std::cout << std::endl;

	      current_volume += outerboundary.area();
	      //std::cout << "vol " << outerboundary.area() << std::endl;

	      // std::cout << "vol holes: ";
	      for (auto it = pit->holes_begin(); it != pit->holes_end(); it++)
	      {
		current_volume += it->area();
		//std::cout << it->area() << ' ';
	      }
	      //std::cout << std::endl;
	    }
	    current_cells_status.push_back(std::make_pair(CUT, current_volume));
	    // std::cout << "    Cut" << std::endl;
	  }
	}
      }
    }
  }
  //-----------------------------------------------------------------------------
#endif
}
