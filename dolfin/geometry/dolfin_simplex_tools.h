#ifndef DOLFIN_SIMPLEX_TOOLS_H
#define DOLFIN_SIMPLEX_TOOLS_H

#include "predicates.h"


#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdlib>

#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/function/MultiMeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/SubMesh.h>
#include <dolfin/la/Vector.h>

namespace tools
{
#define PPause {char dummycharXohs5su8='a';std::cout<<"\n Pause: "<<__FILE__<<" line "<<__LINE__<<" function "<<__FUNCTION__<<std::endl;std::cin>>dummycharXohs5su8;}

  using namespace dolfin;

  //-----------------------------------------------------------------------------
  inline std::vector<Point> convert(const Cell& cell)
  {
    const std::size_t tdim = cell.mesh().topology().dim();
    std::vector<Point> simplex(tdim + 1);
    const MeshGeometry& geometry = cell.mesh().geometry();
    const unsigned int* vertices = cell.entities(0);
    for (std::size_t j = 0; j < tdim + 1; ++j)
      simplex[j] = geometry.point(vertices[j]);
    return simplex;
  }

  //-----------------------------------------------------------------------------
  // typedef std::vector<Point> Simplex;
  // typedef std::vector<Simplex> Polyhedron;
  inline bool tdimcheck(const std::vector<std::vector<dolfin::Point>>& polygon)
  {
    if (polygon.size() == 0) return false;

    const std::size_t tdimtmp = polygon[0].size();
    for (std::size_t i = 1; i < polygon.size(); ++i)
      if (polygon.at(i).size() != tdimtmp)
	return false;
    return true;
  }

  //-----------------------------------------------------------------------------
  inline bool tdimcheck(const std::vector<std::vector<std::vector<dolfin::Point>>>& pvec)
  {
    if (pvec.size() == 0) return false;
    for (std::size_t i = 0; i < pvec.size(); ++i)
      if (pvec[i].size() == 0)
	return false;

    const std::size_t tdimtmp = pvec[0][0].size();
    for (std::size_t i = 1; i < pvec.size(); ++i)
      for (std::size_t j = 0; j < pvec[i].size(); ++j)
	if (pvec.at(i).at(j).size() != tdimtmp)
	  return false;
    return true;
  }


  //-----------------------------------------------------------------------------
  // display quadrature_rule
  // recall typedef std::pair<std::vector<double>, std::vector<double> > quadrature_rule;
  inline void cout_qr(const std::pair<std::vector<double>, std::vector<double> >& qr,
		      std::string color="'b'",
		      std::size_t markersize=16)
  {
    for (std::size_t i = 0; i < qr.second.size(); ++i)
    {
      std::stringstream ss;
      if (qr.second[i] > 0)
	ss<<"'color',"<<color<<",'marker','.','markersize',"<<markersize;
      else
	ss<<"'color',"<<color<<",'marker','o','markersize',"<<markersize-10;
      //std::cout << "plot("<<qr.first[2*i]<<','<<qr.first[2*i+1]<<','<<marker<<"); % "<<qr.second[i]<<' '<<i<<std::endl;
      std::cout << "plot("<<qr.first[2*i]<<','<<qr.first[2*i+1]<<','<<ss.str()<<"); % "<<qr.second[i]<<' '<<i<<std::endl;
    }
  }

  //-----------------------------------------------------------------------------
  inline void cout_normals(const std::vector<double>& n)
  {
    for (std::size_t i = 0; i < n.size()/2; ++i)
      std::cout << i << ":  "<<n[2*i]<<' '<<n[2*i+1]<<std::endl;
  }

  //-----------------------------------------------------------------------------
  inline double area(const std::pair<std::vector<double>, std::vector<double> >& qr)
  {
    double a = 0;
    for (std::size_t i = 0; i < qr.second.size(); ++i)
      a += qr.second[i];
    return a;
  }

  //-----------------------------------------------------------------------------
  // this sorts such that a >= b >= c
  template<class T>
  inline void sort3(T &a, T &b, T &c)
  {
    if (b>a) std::swap(b,a);
    if (c>b) std::swap(c,b);
    if (b>a) std::swap(b,a);
  }

  //-----------------------------------------------------------------------------
  inline double Heron(double a, double b, double c)
  {
    sort3(a,b,c);
    const double s2 = (a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c));
    if (s2 < 0)
    {
      std::cout << "Heron error, negative sqrt: " << s2 << " is to be replaced with 0" << std::endl;
      if (std::abs(s2) < DOLFIN_EPS)
	return 0;
      else
	exit(1);
    }
    return 0.25*std::sqrt( (a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)) );
  }



  //-----------------------------------------------------------------------------
  inline double area(const std::vector<dolfin::Point> &simplex)
  {
    if (simplex.size() == 3)
    {
      const dolfin::Point et=simplex[1]-simplex[0];
      const dolfin::Point es=simplex[2]-simplex[0];
      return Heron(et.norm(), es.norm(), (et-es).norm());
    }
    else if (simplex.size() == 2)
    {
      return (simplex[0]-simplex[1]).norm();
    }
    else if (simplex.size() == 1)
    {
      return 0.;
    }
    else {
      std::cout << "error simplex size = " << simplex.size();
      PPause;
      return -9e99;
    }

    // if (simplex.size() == 3)
    // {
    //   double a[2]={simplex[0][0],simplex[0][1]};
    //   double b[2]={simplex[1][0],simplex[1][1]};
    //   double c[2]={simplex[2][0],simplex[2][1]};
    //   return 0.5*orient2d(a,b,c);
    // }
    // else if (simplex.size() == 2)
    // {
    //   return (simplex[0]-simplex[1]).norm();
    // }
    // else
    // {
    //   PPause;
    //   return -9e99;
    // }
  }

  //-----------------------------------------------------------------------------
  inline std::string drawtriangle(const std::vector<dolfin::Point> &simplex,
				  const std::string& color = "'b'",
				  bool matlab=true)
  {
    std::stringstream ss; ss.precision(15);
    if (simplex.size() == 3)
    {
      if (matlab)
	ss << "drawtriangle(";
      else
	ss << "drawtriangle2(";
      ss<< "["<<simplex[0][0]<<','<<simplex[0][1]<<"],"
	<< "["<<simplex[1][0]<<','<<simplex[1][1]<<"],"
	<< "["<<simplex[2][0]<<','<<simplex[2][1]<<"]";
      if (matlab)
	ss << ","<<color<<");";
      else {
	ss << ",color=" << color << ',' //<< ");";
	   << "plt=plt,axis=gca()"
	   << ");";
      }
    }
    else if (simplex.size() == 2)
    {
      // ss << "hline = line([" << simplex[0][0] << ',' << simplex[1][0] << "],"
      // 	 << "[" << simplex[0][1] << ',' << simplex[1][1] << "]);"
      // 	 << "set(hline,'color'," << color << ");";
      ss << "drawline([" << simplex[0][0] << ',' << simplex[0][1] << "],"
	 <<  "[" << simplex[1][0] << ',' << simplex[1][1] << "],";
      if (matlab)
	ss << color<<",1,1,15);";
      else
	ss << "plt=plt,color="<< color << ",linewidth=5.0);";
    }
    else if (simplex.size() == 1)
    {
      ss << "plot("<<simplex[0][0]<<','<<simplex[0][1]<<',';
      if (matlab)
	ss << "'k.','markersize',15);";
      else {
	PPause;
	dolfin_assert(false); // /not implemented
      }
    }
    else {
      std::cout << "simplex size to plot is " << simplex.size() << std::endl;
      PPause;
      dolfin_assert(false);
    }
    return ss.str();
  }

  //-----------------------------------------------------------------------------
  inline std::string drawtriangle(const dolfin::Cell &cell,
				  const std::string& color = "'b'",
				  bool matlab=true)
  {
    const std::vector<Point> s = convert(cell);
    return drawtriangle(s, color, matlab);
  }

  //-----------------------------------------------------------------------------
  inline std::string drawtriangle(const std::vector<double>& s,
				  const std::string& color = "'b'",
				  bool matlab = false)
  {
    std::vector<dolfin::Point> pp(s.size() / 2);
    for (std::size_t i = 0; i < pp.size(); ++i)
    {
      pp[i][0] = s[2*i];
      pp[i][1] = s[2*i+1];
    }
    return drawtriangle(pp, color, matlab);

    // std::vector<dolfin::Point> ss(3);
    // ss[0] = dolfin::Point(s[0],s[1]);
    // ss[1] = dolfin::Point(s[2],s[3]);
    // ss[2] = dolfin::Point(s[4],s[5]);
    // return drawtriangle(ss, color);
  }

  //-----------------------------------------------------------------------------
  inline std::string drawtriangle(const dolfin::Point& a,
				  const dolfin::Point& b,
				  const dolfin::Point& c,
				  const std::string color = "'b'",
				  bool matlab = false)
  {
    std::vector<dolfin::Point> t = {{ a, b, c}};
    return drawtriangle(t, color);
  }

  //-----------------------------------------------------------------------------
  inline std::string matlabplot(const dolfin::Point& p,
				const std::string m="'k.','markersize',14")
  {
    std::stringstream ss; ss.precision(15);
    ss<<"plot("<<p[0]<<','<<p[1]<<','<<m<<");";
    return ss.str();
  }

  //-----------------------------------------------------------------------------
  inline std::string plot(const dolfin::Point& p,
			  const std::string m="'k.','markersize',14")
  {
    return matlabplot(p,m);
  }

  //-----------------------------------------------------------------------------
  inline std::string drawarrow(const dolfin::Point& v1,
			       const dolfin::Point& v2,
			       const std::string& color = "'b'")
  {
    std::stringstream ss;ss.precision(15);
    ss << "drawarrow([" << v1[0] << ' '<<v1[1] <<"],[" << v2[0]<<' '<<v2[1] << "], "<< color << ");";
    return ss.str();
    // const dolfin::Point v = v2-v1;
    // const Point ones(0,0,1);
    // Point n = ones.cross(v);
    // if (n.norm() < 1e-5) {
    //   const Point ones(0,1,0);
    //   n = ones.cross(v);
    // }
    // const double a = 0.03*norm(v);
    // n /= n.norm();
    // drawline(v1, v2);
    // drawline(v2, v1 + 0.8 * v + a * n);
    // drawline(v2, v1 + 0.8 * v - a * n);
  }

  //void Pause() { char apa; std::cin >> apa; }

  //-----------------------------------------------------------------------------
  template<class U=std::size_t, class T=double>
  inline void dolfin_write_medit_triangles(const std::string &filename,
					   const dolfin::Mesh& mesh,
					   const U t = 0,
					   const std::vector<T>* u = 0)
  {
    std::stringstream ss;
    ss<<filename<<"."<<t<<".mesh";
    std::ofstream file(ss.str().c_str());
    if (!file.good()) { std::cout << "sth wrong with the file " << ss.str()<<'\n'; exit(0); }
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

    if (u)
    {
      std::stringstream ss;
      ss<<filename<<"."<<t<<".bb";
      std::ofstream file(ss.str().c_str());
      if (!file.good()) { std::cout << "sth wrong with the file " << ss.str()<<'\n'; exit(0); }
      file.precision(13);
      const std::size_t nno = mesh.num_vertices();
      const std::size_t nel = mesh.num_cells();
      // Write data (node or element based).
      if (u->size()==nno)
	file << "3 1 " << nno << " 2\n";
      else if (u->size()==nel)
	file << "3 1 " << nel << " 1\n";
      else
      {
	std::cout<<"\n\nstrange sizes u="<<u->size() << " nno=" << nno<<" nel="<<nel << ". Writing available data anyway\n"<<std::endl;
	file << "3 1 " << u->size() << " 1\n";
      }

      // Writing
      for (std::size_t i = 0; i < u->size(); ++i)
	file << (*u)[i] << '\n';
      file.close();
    }
  }

  //-----------------------------------------------------------------------------
  inline void dolfin_write_medit_triangles(const std::string& filename,
					   const dolfin::MultiMesh& mm,
					   //const std::vector<std::vector<double>> *u=0,
					   const int t=0)
  {
    std::stringstream ss;
    ss<<filename<<"."<<t<<".mesh";
    std::ofstream file(ss.str().c_str());
    if (!file.good()) { std::cout << "sth wrong with the file " << ss.str()<<'\n'; exit(0); }
    file.precision(13);

    // write vertices
    std::size_t nno=0;
    for (std::size_t i=0; i<mm.num_parts(); ++i)
      nno += mm.part(i)->num_vertices();
    file << "MeshVersionFormatted 1\nDimension\n2\nVertices\n"
	 << nno<<'\n';
    for (std::size_t i=0; i<mm.num_parts(); ++i) {
      const std::vector<double>& coords = mm.part(i)->coordinates();
      for (std::size_t j=0; j<mm.part(i)->num_vertices(); ++j)
	file << coords[2*j]<<' '<<coords[2*j+1]<<' '<<i+1<<'\n';
    }

    // write connectivity
    std::size_t nel=0;
    for (std::size_t i=0; i<mm.num_parts(); ++i)
      nel += mm.part(i)->num_cells();
    file << "Triangles\n"
	 << nel <<'\n';
    std::size_t offset = 0;
    for (std::size_t i=0; i<mm.num_parts(); ++i) {
      const std::vector<unsigned int>& cells = mm.part(i)->cells();
      offset += (i == 0) ? 0 : mm.part(i-1)->num_vertices();
      for (std::size_t e = 0; e < mm.part(i)->num_cells(); ++e)
      	file << cells[3*e]+offset+1<<' '<<cells[3*e+1]+offset+1<<' '<<cells[3*e+2]+offset+1<<' '<<i+1<<'\n';
      // for (const auto e: mm.uncut_cells(i))
      // 	file << cells[3*e]+offset+1<<' '<<cells[3*e+1]+offset+1<<' '<<cells[3*e+2]+offset+1<<' '<<i+1<<'\n';
      // for (const auto e: mm.cut_cells(i))
      // 	file << cells[3*e]+offset+1<<' '<<cells[3*e+1]+offset+1<<' '<<cells[3*e+2]+offset+1<<' '<<i+1<<'\n';

    }
    file.close();

    {
      std::stringstream ss;
      ss<<filename<<"."<<t<<".bb";
      std::ofstream file(ss.str().c_str());
      if (!file.good()) { std::cout << "sth wrong with the file " << ss.str()<<'\n'; exit(0); }
      file.precision(13);
      file << "3 1 " << nel << " 1\n";

      for (std::size_t i=0; i<mm.num_parts(); ++i)
	for (std::size_t j=0; j<mm.part(i)->num_cells(); ++j)
	  file << i+1 <<'\n';
      file.close();
    }

  }


  //-----------------------------------------------------------------------------
  // Hack to write vtu file
  inline void write_vtu_hack(const std::string& filename,
			     const dolfin::Mesh& mesh,
			     std::size_t i = 0)
  {
    std::stringstream ss;
    ss << i;
    std::ofstream fp(filename+ss.str()+".vtu");

    const std::size_t num_vertices = mesh.num_vertices();
    const std::size_t num_cells = mesh.num_cells();

    fp << "<?xml version=\"1.0\"?>\n"
       << "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\"  >\n"
       << "<UnstructuredGrid>\n"
       << "<Piece  NumberOfPoints=\"" << num_vertices << "\" NumberOfCells=\"" << num_cells << "\">\n"
       << "<Points>\n"
       << "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\">";

    // vertices
    for (dolfin::VertexIterator vertex(mesh); !vertex.end(); ++vertex) {
      for (int d = 0; d < 2; ++d) // dimension
	fp << vertex->x(d) << ' ';
      fp << "0   "; // always write 3d
    }
    fp << "</DataArray>\n"
       << "</Points>\n";

    // cells
    fp << "<Cells>\n"
       << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\"ascii\">";
    const std::vector<unsigned int>& cells = mesh.cells();
    for (std::size_t e = 0; e < num_cells; ++e) {
      // tets:
      //fp << cells[4*e] << ' ' << cells[4*e+1] << ' ' << cells[4*e+2] << ' ' << cells[4*e+3] << "  ";
      // tris:
      fp << cells[3*e]<<' '<<cells[3*e+1]<<' '<<cells[3*e+2]<<"  ";
    }
    fp << "</DataArray>\n";

    // offset
    fp << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\"ascii\">";
    for (std::size_t e = 0, offset=3; e < num_cells; ++e, offset += 3) // offset is 3 or 4
      fp << offset << ' ';
    fp << "</DataArray>\n";

    // types
    const std::size_t vtk_element_type = 5; // tet=10, tri=5
    fp << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\">";
    for (std::size_t e = 0; e < num_cells; ++e)
      fp << vtk_element_type << ' ';
    fp << "</DataArray>\n"
       << "</Cells>\n";

    // data
    fp.precision(16);
    const std::size_t size = num_vertices;
    std::vector<double> values(size, i);
    //u.compute_vertex_values(values, mesh);
    const std::string encode_string = "ascii";

    // // write velocity
    // const std::string velocity_name = u.name() + "_velocity";
    // fp << "<PointData>\n"
    //    << "<DataArray  type=\"Float64\"  Name=\"" << velocity_name << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">";
    // for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    // {
    //   for (std::size_t i = 0; i < 3; ++i) // Only write 3 components!
    // 	fp << values[vertex->index() + i*num_vertices] << " ";
    //   fp << " ";
    // }
    // fp << "</DataArray>\n";

    // // write pressure
    // const std::string pressure_name = u.name() + "_pressure";
    // fp << "<DataArray  type=\"Float64\"  Name=\"" << pressure_name << "\"  NumberOfComponents=\"1\" format=\""<< encode_string <<"\">";
    // for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    //   fp << values[vertex->index() + 3*num_vertices] << ' ';
    // fp << "</DataArray>\n"
    //    << "</PointData>\n";

    const std::string name = "data_part_"+ss.str();
    fp << "<PointData>\n"
      //<< "<DataArray  type=\"Float64\"  Name=\"" << name << "\"  NumberOfComponents=\"1\" format=\""<< encode_string <<"\">";
       << "<DataArray  type=\"Float64\"  Name=\"" << name << "\" format=\""<< encode_string <<"\">";
    for (dolfin::VertexIterator vertex(mesh); !vertex.end(); ++vertex)
      fp << values[vertex->index()] << ' ';
    fp << "</DataArray>\n"
       << "</PointData>\n";


    fp << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
    fp.close();
  }

  //-----------------------------------------------------------------------------
  inline std::string zoom(bool matlab=true)
  {
    if (matlab) return "axis equal;";
    else
      return "plt.autoscale(enable=True,axis='both',tight=None);";
  }

  //-----------------------------------------------------------------------------
  inline void writemarkers(const MultiMesh& mm,
			   std::size_t step = 0)
  {
    for (std::size_t part = 0; part < mm.num_parts(); ++part)
    {
      std::stringstream ss; ss << part;
      const std::size_t n = mm.part(part)->num_cells();
      std::vector<int> uncut(n, -1), cut(n, -1), covered(n, -1);
      for (const auto c: mm.uncut_cells(part)) uncut[c] = 0;
      for (const auto c: mm.cut_cells(part)) cut[c] = 1;
      for (const auto c: mm.covered_cells(part)) covered[c] = 2;
      dolfin_write_medit_triangles("uncut"+ss.str(),*mm.part(part),step,&uncut);
      dolfin_write_medit_triangles("cut"+ss.str(),*mm.part(part),step,&cut);
      dolfin_write_medit_triangles("covered"+ss.str(),*mm.part(part),step,&covered);
    }
    dolfin_write_medit_triangles("multimesh",mm,step);

  }

  //------------------------------------------------------------------------------
  inline double compute_volume(const MultiMesh& multimesh)
  {
    std::cout << std::endl << __FUNCTION__<< std::endl;

    double volume = 0;
    std::vector<double> all_volumes;

    std::ofstream file("quadrature_volume.txt");
    if (!file.good()) { std::cout << "file not good"<<std::endl; exit(0); }
    file.precision(20);

    // Sum contribution from all parts
    std::cout << "Sum contributions"<<std::endl;
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      std::cout << "% part " << part;
      double part_volume = 0;
      std::vector<double> status(multimesh.part(part)->num_cells(), 0);

      // Uncut cell volume given by function volume
      double uncut_volume = 0;
      const auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
	const Cell cell(*multimesh.part(part), *it);
	volume += cell.volume();
	part_volume += cell.volume();
	uncut_volume += cell.volume();
	status[*it] = 1;
      }

      std::cout << "\t uncut volume "<< uncut_volume<<' ';

      // Cut cell volume given by quadrature rule
      double cut_volume = 0;
      const auto& cut_cells = multimesh.cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
	const auto& qr = multimesh.quadrature_rule_cut_cell(part, *it);
	for (std::size_t i = 0; i < qr.second.size(); ++i)
	{
	  file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<std::endl;
	  volume += qr.second[i];
	  part_volume += qr.second[i];
	  cut_volume += qr.second[i];
	}
	status[*it] = 2;
      }

      std::cout << "\tcut volume " << cut_volume << "\ttotal volume " << part_volume << std::endl;

      all_volumes.push_back(part_volume);

      dolfin_write_medit_triangles("status",*multimesh.part(part),part,&status);
    }
    file.close();

    return volume;
  }

  //-----------------------------------------------------------------------------
  inline double compute_volume_overlap(const MultiMesh& multimesh)
  {
    std::cout << std::endl << __FUNCTION__ << std::endl;

    // Mimic MultiMeshAssembler::_assemble_overlap
    double vol = 0;

    // Iterate over parts
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      double vol_part = 0;

      // Get quadrature rules
      const auto& quadrature_rules = multimesh.quadrature_rule_overlap(part);

      // Get collision map
      const auto& cmap = multimesh.collision_map_cut_cells(part);
      // Iterate over all cut cells in collision map
      for (auto it = cmap.begin(); it != cmap.end(); ++it)
      {
	// Get cut cell
	const unsigned int cut_cell_index = it->first;
	const Cell cut_cell(*multimesh.part(part), cut_cell_index);

	// Iterate over cutting cells
	const auto& cutting_cells = it->second;
	for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
	{
	  // Get cutting part and cutting cell
	  const std::size_t cutting_part = jt->first;
	  const std::size_t cutting_cell_index = jt->second;
	  const Cell cutting_cell(*multimesh.part(cutting_part), cutting_cell_index);

	  // Get quadrature rule for interface part defined by
	  // intersection of the cut and cutting cells
	  const std::size_t k = jt - cutting_cells.begin();
	  dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
	  const auto& qr = quadrature_rules.at(cut_cell_index)[k];

	  // Skip if there are no quadrature points
	  const std::size_t num_quadrature_points = qr.second.size();

	  if (num_quadrature_points > 0)
	  {
	    for (std::size_t i = 0; i < num_quadrature_points; ++i)
	    {
	      vol_part += qr.second[i];
	      vol += qr.second[i];
	    }
	  }
	}
      }
      std::cout << " part " << part << " overlap volume = " << vol_part << std::endl;
    }
    std::cout << " total overlap volume = " << vol << std::endl;
    return vol;
  }

  //------------------------------------------------------------------------------
  inline double compute_interface_area(const MultiMesh& multimesh)
  {
    std::cout << std::endl << __FUNCTION__ << std::endl;

    double area = 0;
    std::vector<double> all_areas;

    std::ofstream file("quadrature_interface.txt");
    if (!file.good()) { std::cout << "file not good"<<std::endl; exit(0); }
    file.precision(20);

    // Sum contribution from all parts
    std::cout << "Sum contributions"<<std::endl;
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      std::cout << "% part " << part << ' ';
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
	    //std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<std::endl;
	  }
	  //std::cout << std::endl;
	}
      }
      std::cout << "total area " << part_area << std::endl;
      all_areas.push_back(part_area);
    }
    file.close();

    return area;
  }


  //------------------------------------------------------------------------------
  inline void plot_normals(const MultiMesh& multimesh)
  {
    std::cout << std::endl << __FUNCTION__ << std::endl;
    const std::vector<std::string> colors = {{ "'b'", "'g'", "'r'" }};
    const std::vector<std::string> marker = {{ "'.'", "'o'", "'x'" }};

    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
      // const std::size_t part = 1;
    {
      std::cout << "% part " << part << ' ' <<std::endl;
      const auto& cmap = multimesh.collision_map_cut_cells(part);
      const auto& qr_interface = multimesh.quadrature_rule_interface(part);
      const auto& normals = multimesh.facet_normals(part);

      for (auto it = cmap.begin(); it != cmap.end(); ++it)
      {
	const unsigned int cut_cell_index = it->first;
	const auto& cutting_cells = it->second;

	const Cell cut_cell(*multimesh.part(part), cut_cell_index);
	std::cout << drawtriangle(cut_cell, colors[part]);

	const auto& qr = multimesh.quadrature_rule_cut_cell(part, cut_cell_index);
	cout_qr(qr, colors[part]);

	// Iterate over cutting cells
	for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
	{
	  const std::size_t cutting_cell_part = jt->first;

	  const Cell cutting_cell(*multimesh.part(cutting_cell_part), jt->second);
	  std::cout << drawtriangle(cutting_cell, colors[cutting_cell_part]);

	  // Get quadrature rule for interface part defined by
	  // intersection of the cut and cutting cells
	  const std::size_t k = jt - cutting_cells.begin();
	  const auto& qr = qr_interface.at(cut_cell_index)[k];
	  const auto& nn = normals.at(cut_cell_index)[k];

	  for (std::size_t i = 0; i < qr.second.size(); ++i)
	  {
	    const Point p(qr.first[2*i], qr.first[2*i+1]);
	    std::cout << plot(p,"'k.','markersize',12");
	    const Point n(nn[2*i], nn[2*i+1]);
	    const double d = 0.1;
	    std::cout << drawarrow(p, p+d*n, colors[cutting_cell_part]);
	  }
	}
	std::cout << std::endl;
      }

      // for (const auto cell_no: multimesh.cut_cells(part))
      // {
      //   const auto qrmap = multimesh.quadrature_rule_interface(part).find(cell_no);
      //   const std::vector<quadrature_rule> qr = qrmap->second;

      //   const auto fnmap = multimesh.facet_normals(part).find(cell_no);
      //   const std::vector<std::vector<double>> normals = fnmap->second;

      //   //std::cout << qr.size() << ' ' << normals.size() << std::endl;
      //   dolfin_assert(qr.size() == normals.size());

      //   for (std::size_t i = 0; i < qr.size(); ++i)
      //   {
      // 	for (std::size_t j = 0; j < qr[i].second.size(); ++j)
      // 	{
      // 	  const Point p(qr[i].first[2*j], qr[i].first[2*j+1]);
      // 	  std::cout << plot(p,"'k.'");
      // 	  const Point n(normals[i][2*j],normals[i][2*j+1]);
      // 	  const double d = 0.01;
      // 	  std::cout << drawarrow(p, p+d*n);
      // 	}
      // 	std::cout << std::endl;
      //   }
      // }

    }
  }


  //------------------------------------------------------------------------------
  inline void evaluate_at_qr(const MultiMesh& mm,
			     const Expression& uexact,
			     const MultiMeshFunction& uh)
  {
    std::cout << __FUNCTION__ << std::endl;
    double maxee = -1;

    for (std::size_t part = 0; part < mm.num_parts(); ++part)
    {
      std::cout << "\npart " << part << std::endl;

      // get vertex values
      std::vector<double> vertex_values;
      uh.part(part)->compute_vertex_values(vertex_values, *mm.part(part));

      const std::vector<std::string> colors = {{ "'b'", "'g'", "'r'" }};
      std::vector<std::size_t> cells;

      // cells colliding with the cut cells
      const auto collision_map = mm.collision_map_cut_cells(part);

      // loop over cut cells
      for (const auto cut_cell_no: mm.cut_cells(part))
      {
	// all qr on cut_cell_no
	const auto qr = mm.quadrature_rule_cut_cell(part, cut_cell_no);

	// loop over qr
	for (std::size_t i = 0; i < qr.second.size(); ++i)
	{
	  const Point p(qr.first[2*i], qr.first[2*i+1]);
	  const double uhval = (*uh.part(part))(p);
	  const double uexactval = uexact(p);
	  const double ee = std::abs(uhval-uexactval);
	  maxee = ee > maxee ? ee : maxee;
	  std::cout << p.x()<<' '<<p.y()<<' '<<uhval<<' '<<uexactval<<' '<<ee<<' '<<maxee<<std::endl;

	  //   // if evaluated function big...
	  //   if (std::abs(uhval) > 1)
	  //   {
	  //     // save cell no
	  //     cells.push_back(cut_cell_no);
	  //     const std::string color = qr.second[i] > 0 ? "'.'" : "'x'";
	  //     std::cout << matlabplot(p,color) <<" % " << qr.second[i] << ' '
	  // 	      << /\*std::setprecision(15) <<*\/ uhval << " (";

	  //     // print nodal uh values
	  //     const Cell cell(*mm.part(part), cut_cell_no);
	  //     for (std::size_t j = 0; j < cell.num_vertices(); ++j)
	  //       std::cout << cell.entities(0)[j] << ' '<<vertex_values[cell.entities(0)[j]] <<' ';
	  //     std::cout << ")"<<std::endl;
	  //   }
	}
      }

      // // make cell numbers unique
      // std::sort(cells.begin(), cells.end());
      // const auto new_end = std::unique(cells.begin(), cells.end());
      // cells.erase(new_end, cells.end());

      // // loop over all cells with large uh values
      // for (const auto cell_no: cells)
      // {
      // 	std::cout << "% cell with large uh:"<<std::endl;
      // 	const Cell cell(*mm.part(part), cell_no);
      // 	std::cout << drawtriangle(cell);

      // 	// compute net weight (~visible area)
      // 	const auto qr = mm.quadrature_rule_cut_cell(part, cell_no);
      // 	double net_weight = 0;
      // 	std::cout << " % ";
      // 	for (const auto w: qr.second)
      // 	{
      // 	  net_weight += w;
      // 	  std::cout << ' '<<w;
      // 	}
      // 	std::cout << "\n% net weight = " << net_weight << std::endl;

      // 	// also display all colliding cells
      // 	const auto it = collision_map.find(cell_no);
      // 	dolfin_assert(it->first == cell_no);
      // 	std::cout << "% colliding:"<<std::endl;
      // 	for (const auto cpair: it->second)
      // 	{
      // 	  const Cell cutting_cell(*mm.part(cpair.first), cpair.second);
      // 	  std::cout << drawtriangle(cutting_cell,colors[cpair.first]);
      // 	}
      // }

    }
    std::cout << "max error in qr points " << maxee << std::endl;
    PPause;
  }

  //------------------------------------------------------------------------------
  template<class TFunctionSpace>// eg P1::FunctionSpace
  inline void find_max(const MultiMesh& multimesh,
		       const MultiMeshFunction& u,
		       std::vector<double>& maxvals_parts,
		       std::size_t step = 0
		       // ,
		       // File& uncut0_file, File& uncut1_file, File& uncut2_file,
		       // File& cut0_file, File& cut1_file, File& cut2_file,
		       // File& covered0_file, File& covered1_file, File& covered2_file
		       )

  {
    std::cout << __FUNCTION__ << std::endl;
    std::cout << "\tSolution: max min step " << step <<' ' << u.vector()->max() << ' ' << u.vector()->min() << std::endl;

    maxvals_parts.assign(multimesh.num_parts(), -9e99);

    for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
    {
      // get max on vertex values
      std::vector<double> vertex_values;
      u.part(part)->compute_vertex_values(vertex_values,
					  *multimesh.part(part));
      const double maxvv = *std::max_element(vertex_values.begin(),
					     vertex_values.end());

      // get max on uncut, cut and covered
      const std::vector<std::vector<unsigned int>> cells
	= {{ multimesh.uncut_cells(part),
	     multimesh.cut_cells(part),
	     multimesh.covered_cells(part) }};
      const std::vector<std::string> type = {{ "uncut", "cut", "covered" }};
      std::vector<double> maxvals(cells.size(), 0);

      for (std::size_t k = 0; k < cells.size(); ++k)
      {
	std::cout << "part " << part << " "<<k << ' '<<type[k]<< std::endl;
	if (cells[k].size())
	{
	  // Create meshfunction using markers
	  auto mesh_part = std::make_shared<Mesh>(*multimesh.part(part));
	  auto foo = std::make_shared<MeshFunction<std::size_t> >(mesh_part, mesh_part->topology().dim());
	  foo->set_all(0); // dummy
	  for (const auto cell: cells[k])
	    foo->set_value(cell, k+1);

	  // Create submesh out of meshfunction
	  auto sm = std::make_shared<SubMesh>(*multimesh.part(part), *foo, k+1);

	  // Interpolate on submesh
	  auto V = std::make_shared<TFunctionSpace>(sm);
	  auto usm = std::make_shared<Function>(V);

	  // test
	  usm->set_allow_extrapolation(true);

	  usm->interpolate(*u.part(part));

	  // Get max values on submesh
	  std::vector<double> vertex_values;
	  usm->compute_vertex_values(vertex_values);
	  maxvals[k] = *std::max_element(vertex_values.begin(), vertex_values.end());

	  // if (part == 0)
	  //   if (k == 0 or k == 1) {
	  //     std::cout << k <<std::endl;
	  //     for (const auto cell: cells[k])
	  // 	std::cout << cell << ' ';
	  //     std::cout << std::endl;
	  //   }

	  // if (marker == 1 and part == 0) {
	  //   for (const auto v: vertex_values)
	  //     std::cout << v<<' ';
	  //   std::cout << std::endl;
	  // }

	  // // save
	  // switch(k) {
	  // case 0: { // uncut
	  //   if (part == 0) uncut0_file << (*usm);
	  //   else if (part == 1) uncut1_file << (*usm);
	  //   else if (part == 2) uncut2_file << (*usm);
	  //   break;
	  // }
	  // case 1: { // cut
	  //   if (part == 0) cut0_file << (*usm);
	  //   else if (part == 1) cut1_file << (*usm);
	  //   else if (part == 2) cut2_file << (*usm);
	  //   break;
	  // }
	  // case 2: { // covered
	  //   if (part == 0) covered0_file << (*usm);
	  //   else if (part == 1) covered1_file << (*usm);
	  //   else if (part == 2) covered2_file << (*usm);
	  // }
	  // }
	}
      }

      std::cout << "\tpart " << part
		<< " step " << step
		<< " all vertices " << maxvv
		<< " uncut " << maxvals[0]
		<< " cut " << maxvals[1]
		<< " covered " << maxvals[2] << std::endl;

      maxvals_parts[part] = std::max(std::max(maxvals[0], maxvals[1]), maxvals[2]);
      //if (maxvals[0] < 1) { exit(0); }
    }

  }
  //------------------------------------------------------------------------------



}

#endif
