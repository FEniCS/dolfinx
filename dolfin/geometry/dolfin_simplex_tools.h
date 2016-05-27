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
#include <dolfin/mesh/Cell.h>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>

namespace tools
{
#define PPause {char dummycharXohs5su8='a';std::cout<<"\n Pause: "<<__FILE__<<" line "<<__LINE__<<" function "<<__FUNCTION__<<std::endl;std::cin>>dummycharXohs5su8;}

  /* typedef std::vector<Point> Simplex; */
  /* typedef std::vector<Simplex> Polyhedron; */
  inline bool tdimcheck(const std::vector<std::vector<dolfin::Point>>& polygon)
  {
    if (polygon.size() == 0) return false;

    const std::size_t tdimtmp = polygon[0].size();
    for (std::size_t i = 1; i < polygon.size(); ++i)
      if (polygon.at(i).size() != tdimtmp)
	return false;
    return true;
  }

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




  // display quadrature_rule
  // recall typedef std::pair<std::vector<double>, std::vector<double> > quadrature_rule;
  inline void cout_qr(const std::pair<std::vector<double>, std::vector<double> >& qr,
		      const std::string marker="'rx'")
  {
    for (std::size_t i = 0; i < qr.second.size(); ++i)
    {
      std::cout << "plot("<<qr.first[2*i]<<','<<qr.first[2*i+1]<<','<<marker<<"); % "<<qr.second[i]<<' '<<i<<std::endl;
    }
  }

  inline void cout_normals(const std::vector<double>& n)
  {
    for (std::size_t i = 0; i < n.size()/2; ++i)
      std::cout << i << ":  "<<n[2*i]<<' '<<n[2*i+1]<<std::endl;
  }

  inline double area(const std::pair<std::vector<double>, std::vector<double> >& qr)
  {
    double a = 0;
    for (std::size_t i = 0; i < qr.second.size(); ++i)
      a += qr.second[i];
    return a;
  }

  // this sorts such that a >= b >= c
  template<class T>
  inline void sort3(T &a, T &b, T &c)
  {
    if (b>a) std::swap(b,a);
    if (c>b) std::swap(c,b);
    if (b>a) std::swap(b,a);
  }

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

    /* if (simplex.size() == 3) */
    /* { */
    /*   double a[2]={simplex[0][0],simplex[0][1]}; */
    /*   double b[2]={simplex[1][0],simplex[1][1]}; */
    /*   double c[2]={simplex[2][0],simplex[2][1]}; */
    /*   return 0.5*orient2d(a,b,c); */
    /* } */
    /* else if (simplex.size() == 2) */
    /* { */
    /*   return (simplex[0]-simplex[1]).norm(); */
    /* } */
    /* else */
    /* { */
    /*   PPause; */
    /*   return -9e99; */
    /* } */
  }

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
      /* ss << "hline = line([" << simplex[0][0] << ',' << simplex[1][0] << "]," */
      /* 	 << "[" << simplex[0][1] << ',' << simplex[1][1] << "]);" */
      /* 	 << "set(hline,'color'," << color << ");"; */
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

  inline std::string drawtriangle(const dolfin::Cell &cell,
				  const std::string& color = "'b'",
				  bool matlab=true)
  {
    const std::size_t tdim = cell.mesh().topology().dim();
    std::vector<dolfin::Point> tri(tdim+1);
    for (std::size_t i = 0; i < tdim+1; ++i)
      tri[i] = cell.mesh().geometry().point(cell.entities(0)[i]);
    return drawtriangle(tri, color, matlab);
  }

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

    /* std::vector<dolfin::Point> ss(3); */
    /* ss[0] = dolfin::Point(s[0],s[1]); */
    /* ss[1] = dolfin::Point(s[2],s[3]); */
    /* ss[2] = dolfin::Point(s[4],s[5]); */
    /* return drawtriangle(ss, color); */
  }

  inline std::string drawtriangle(const dolfin::Point& a,
				  const dolfin::Point& b,
				  const dolfin::Point& c,
				  const std::string color = "'b'",
				  bool matlab = false)
  {
    std::vector<dolfin::Point> t = {{ a, b, c}};
    return drawtriangle(t, color);
  }

  inline std::string matlabplot(const dolfin::Point& p,
				const std::string m="'k.','markersize',14")
  {
    std::stringstream ss; ss.precision(15);
    ss<<"plot("<<p[0]<<','<<p[1]<<','<<m<<");";
    return ss.str();
  }

  inline std::string plot(const dolfin::Point& p,
			  const std::string m="'k.','markersize',14")
  {
    return matlabplot(p,m);
  }

  inline std::string drawarrow(const dolfin::Point& v1,
			       const dolfin::Point& v2,
			       const std::string& color = "'b'")
  {
    std::stringstream ss;ss.precision(15);
    ss << "drawarrow([" << v1[0] << ' '<<v1[1] <<"],[" << v2[0]<<' '<<v2[1] << "], "<< color << ");";
    return ss.str();
    /* const dolfin::Point v = v2-v1; */
    /* const Point ones(0,0,1); */
    /* Point n = ones.cross(v); */
    /* if (n.norm() < 1e-5) { */
    /*   const Point ones(0,1,0); */
    /*   n = ones.cross(v); */
    /* } */
    /* const double a = 0.03*norm(v); */
    /* n /= n.norm(); */
    /* drawline(v1, v2); */
    /* drawline(v2, v1 + 0.8 * v + a * n); */
    /* drawline(v2, v1 + 0.8 * v - a * n); */
  }

  //void Pause() { char apa; std::cin >> apa; }

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
      /* for (const auto e: mm.uncut_cells(i)) */
      /* 	file << cells[3*e]+offset+1<<' '<<cells[3*e+1]+offset+1<<' '<<cells[3*e+2]+offset+1<<' '<<i+1<<'\n'; */
      /* for (const auto e: mm.cut_cells(i)) */
      /* 	file << cells[3*e]+offset+1<<' '<<cells[3*e+1]+offset+1<<' '<<cells[3*e+2]+offset+1<<' '<<i+1<<'\n'; */

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

    /* // write velocity */
    /* const std::string velocity_name = u.name() + "_velocity"; */
    /* fp << "<PointData>\n" */
    /*    << "<DataArray  type=\"Float64\"  Name=\"" << velocity_name << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">"; */
    /* for (VertexIterator vertex(mesh); !vertex.end(); ++vertex) */
    /* { */
    /*   for (std::size_t i = 0; i < 3; ++i) // Only write 3 components! */
    /* 	fp << values[vertex->index() + i*num_vertices] << " "; */
    /*   fp << " "; */
    /* } */
    /* fp << "</DataArray>\n"; */

    /* // write pressure */
    /* const std::string pressure_name = u.name() + "_pressure"; */
    /* fp << "<DataArray  type=\"Float64\"  Name=\"" << pressure_name << "\"  NumberOfComponents=\"1\" format=\""<< encode_string <<"\">"; */
    /* for (VertexIterator vertex(mesh); !vertex.end(); ++vertex) */
    /*   fp << values[vertex->index() + 3*num_vertices] << ' '; */
    /* fp << "</DataArray>\n" */
    /*    << "</PointData>\n"; */

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

  inline std::string zoom(bool matlab=true)
  {
    if (matlab) return "axis equal;";
    else
      return "plt.autoscale(enable=True,axis='both',tight=None);";
  }



  inline bool is_degenerate(std::vector<dolfin::Point> s)
  {
    bool is_degenerate = false;
    switch (s.size())
    {
    case 4:
      std::cout << "not implemented\n";
      PPause;
      break;
    case 3:
      is_degenerate = orient2d(s[0].coordinates(),
			       s[1].coordinates(),
			       s[2].coordinates()) == 0;
      break;
    case 2:
      {
	double r[2] = { dolfin::rand(), dolfin::rand() };
	is_degenerate = orient2d(s[0].coordinates(), s[1].coordinates(), r) == 0;
	break;
      }
    case 1:
      is_degenerate = true;
      break;
    default: { PPause; }
    }

    if (is_degenerate)
    {
      std::cout << drawtriangle(s)<<" % is degenerate (s.size() = "<<s.size()
		<<" volume = " << area(s) << std::endl;
      //PPause;
    }
    return is_degenerate;
  }

}

#endif
