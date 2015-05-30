#ifndef DOLFIN_WRITE_MEDIT_H
#define DOLFIN_WRITE_MEDIT_H

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <dolfin.h>

namespace medit
{
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
    return 0.25*std::sqrt( (a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)) );
  }



  inline double area(const std::vector<dolfin::Point> &tri)
  {
    const dolfin::Point et=tri[1]-tri[0];
    const dolfin::Point es=tri[2]-tri[0];
    return Heron(et.norm(), es.norm(), (et-es).norm());
  }


  inline std::string drawtriangle(const std::vector<dolfin::Point> &tri,
				  const std::string& color = "'b'")
  {
    std::stringstream ss;
    ss << "drawtriangle("
       << "["<<tri[0][0]<<' '<<tri[0][1]<<"],"
       << "["<<tri[1][0]<<' '<<tri[1][1]<<"],"
       << "["<<tri[2][0]<<' '<<tri[2][1]<<"],"
       << color << ");";
    return ss.str();
  }

  inline std::string drawtriangle(const dolfin::Cell &cell,
				  const std::string& color = "'b'")
  {
    std::vector<dolfin::Point> tri(3);
    for (int i = 0; i < 3; ++i)
      tri[i] = cell.mesh().geometry().point(cell.entities(0)[i]);
    return drawtriangle(tri, color);
  }

  inline std::string matlabplot(const dolfin::Point& p,
				const std::string m="'.'")
  {
    std::stringstream ss;
    ss<<"plot("<<p[0]<<','<<p[1]<<','<<m<<");";
    return ss.str();
  }

#define PPause {char dummycharXohs5su8='a';std::cout<<"\n Pause: "<<__FILE__<<" line "<<__LINE__<<" function "<<__FUNCTION__<<std::endl;std::cin>>dummycharXohs5su8;}

  //void Pause() { char apa; std::cin >> apa; }

  inline void dolfin_write_medit_triangles(const std::string &filename,
					   const dolfin::Mesh& mesh,
					   const int t = 0)
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
  }

  inline void dolfin_write_medit_triangles(const std::string& filename,
					   const dolfin::MultiMesh& mm,
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
    std::size_t offset=-mm.part(0)->num_vertices();
    for (std::size_t i=0; i<mm.num_parts(); ++i) {
      const std::vector<unsigned int>& cells = mm.part(i)->cells();
      offset+=mm.part(i)->num_vertices();
      for (std::size_t e = 0; e < mm.part(i)->num_cells(); ++e)
	file << cells[3*e]+offset+1<<' '<<cells[3*e+1]+offset+1<<' '<<cells[3*e+2]+offset+1<<' '<<i+1<<'\n';
    }
    file.close();
  }



  /* // Hack to write vtu file */
  /* inline void write_vtu_hack(const std::string& filename, */
  /* 			     const Mesh& mesh, */
  /* 			     std::size_t i = 0) */
  /* { */
  /*   std::stringstream ss; */
  /*   ss << i; */
  /*   std::ofstream fp(filename+ss.str()+".vtk"); */

  /*   const std::size_t num_vertices = mesh.num_vertices(); */
  /*   const std::size_t num_cells = mesh.num_cells(); */

  /*   fp << "<?xml version=\"1.0\"?>\n" */
  /*      << "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\"  >\n" */
  /*      << "<UnstructuredGrid>\n" */
  /*      << "<Piece  NumberOfPoints=\"" << num_vertices << "\" NumberOfCells=\"" << num_cells << "\">\n" */
  /*      << "<Points>\n" */
  /*      << "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\">"; */

  /*   // vertices */
  /*   for (VertexIterator vertex(mesh); !vertex.end(); ++vertex) */
  /*     for (int d = 0; d < 3; ++d) */
  /* 	fp << vertex->x(d) << ' '; */
  /*   fp << "</DataArray>\n" */
  /*      << "</Points>\n"; */

  /*   // cells */
  /*   fp << "<Cells>\n" */
  /*      << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\"ascii\">"; */
  /*   const std::vector<unsigned int>& cells = mesh.cells(); */
  /*   for (std::size_t e = 0; e < num_cells; ++e) */
  /*     fp << cells[4*e] << ' ' << cells[4*e+1] << ' ' << cells[4*e+2] << ' ' << cells[4*e+3] << "  "; */
  /*   fp << "</DataArray>\n"; */

  /*   // offset */
  /*   fp << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\"ascii\">"; */
  /*   for (std::size_t e = 0, offset=4; e < num_cells; ++e, offset += 4) */
  /*     fp << offset << ' '; */
  /*   fp << "</DataArray>\n"; */

  /*   // types */
  /*   const std::size_t vtk_element_type = 10; */
  /*   fp << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\">"; */
  /*   for (std::size_t e = 0; e < num_cells; ++e) */
  /*     fp << vtk_element_type << ' '; */
  /*   fp << "</DataArray>\n" */
  /*      << "</Cells>\n"; */

  /*   // data */
  /*   fp.precision(16); */
  /*   const std::size_t size = num_vertices; */
  /*   std::vector<double> values(size, i); */
  /*   //u.compute_vertex_values(values, mesh); */
  /*   const std::string encode_string = "ascii"; */

  /*   /\* // write velocity *\/ */
  /*   /\* const std::string velocity_name = u.name() + "_velocity"; *\/ */
  /*   /\* fp << "<PointData>\n" *\/ */
  /*   /\*    << "<DataArray  type=\"Float64\"  Name=\"" << velocity_name << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">"; *\/ */
  /*   /\* for (VertexIterator vertex(mesh); !vertex.end(); ++vertex) *\/ */
  /*   /\* { *\/ */
  /*   /\*   for (std::size_t i = 0; i < 3; ++i) // Only write 3 components! *\/ */
  /*   /\* 	fp << values[vertex->index() + i*num_vertices] << " "; *\/ */
  /*   /\*   fp << " "; *\/ */
  /*   /\* } *\/ */
  /*   /\* fp << "</DataArray>\n"; *\/ */

  /*   /\* // write pressure *\/ */
  /*   /\* const std::string pressure_name = u.name() + "_pressure"; *\/ */
  /*   /\* fp << "<DataArray  type=\"Float64\"  Name=\"" << pressure_name << "\"  NumberOfComponents=\"1\" format=\""<< encode_string <<"\">"; *\/ */
  /*   /\* for (VertexIterator vertex(mesh); !vertex.end(); ++vertex) *\/ */
  /*   /\*   fp << values[vertex->index() + 3*num_vertices] << ' '; *\/ */
  /*   /\* fp << "</DataArray>\n" *\/ */
  /*   /\*    << "</PointData>\n"; *\/ */

  /*   const std::string name = "data_part_"+ss.str(); */
  /*   fp << "<DataArray  type=\"Float64\"  Name=\"" << name << "\"  NumberOfComponents=\"1\" format=\""<< encode_string <<"\">"; */
  /*   for (VertexIterator vertex(mesh); !vertex.end(); ++vertex) */
  /*     fp << values[vertex->index()] << ' '; */
  /*   fp << "</DataArray>\n" */
  /*      << "</PointData>\n"; */


  /*   fp << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"; */
  /*   fp.close(); */
  /* } */


}

#endif
