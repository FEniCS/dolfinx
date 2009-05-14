// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2005-2006.
// Modified by Kristian Oelgaard 2006.
// Modified by Niclas Jansson 2008.
// Modified by Martin Alnes 2008.
//
// First added:  2005-07-05
// Last changed: 2008-12-15

#include <sstream>
#include <fstream>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Vector.h>
#include "PVTKFile.h"


using namespace dolfin;

//----------------------------------------------------------------------------
PVTKFile::PVTKFile(const std::string filename) : GenericFile(filename)
{
  type = "VTK";
}
//----------------------------------------------------------------------------
PVTKFile::~PVTKFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(const Mesh& mesh)
{
  // Update vtu file name and clear file
  vtu_name_update(counter);

  // Only the root updates the pvd file
  if (MPI::process_number() == 0)
  {
    // Update pvtu file name and clear file
    pvtu_name_update(counter);

    // Write pvd file
    pvd_file_write(counter);

    // Write pvtu file
    pvtu_file_write();
  }

  // Write headers
  vtk_header_open(mesh);

  // Write mesh
  mesh_write(mesh);

  // Close headers
  vtk_header_close();

  // Increase the number of times we have saved the mesh
  counter++;

  info(1, "Saved mesh %s (%s) to file %s in VTK format.",
          mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(const MeshFunction<int>& meshfunction)
{
  mesh_function_write(meshfunction);
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(const MeshFunction<unsigned int>& meshfunction)
{
  mesh_function_write(meshfunction);
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(const MeshFunction<double>& meshfunction)
{
  mesh_function_write(meshfunction);
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(const Function& u)
{
  // Update vtu file name and clear file
  vtu_name_update(counter);

  // Write pvd file

  // Only the root updates the pvd file
  if (MPI::process_number() == 0)
  {
    // Update pvtu file name and clear file
    pvtu_name_update(counter);

    // Write pvd file
    pvd_file_write(counter);

    // Write pvtu file
    pvtu_file_write_func(u);
  }

  // FIXME: need to fix const-ness
  Mesh& mesh = const_cast<Mesh&>(u.function_space().mesh());

  // Write headers
  vtk_header_open(mesh);

  // Write Mesh
  mesh_write(mesh);

  // Write results
  results_write(u);

  // Close headers
  vtk_header_close();

  // Increase the number of times we have saved the function
  counter++;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in VTK format." << endl;

}
//----------------------------------------------------------------------------
void PVTKFile::mesh_write(const Mesh& mesh) const
{
  // Open file
  FILE* fp = fopen(vtu_filename.c_str(), "a");
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  // Write vertex positions
  fprintf(fp, "<Points>  \n");
  fprintf(fp, "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\">  \n");
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    fprintf(fp," %f %f %f \n", p.x(), p.y(), p.z());
  }
  fprintf(fp, "</DataArray>  \n");
  fprintf(fp, "</Points>  \n");

  // Write cell connectivity
  fprintf(fp, "<Cells>  \n");
  fprintf(fp, "<DataArray  type=\"Int32\"  Name=\"connectivity\"  format=\"ascii\">  \n");
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (VertexIterator v(*c); !v.end(); ++v)
      fprintf(fp," %8u ",v->index());
    fprintf(fp," \n");
  }
  fprintf(fp, "</DataArray> \n");

  // Write offset into connectivity array for the end of each cell
  fprintf(fp, "<DataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\">  \n");
  for (uint offsets = 1; offsets <= mesh.num_cells(); offsets++)
  {
    if (mesh.type().cell_type() == CellType::tetrahedron )
      fprintf(fp, " %8u \n",  offsets*4);
    if (mesh.type().cell_type() == CellType::triangle )
      fprintf(fp, " %8u \n", offsets*3);
    if (mesh.type().cell_type() == CellType::interval )
      fprintf(fp, " %8u \n",  offsets*2);
  }
  fprintf(fp, "</DataArray> \n");

  //Write cell type
  fprintf(fp, "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\">  \n");
  for (uint types = 1; types <= mesh.num_cells(); types++)
  {
    if (mesh.type().cell_type() == CellType::tetrahedron )
      fprintf(fp, " 10 \n");
    if (mesh.type().cell_type() == CellType::triangle )
      fprintf(fp, " 5 \n");
    if (mesh.type().cell_type() == CellType::interval )
      fprintf(fp, " 3 \n");
  }
  fprintf(fp, "</DataArray> \n");
  fprintf(fp, "</Cells> \n");

  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::results_write(const Function& u) const
{
  // For brevity
  const FunctionSpace & V = u.function_space();
  const Mesh& mesh(V.mesh());
  const FiniteElement& element(V.element());

  // Get rank of Function
  const uint rank = element.value_rank();
  if(rank > 2)
    error("Only scalar, vector and tensor functions can be saved in VTK format.");

  // Get number of components
  uint dim = 1;
  for (uint i = 0; i < rank; i++)
    dim *= element.value_dimension(i);

  // FIXME: Cell based data.

  // Allocate memory for function values at vertices
  const uint size = mesh.num_vertices()*dim;
  double* values = new double[size];

  // Get function values at vertices
  u.interpolate(values);

  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);

  // Write function data at mesh vertices
  if (rank == 0)
  {
    fp << "<PointData  Scalars=\"U\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"U\"  format=\"ascii\"> " << std::endl;
  }
  else if (rank == 1)
  {
    if(!(dim == 2 || dim == 3))
      error("don't know what to do with vector function with dim other than 2 or 3.");
    fp << "<PointData  Vectors=\"U\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"U\"  NumberOfComponents=\"3\" format=\"ascii\"> " << std::endl;
  }
  else if (rank == 2)
  {
    if(!(dim == 4 || dim == 9))
      error("Don't know what to do with tensor function with dim other than 4 or 9.");
    fp << "<PointData  Tensors=\"U\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"U\"  NumberOfComponents=\"9\" format=\"ascii\">    " << std::endl;
  }

  std::ostringstream ss;
  ss << std::scientific;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    ss.str("");

    if(rank == 1 && dim == 2)
    {
      // Append 0.0 to 2D vectors to make them 3D
      for(uint i=0; i<dim; i++)
        ss << " " << values[vertex->index() + i*mesh.num_vertices()];
      ss << " " << 0.0;
    }
    else if(rank == 2 && dim == 4)
    {
      // Pad with 0.0 to 2D tensors to make them 3D
      for(uint i=0; i<2; i++)
      {
        ss << " " << values[vertex->index() + (2*i+0)*mesh.num_vertices()];
        ss << " " << values[vertex->index() + (2*i+1)*mesh.num_vertices()];
        ss << " " << 0.0;
      }
      ss << " " << 0.0;
      ss << " " << 0.0;
      ss << " " << 0.0;
    }
    else
    {
      // Write all components
      for(uint i=0; i<dim; i++)
        ss << " " << values[vertex->index() + i*mesh.num_vertices()];
    }
    ss << std::endl;

    fp << ss.str();
  }
  fp << "</DataArray> " << std::endl;
  fp << "</PointData> " << std::endl;

  delete [] values;
}
//----------------------------------------------------------------------------
void PVTKFile::pvd_file_write(uint num)
{
  std::fstream pvd_file;

  if( num == 0)
  {
    // Open pvd file
    pvd_file.open(filename.c_str(), std::ios::out|std::ios::trunc);
    // Write header
    pvd_file << "<?xml version=\"1.0\"?> " << std::endl;
    pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\" > " << std::endl;
    pvd_file << "<Collection> " << std::endl;
  }
  else
  {
    // Open pvd file
    pvd_file.open(filename.c_str(),  std::ios::out|std::ios::in);
    pvd_file.seekp(mark);

  }
  // Remove directory path from name for pvd file
  std::string fname;
  fname.assign(pvtu_filename, filename.find_last_of("/") + 1, pvtu_filename.size());

  // Data file name
  pvd_file << "<DataSet timestep=\"" << num << "\" part=\"0\"" << " file=\"" <<  fname <<  "\"/>" << std::endl;
  mark = pvd_file.tellp();

  // Close headers
  pvd_file << "</Collection> " << std::endl;
  pvd_file << "</VTKFile> " << std::endl;

  // Close file
  pvd_file.close();

}
//----------------------------------------------------------------------------
void PVTKFile::pvtu_file_write()
{
  std::fstream pvtu_file;

  // Open pvtu file
  pvtu_file.open(pvtu_filename.c_str(), std::ios::out|std::ios::trunc);
  // Write header
  pvtu_file << "<?xml version=\"1.0\"?> " << std::endl;
  pvtu_file << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\">" << std::endl;
  pvtu_file << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;

  pvtu_file << "<PCellData>" << std::endl;
  pvtu_file << "<PDataArray  type=\"Int32\"  Name=\"connectivity\"  format=\"ascii\"/>" << std::endl;
  pvtu_file << "<PDataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\"/>" << std::endl;
  pvtu_file << "<PDataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\"/>"  << std::endl;
  pvtu_file<<"</PCellData>" << std::endl;

  pvtu_file << "<PPoints>" <<std::endl;
  pvtu_file << "<PDataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\"/>" << std::endl;
  pvtu_file << "</PPoints>" << std::endl;

  // Remove rank from vtu filename ( <rank>.vtu)
  std::string fname;
  fname.assign(vtu_filename, filename.find_last_of("/") + 1, vtu_filename.size() - 5 );
  for(uint i=0; i< MPI::num_processes(); i++)
    pvtu_file << "<Piece Source=\"" << fname << i << ".vtu\"/>" << std::endl;

  pvtu_file << "</PUnstructuredGrid>" << std::endl;
  pvtu_file << "</VTKFile>" << std::endl;
  pvtu_file.close();

}//----------------------------------------------------------------------------
void PVTKFile::pvtu_file_write_func(const Function& u)
{
  std::fstream pvtu_file;

  // Open pvtu file
  pvtu_file.open(pvtu_filename.c_str(), std::ios::out|std::ios::trunc);
  // Write header
  pvtu_file << "<?xml version=\"1.0\"?> " << std::endl;
  pvtu_file << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\">" << std::endl;
  pvtu_file << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;

  if(u.function_space().element().value_rank() == 0) {
    pvtu_file << "<PPointData Scalars=\"U\">" << std::endl;
    pvtu_file << "<PDataArray  type=\"Float64\"  Name=\"U\"  format=\"ascii\"/>" << std::endl;
  }
  else {
    pvtu_file << "<PPointData Vectors=\"U\">" << std::endl;
    pvtu_file << "<PDataArray  type=\"Float64\"  Name=\"U\"  NumberOfComponents=\"3\" format=\"ascii\"/>" << std::endl;
    }

  pvtu_file << "</PPointData>" << std::endl;

  pvtu_file << "<PCellData>" << std::endl;
  pvtu_file << "<PDataArray  type=\"Int32\"  Name=\"connectivity\"  format=\"ascii\"/>" << std::endl;
  pvtu_file << "<PDataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\"/>" << std::endl;
  pvtu_file << "<PDataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\"/>"  << std::endl;
  pvtu_file<<"</PCellData>" << std::endl;

  pvtu_file << "<PPoints>" <<std::endl;
  pvtu_file << "<PDataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\"/>" << std::endl;
  pvtu_file << "</PPoints>" << std::endl;

  std::string fname;
  // Remove rank from vtu filename ( <rank>.vtu)
  fname.assign(vtu_filename, filename.find_last_of("/") + 1, vtu_filename.size() - 5 );
  for(uint i=0; i< MPI::num_processes(); i++)
    pvtu_file << "<Piece Source=\"" << fname << i << ".vtu\"/>" << std::endl;


  pvtu_file << "</PUnstructuredGrid>" << std::endl;
  pvtu_file << "</VTKFile>" << std::endl;
  pvtu_file.close();
}
//----------------------------------------------------------------------------
void PVTKFile::vtk_header_open(const Mesh& mesh) const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  // Write headers
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\"   >\n");
  fprintf(fp, "<UnstructuredGrid>  \n");
  fprintf(fp, "<Piece  NumberOfPoints=\" %8u\"  NumberOfCells=\" %8u\">  \n",
	  mesh.num_vertices(), mesh.num_cells());

  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::vtk_header_close() const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  // Close headers
  fprintf(fp, "</Piece> \n </UnstructuredGrid> \n </VTKFile>");

  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::vtu_name_update(const int counter)
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());

  fileid << counter;
   newfilename << filestart << fileid.str() << "_" << MPI::process_number() <<".vtu";
  vtu_filename = newfilename.str();

  // Make sure file is empty
  FILE* fp = fopen(vtu_filename.c_str(), "w");
  if (!fp)
    error("Unable to open file %s", filename.c_str());
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::pvtu_name_update(const int counter)
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());

  fileid << counter;
  newfilename << filestart << fileid.str() << ".pvtu";

  pvtu_filename = newfilename.str();

  // Make sure file is empty
  FILE* fp = fopen(pvtu_filename.c_str(), "w");
  if (!fp)
    error("Unable to open file %s", filename.c_str());
  fclose(fp);
}
//----------------------------------------------------------------------------
template<class T>
void PVTKFile::mesh_function_write(T& meshfunction)
{
  // Update vtu file name and clear file
  vtu_name_update(counter);

  // Write pvd file
  if(MPI::process_number() == 0)
    pvd_file_write(counter);

  const Mesh& mesh = meshfunction.mesh();

  if( meshfunction.dim() != mesh.topology().dim() )
    error("VTK output of mesh functions is implemenetd for cell-based functions only.");

  // Write headers
  vtk_header_open(mesh);

  // Write mesh
  mesh_write(mesh);

  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);

  fp << "<CellData  Scalars=\"U\">" << std::endl;
  fp << "<DataArray  type=\"Float64\"  Name=\"U\"  format=\"ascii\">" << std::endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    fp << meshfunction.get( cell->index() )  << std::endl;
  fp << "</DataArray>" << std::endl;
  fp << "</CellData>" << std::endl;

  // Close file
  fp.close();

  // Close headers
  vtk_header_close();

  // Increase the number of times we have saved the mesh function
  counter++;

  cout << "saved mesh function " << counter << " times." << endl;

  cout << "Saved mesh function " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in VTK format." << endl;
}
//-----------------------------------------------------------------------------
