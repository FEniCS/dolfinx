// Copyright (C) 2005-2009 Garth N. Wells
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2005-2006.
// Modified by Kristian Oelgaard 2006.
// Modified by Martin Alnes 2008.
// Modified by Niclas Jansson 2009.
//
// First added:  2005-07-05
// Last changed: 2011-03-17

#include <ostream>
#include <sstream>
#include <vector>
#include <boost/cstdint.hpp>
#include <boost/detail/endian.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include "Encoder.h"
#include "VTKWriter.h"
#include "VTKFile.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKFile::VTKFile(const std::string filename, std::string encoding)
               : GenericFile(filename), encoding(encoding), binary(false),
                 compress(false)
{
  if (encoding != "ascii" && encoding != "base64" && encoding != "compressed")
    error("Requested VTK file encoding '%s' is unknown. Options are 'ascii', \n 'base64' or 'compressed'.");

  if (encoding == "ascii")
  {
    encode_string = "ascii";
    binary = false;
  }
  else if (encoding == "base64" || encoding == "compressed")
  {
    encode_string = "binary";
    binary = true;
    if (encoding == "compressed")
      compress = true;
  }
  else
    error("Unknown encoding type.");
}
//----------------------------------------------------------------------------
VTKFile::~VTKFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const Mesh& mesh)
{
  // Get vtu file name and intialise out files
  std::string vtu_filename = init(mesh, mesh.topology().dim());

  // Write local mesh to vtu file
  VTKWriter::write_mesh(mesh, mesh.topology().dim(), vtu_filename, binary, compress);

  // Parallel-specfic files
  if (MPI::num_processes() > 1 && MPI::process_number() == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    pvtu_mesh_write(pvtu_filename, vtu_filename);
  }

  // Finalise and write pvd files
  finalize(vtu_filename, counter);

  log(TRACE, "Saved mesh %s (%s) to file %s in VTK format.",
      mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const MeshFunction<bool>& meshfunction)
{
  mesh_function_write(meshfunction);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const MeshFunction<unsigned int>& meshfunction)
{
  mesh_function_write(meshfunction);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const MeshFunction<int>& meshfunction)
{
  mesh_function_write(meshfunction);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const MeshFunction<double>& meshfunction)
{
  mesh_function_write(meshfunction);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const Function& u)
{
  write(u, counter);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const std::pair<const Function*, double> u)
{
  write(*(u.first), u.second);
}
//----------------------------------------------------------------------------
void VTKFile::write(const Function& u, double time)
{
  const Mesh& mesh = u.function_space().mesh();

  // Get vtu file name and intialise
  std::string vtu_filename = init(mesh, mesh.topology().dim());

  // Write mesh
  VTKWriter::write_mesh(mesh, mesh.topology().dim(), vtu_filename, binary,
                        compress);

  // Write results
  results_write(u, vtu_filename);

  // Parallel-specfic files
  if (MPI::num_processes() > 1 && MPI::process_number() == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    pvtu_mesh_write(pvtu_filename, vtu_filename);
    pvtu_results_write(u, pvtu_filename);
  }

  // Finalise and write pvd files
  finalize(vtu_filename, time);

  log(TRACE, "Saved function %s (%s) to file %s in VTK format.",
      u.name().c_str(), u.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
std::string VTKFile::init(const Mesh& mesh, uint cell_dim) const
{
  // Get vtu file name and clear file
  std::string vtu_filename = vtu_name(MPI::process_number(),
                                      MPI::num_processes(),
                                      counter,
                                      ".vtu");
  clear_file(vtu_filename);

  // Number of cells
  const uint num_cells = mesh.topology().size(cell_dim);

  // Write headers
  vtk_header_open(mesh.num_vertices(), num_cells, vtu_filename);

  if (MPI::num_processes() > 1 && MPI::process_number() == 0)
  {
    // Get pvtu file name and clear file
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    clear_file(pvtu_filename);
    pvtu_header_open(pvtu_filename);
  }

  return vtu_filename;
}
//----------------------------------------------------------------------------
void VTKFile::finalize(std::string vtu_filename, double time)
{
  // Close headers
  vtk_header_close(vtu_filename);

  // Parallel-specfic files
  if (MPI::num_processes() > 1)
  {
    if (MPI::process_number() == 0)
    {
      // Close pvtu headers
      std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
      pvtu_header_close(pvtu_filename);

      // Write pvd file (parallel)
      pvd_file_write(counter, counter, pvtu_filename);
    }
  }
  else
    pvd_file_write(counter, time, vtu_filename);

  // Increase the number of times we have saved the object
  counter++;
}
//----------------------------------------------------------------------------
void VTKFile::results_write(const Function& u, std::string vtu_filename) const
{
  // Get rank of Function
  const uint rank = u.value_rank();
  if(rank > 2)
    error("Only scalar, vector and tensor functions can be saved in VTK format.");

  // Get number of components
  const uint dim = u.value_size();

  // Check that function type can be handled
  if (rank == 1)
  {
    if(!(dim == 2 || dim == 3))
      error("Don't know what to do with vector function with dim other than 2 or 3.");
  }
  else if (rank == 2)
  {
    if(!(dim == 4 || dim == 9))
      error("Don't know what to do with tensor function with dim other than 4 or 9.");
  }

  // Test for cell-based element type
  const Mesh& mesh(u.function_space().mesh());
  uint cell_based_dim = 1;
  for (uint i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();

  const GenericDofMap& dofmap(u.function_space().dofmap());
  if (dofmap.max_cell_dimension() == cell_based_dim)
    VTKWriter::write_cell_data(u, vtu_filename, binary, compress);
  else
    write_point_data(u, mesh, vtu_filename);
}
//----------------------------------------------------------------------------
void VTKFile::write_point_data(const GenericFunction& u, const Mesh& mesh,
                               std::string vtu_filename) const
{
  const uint rank = u.value_rank();
  const uint num_vertices = mesh.num_vertices();

  // Get number of components
  const uint dim = u.value_size();

  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);

  // Allocate memory for function values at vertices
  const uint size = num_vertices*dim;
  Array<double> values(size);

  // Get function values at vertices and zero any small values
  u.compute_vertex_values(values, mesh);
  assert(values.size() == size);
  values.zero_eps(DOLFIN_EPS);
  if (rank == 0)
  {
    fp << "<PointData  Scalars=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  format=\""<< encode_string <<"\">" << std::endl;
  }
  else if (rank == 1)
  {
    fp << "<PointData  Vectors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">" << std::endl;
  }
  else if (rank == 2)
  {
    fp << "<PointData  Tensors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"9\" format=\""<< encode_string <<"\">" << std::endl;
  }

  if (encoding == "ascii")
  {
    std::ostringstream ss;
    ss << std::scientific;
    for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    {
      if(rank == 1 && dim == 2)
      {
        // Append 0.0 to 2D vectors to make them 3D
        for(uint i = 0; i < 2; i++)
          ss << " " << values[vertex->index() + i*num_vertices];
        ss << " " << 0.0;
      }
      else if (rank == 2 && dim == 4)
      {
        // Pad 2D tensors with 0.0 to make them 3D
        for(uint i = 0; i < 2; i++)
        {
          ss << " " << values[vertex->index() + (2*i + 0)*num_vertices];
          ss << " " << values[vertex->index() + (2*i + 1)*num_vertices];
          ss << " " << 0.0;
        }
        ss << " " << 0.0;
        ss << " " << 0.0;
        ss << " " << 0.0;
      }
      else
      {
        // Write all components
        for(uint i = 0; i < dim; i++)
          ss << " " << values[vertex->index() + i*num_vertices];
      }
      ss << std::endl;
    }

    // Send to file
    fp << ss.str();
  }
  else if (encoding == "base64" || encoding == "compressed")
  {
    // Number of zero paddings per point
    uint padding_per_point = 0;
    if (rank == 1 && dim == 2)
      padding_per_point = 1;
    else if (rank == 2 && dim == 4)
      padding_per_point = 5;

    // Number of data entries per point and total number
    const uint num_data_per_point = dim + padding_per_point;
    const uint num_total_data_points = num_vertices*num_data_per_point;

    std::vector<float> data(num_total_data_points, 0);
    for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    {
      const uint index = vertex->index();
      for(uint i = 0; i < dim; i++)
        data[index*num_data_per_point + i] = values[index + i*num_vertices];
    }

    // Create encoded stream
    fp << VTKWriter::encode_stream(data, compress) << std::endl;
  }

  fp << "</DataArray> " << std::endl;
  fp << "</PointData> " << std::endl;
}
//----------------------------------------------------------------------------
void VTKFile::pvd_file_write(uint step, double time, std::string _filename)
{
  std::fstream pvd_file;

  if (step == 0)
  {
    // Open pvd file
    pvd_file.open(filename.c_str(), std::ios::out|std::ios::trunc);

    // Write header
    pvd_file << "<?xml version=\"1.0\"?> " << std::endl;
    pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\">" << std::endl;
    pvd_file << "<Collection> " << std::endl;
  }
  else
  {
    // Open pvd file
    pvd_file.open(filename.c_str(), std::ios::out|std::ios::in);
    pvd_file.seekp(mark);
  }

  // Remove directory path from name for pvd file
  std::string fname = strip_path(_filename);

  // Data file name
  pvd_file << "<DataSet timestep=\"" << time << "\" part=\"0\"" << " file=\"" <<  fname <<  "\"/>" << std::endl;
  mark = pvd_file.tellp();

  // Close headers
  pvd_file << "</Collection>" << std::endl;
  pvd_file << "</VTKFile>" << std::endl;

  // Close file
  pvd_file.close();
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_mesh_write(std::string pvtu_filename,
                              std::string vtu_filename) const
{
  // Open pvtu file
  std::ofstream pvtu_file;
  pvtu_file.open(pvtu_filename.c_str(), std::ios::app);

  pvtu_file << "<PCellData>" << std::endl;
  pvtu_file << "<PDataArray  type=\"UInt32\"  Name=\"connectivity\"/>" << std::endl;
  pvtu_file << "<PDataArray  type=\"UInt32\"  Name=\"offsets\"/>" << std::endl;
  pvtu_file << "<PDataArray  type=\"UInt8\"  Name=\"types\"/>"  << std::endl;
  pvtu_file << "</PCellData>" << std::endl;

  pvtu_file << "<PPoints>" <<std::endl;
  pvtu_file << "<PDataArray  type=\"Float32\"  NumberOfComponents=\"3\"/>" << std::endl;
  pvtu_file << "</PPoints>" << std::endl;

  for(uint i = 0; i < MPI::num_processes(); i++)
  {
    std::string tmp_string = strip_path(vtu_name(i, MPI::num_processes(), counter, ".vtu"));
    pvtu_file << "<Piece Source=\"" << tmp_string << "\"/>" << std::endl;
  }

  pvtu_file.close();
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_results_write(const Function& u, std::string pvtu_filename) const
{
  // Type of data (point or cell). Point by default.
  std::string data_type = "point";

  // For brevity
  const FunctionSpace& V = u.function_space();
  const Mesh& mesh = V.mesh();
  const FiniteElement& element = V.element();
  const GenericDofMap& dofmap = V.dofmap();

  // Get rank of Function
  const uint rank = element.value_rank();
  if(rank > 2)
    error("Only scalar, vector and tensor functions can be saved in VTK format.");

  // Get number of components
  const uint dim = u.value_size();

  // Test for cell-based element type
  uint cell_based_dim = 1;
  for (uint i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();
  if (dofmap.max_cell_dimension() == cell_based_dim)
    data_type = "cell";

  // Write file
  pvtu_results_write(dim, rank, data_type, u.name(), pvtu_filename);
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_results_write(uint dim, uint rank, std::string data_type,
                                 std::string name,
                                 std::string pvtu_filename) const
{
  // Open pvtu file
  std::ofstream pvtu_file(pvtu_filename.c_str(), std::ios::app);

  // Write function data at mesh cells
  if (data_type == "cell")
  {
    // Write headers
    if (rank == 0)
    {
      pvtu_file << "<PCellData  Scalars=\"" << name << "\"> " << std::endl;
      pvtu_file << "<PDataArray  type=\"Float32\"  Name=\"" << name << "\">" << std::endl;
    }
    else if (rank == 1)
    {
      if (!(dim == 2 || dim == 3))
        error("Do not know what to do with vector function with dim other than 2 or 3.");
      pvtu_file << "<PCellData  Vectors=\"" << name << "\"> " << std::endl;
      pvtu_file << "<PDataArray  type=\"Float32\"  Name=\"" << name << "\"  NumberOfComponents=\"3\">" << std::endl;
    }
    else if (rank == 2)
    {
      if(!(dim == 4 || dim == 9))
        error("Don't know what to do with tensor function with dim other than 4 or 9.");
      pvtu_file << "<PCellData  Tensors=\"" << name << "\"> " << std::endl;
      pvtu_file << "<PDataArray  type=\"Float32\"  Name=\"" << name << "\"  NumberOfComponents=\"9\">" << std::endl;
    }
    else
      error("Don't know how to write function of this rank to VTK file.");

    pvtu_file << "</PDataArray> " << std::endl;
    pvtu_file << "</PCellData> " << std::endl;
  }
  else if (data_type == "point")
  {
    if (rank == 0)
    {
      pvtu_file << "<PPointData  Scalars=\"" << name << "\"> " << std::endl;
      pvtu_file << "<PDataArray  type=\"Float32\"  Name=\"" << name << "\">" << std::endl;
    }
    else if (rank == 1)
    {
      if (!(dim == 2 || dim == 3))
        error("Do not know what to do with vector function with dim other than 2 or 3.");
      pvtu_file << "<PPointData  Vectors=\"" << name << "\"> " << std::endl;
      pvtu_file << "<PDataArray  type=\"Float32\"  Name=\"" << name << "\"  NumberOfComponents=\"3\">" << std::endl;
    }
    else if (rank == 2)
    {
      if(!(dim == 4 || dim == 9))
        error("Don't know what to do with tensor function with dim other than 4 or 9.");
      pvtu_file << "<PPointData  Tensors=\"" << name << "\"> " << std::endl;
      pvtu_file << "<PDataArray  type=\"Float32\"  Name=\"" << name << "\"  NumberOfComponents=\"9\">" << std::endl;
    }
    else
      error("Don't know how to write function of this rank to VTK file.");

    pvtu_file << "</PDataArray> " << std::endl;
    pvtu_file << "</PPointData> " << std::endl;
  }

  pvtu_file.close();
}
//----------------------------------------------------------------------------
void VTKFile::vtk_header_open(uint num_vertices, uint num_cells,
                              std::string vtu_filename) const
{
  // Open file
  std::ofstream file(vtu_filename.c_str(), std::ios::app);
  if (!file.is_open())
    error("Unable to open file %s", filename.c_str());

  // Figure out endianness of machine
  std::string endianness = "";
  if (encode_string == "binary")
  {
    #if defined BOOST_LITTLE_ENDIAN
    endianness = "byte_order=\"LittleEndian\"";
    #elif defined BOOST_BIG_ENDIAN
    endianness = "byte_order=\"BigEndian\"";;
    #else
    error("Unable to determine the endianness of the machine for VTK binary output.");
    #endif
  }

  // Compression string
  std::string compressor = "";
  if (encoding == "compressed")
    compressor = "compressor=\"vtkZLibDataCompressor\"";

  // Write headers
  file << "<?xml version=\"1.0\"?>" << std::endl;
  file << "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\" " << endianness <<  " " << compressor << ">" << std::endl;
  file << "<UnstructuredGrid>" << std::endl;
  file << "<Piece  NumberOfPoints=\"" << num_vertices << "\" NumberOfCells=\"" << num_cells << "\">" << std::endl;

  // Close file
  file.close();
}
//----------------------------------------------------------------------------
void VTKFile::vtk_header_close(std::string vtu_filename) const
{
  // Open file
  std::ofstream file(vtu_filename.c_str(), std::ios::app);
  if ( !file.is_open() )
    error("Unable to open file %s", filename.c_str());

  // Close headers
  file << "</Piece>" << std::endl << "</UnstructuredGrid>" << std::endl << "</VTKFile>";

  // Close file
  file.close();
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_header_open(std::string pvtu_filename) const
{
  // Open pvtu file
  std::ofstream pvtu_file(pvtu_filename.c_str(), std::ios::trunc);

  // Write header
  pvtu_file << "<?xml version=\"1.0\"?>" << std::endl;
  pvtu_file << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\">" << std::endl;
  pvtu_file << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
  pvtu_file.close();
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_header_close(std::string pvtu_filename) const
{
  // Open pvtu file
  std::ofstream pvtu_file(pvtu_filename.c_str(), std::ios::app);

  pvtu_file << "</PUnstructuredGrid>" << std::endl;
  pvtu_file << "</VTKFile>" << std::endl;
  pvtu_file.close();
}
//----------------------------------------------------------------------------
std::string VTKFile::vtu_name(const int process, const int num_processes,
                              const int counter, std::string ext) const
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(filename, 0, filename.find_last_of("."));
  extension.assign(filename, filename.find_last_of("."), filename.size());

  fileid << counter;

  // Add process number to .vtu file name
  std::string proc = "";
  if (num_processes > 1)
  {
    std::ostringstream _p;
    _p << "_p" << process << "_";
    proc = _p.str();
  }
  newfilename << filestart << proc << fileid.str() << ext;

  return newfilename.str();
}
//----------------------------------------------------------------------------
template<class T>
void VTKFile::mesh_function_write(T& meshfunction)
{
  const Mesh& mesh = meshfunction.mesh();
  const uint cell_dim = meshfunction.dim();

  // Throw error for MeshFunctions on vertices for interval elements
  if (mesh.topology().dim() == 1 && cell_dim == 0)
    error("VTK output for MeshFunctions on interval facets is not supported.");

  if (cell_dim != mesh.topology().dim() && cell_dim != mesh.topology().dim() - 1)
    error("VTK output of mesh functions is implemented for cell- and facet-based functions only.");

  // Update vtu file name and clear file
  std::string vtu_filename = init(mesh, cell_dim);

  // Write mesh
  VTKWriter::write_mesh(mesh, cell_dim, vtu_filename, binary, compress);

  // Parallel-specfic files
  if (MPI::num_processes() > 1 && MPI::process_number() == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    pvtu_mesh_write(pvtu_filename, vtu_filename);
    pvtu_results_write(1, 0, "cell", meshfunction.name(), pvtu_filename);
  }

  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);

  fp << "<CellData  Scalars=\"" << meshfunction.name() << "\">" << std::endl;
  fp << "<DataArray  type=\"Float32\"  Name=\"" << meshfunction.name() << "\"  format=\"ascii\">" << std::endl;
  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
    fp << meshfunction[cell->index()] << std::endl;
  fp << "</DataArray>" << std::endl;
  fp << "</CellData>" << std::endl;

  // Close file
  fp.close();

  // Write pvd files
  finalize(vtu_filename, counter);

  log(TRACE, "Saved mesh function %s (%s) to file %s in VTK format.",
      mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
void VTKFile::clear_file(std::string file) const
{
  // Open file and clear
  std::ofstream _file(file.c_str(), std::ios::trunc);
  if ( !_file.is_open() )
    error("Unable to open file %s", file.c_str());
  _file.close();
}
//----------------------------------------------------------------------------
std::string VTKFile::strip_path(std::string file) const
{
  std::string fname;
  fname.assign(file, filename.find_last_of("/") + 1, file.size());
  return fname;
}
//----------------------------------------------------------------------------
