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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2005-2011
// Modified by Kristian Oelgaard 2006
// Modified by Martin Alnes 2008
// Modified by Niclas Jansson 2009
//
// First added:  2005-07-05
// Last changed: 2011-11-14

#include <ostream>
#include <sstream>
#include <vector>
#include <boost/cstdint.hpp>
#include <boost/detail/endian.hpp>

#include "pugixml.hpp"

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
  : GenericFile(filename, "VTK"),
    encoding(encoding), binary(false), compress(false)
{
  if (encoding != "ascii" && encoding != "base64" && encoding != "compressed")
  {
    dolfin_error("VTKFile.cpp",
                 "create VTK file",
                 "Unknown encoding (\"%s\"). "
                 "Known encodings are \"ascii\", \"base64\" and \"compressed\"",
                 encoding.c_str());
  }

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
  {
    dolfin_error("VTKFile.cpp",
                 "create VTK file",
                 "Unknown encoding (\"%s\"). "
                 "Known encodings are \"ascii\", \"base64\" and \"compressed\"",
                 encoding.c_str());
  }
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
    pvtu_write_mesh(pvtu_filename);
    pvd_file_write(counter, counter, pvtu_filename);
  }
  else if (MPI::num_processes() == 1)
    pvd_file_write(counter, counter, vtu_filename);

  // Finalise
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
  u.gather();
  write(u, counter);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(const std::pair<const Function*, double> u)
{
  dolfin_assert(u.first);
  u.first->gather();
  write(*(u.first), u.second);
}
//----------------------------------------------------------------------------
void VTKFile::write(const Function& u, double time)
{
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

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
    pvtu_write(u, pvtu_filename);
    pvd_file_write(counter, time, pvtu_filename);
  }
  else if (MPI::num_processes() == 1)
    pvd_file_write(counter, time, vtu_filename);

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

  return vtu_filename;
}
//----------------------------------------------------------------------------
void VTKFile::finalize(std::string vtu_filename, double time)
{
  // Close headers
  vtk_header_close(vtu_filename);

  // Increase the number of times we have saved the object
  counter++;
}
//----------------------------------------------------------------------------
void VTKFile::results_write(const Function& u, std::string vtu_filename) const
{
  // Get rank of Function
  const uint rank = u.value_rank();
  if (rank > 2)
  {
    dolfin_error("VTKFile.cpp",
                 "write data to VTK file",
                 "Only scalar, vector and tensor functions can be saved in VTK format");
  }

  // Get number of components
  const uint dim = u.value_size();

  // Check that function type can be handled
  if (rank == 1)
  {
    if (!(dim == 2 || dim == 3))
    {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle vector function with dimension other than 2 or 3");
    }
  }
  else if (rank == 2)
  {
    if (!(dim == 4 || dim == 9))
    {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle tensor function with dimension other than 4 or 9");
    }
  }

  // Test for cell-based element type
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();
  uint cell_based_dim = 1;
  for (uint i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap= *u.function_space()->dofmap();
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
  dolfin_assert(values.size() == size);
  values.zero_eps(DOLFIN_EPS);
  if (rank == 0)
  {
    fp << "<PointData  Scalars=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  format=\""<< encode_string <<"\">";
  }
  else if (rank == 1)
  {
    fp << "<PointData  Vectors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">";
  }
  else if (rank == 2)
  {
    fp << "<PointData  Tensors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"9\" format=\""<< encode_string <<"\">";
  }

  if (encoding == "ascii")
  {
    std::ostringstream ss;
    ss << std::scientific;
    for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    {
      if (rank == 1 && dim == 2)
      {
        // Append 0.0 to 2D vectors to make them 3D
        for(uint i = 0; i < 2; i++)
          ss << values[vertex->index() + i*num_vertices] << " ";
        ss << 0.0 << "  ";
      }
      else if (rank == 2 && dim == 4)
      {
        // Pad 2D tensors with 0.0 to make them 3D
        for(uint i = 0; i < 2; i++)
        {
          ss << values[vertex->index() + (2*i + 0)*num_vertices] << " ";
          ss << values[vertex->index() + (2*i + 1)*num_vertices] << " ";
          ss << 0.0 << " ";
        }
        ss << 0.0 << " ";
        ss << 0.0 << " ";
        ss << 0.0 << "  ";
      }
      else
      {
        // Write all components
        for(uint i = 0; i < dim; i++)
          ss << values[vertex->index() + i*num_vertices] << " ";
        ss << " ";
      }
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
  pugi::xml_document xml_doc;
  if (step == 0)
  {
    pugi::xml_node vtk_node = xml_doc.append_child("VTKFile");
    vtk_node.append_attribute("type") = "Collection";
    vtk_node.append_attribute("version") = "0.1";
    vtk_node.append_child("Collection");
  }
  else
  {
    pugi::xml_parse_result result = xml_doc.load_file(filename.c_str());
    if (!result)
    {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "XML parsing error when reading from existing file");
    }
  }

  // Remove directory path from name for pvd file
  const std::string fname = strip_path(_filename);

  // Get Collection node
  pugi::xml_node xml_collections = xml_doc.child("VTKFile").child("Collection");
  dolfin_assert(xml_collections);

  // Append data set
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = fname.c_str();

  // Save file
  xml_doc.save_file(filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_write_mesh(pugi::xml_node xml_node) const
{
  // Vertex data
  pugi::xml_node vertex_data_node = xml_node.append_child("PPoints");
  pugi::xml_node data_node = vertex_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float32";
  data_node.append_attribute("NumberOfComponents") = "3";

  // Cell data
  pugi::xml_node cell_data_node = xml_node.append_child("PCellData");

  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "UInt32";
  data_node.append_attribute("Name") = "connectivity";

  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "UInt32";
  data_node.append_attribute("Name") = "offsets";

  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "UInt8";
  data_node.append_attribute("Name") = "types";
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_write_function(uint dim, uint rank,
                                  const std::string data_location,
                                  const std::string name,
                                  const std::string filename) const
{
  // Create xml doc
  pugi::xml_document xml_doc;
  pugi::xml_node vtk_node = xml_doc.append_child("VTKFile");
  vtk_node.append_attribute("type") = "PUnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
  grid_node.append_attribute("GhostLevel") = 0;

  // Mesh
  pvtu_write_mesh(grid_node);

  // Get type based on rank
  std::string rank_type;
  uint num_components = 0;
  if (rank == 0)
  {
    rank_type = "Scalars";
    num_components = 0;
  }
  else if (rank == 1)
  {
    rank_type = "Vectors";
    if (!(dim == 2 || dim == 3))
    {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle vector function with dimension other than 2 or 3");
    }
    num_components = 3;
  }
  else if (rank == 2)
  {
    rank_type = "Tensors";
    if (!(dim == 4 || dim == 9))
    {
      dolfin_error("VTKFile.cpp",
                   "write data to VTK file",
                   "Don't know how to handle tensor function with dimension other than 4 or 9");
    }
    num_components = 9;
  }
  else
  {
    dolfin_error("VTKFile.cpp",
                 "write data to VTK file",
                 "Cannot handle XML output of rank %d", rank);
  }

  // Add function data
  pugi::xml_node data_node;
  if (data_location == "point")
    data_node = grid_node.append_child("PPointData");
  else if (data_location == "cell")
    data_node = grid_node.append_child("PCellData");

  data_node.append_attribute(rank_type.c_str()) = name.c_str();
  pugi::xml_node data_array_node = data_node.append_child("PDataArray");
  data_array_node.append_attribute("type") = "Float32";
  data_array_node.append_attribute("Name") = name.c_str();
  data_array_node.append_attribute("NumberOfComponents") = num_components;

  // Write vtu file list
  for(uint i = 0; i < MPI::num_processes(); i++)
  {
    const std::string tmp_string = strip_path(vtu_name(i, MPI::num_processes(), counter, ".vtu"));
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = tmp_string.c_str();
  }

  xml_doc.save_file(filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_write_mesh(const std::string filename) const
{
  // Create xml doc
  pugi::xml_document xml_doc;
  pugi::xml_node vtk_node = xml_doc.append_child("VTKFile");
  vtk_node.append_attribute("type") = "PUnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
  grid_node.append_attribute("GhostLevel") = 0;

  // Mesh
  pvtu_write_mesh(grid_node);

  // Write vtu file list
  for(uint i = 0; i < MPI::num_processes(); i++)
  {
    const std::string tmp_string = strip_path(vtu_name(i, MPI::num_processes(), counter, ".vtu"));
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = tmp_string.c_str();
  }

  xml_doc.save_file(filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_write(const Function& u, const std::string filename) const
{
  dolfin_assert(u.function_space()->element());
  const uint rank = u.function_space()->element()->value_rank();
  if (rank > 2)
  {
    dolfin_error("VTKFile.cpp",
                 "write data to VTK file",
                 "Only scalar, vector and tensor functions can be saved in VTK format");
  }

  // Get number of components
  const uint dim = u.value_size();

  // Test for cell-based element type
  std::string data_type = "point";
  uint cell_based_dim = 1;
  dolfin_assert(u.function_space()->mesh());
  dolfin_assert(u.function_space()->dofmap());
  for (uint i = 0; i < rank; i++)
    cell_based_dim *= u.function_space()->mesh()->topology().dim();
  if (u.function_space()->dofmap()->max_cell_dimension() == cell_based_dim)
    data_type = "cell";

  pvtu_write_function(dim, rank, data_type, u.name(), filename);
}
//----------------------------------------------------------------------------
void VTKFile::vtk_header_open(uint num_vertices, uint num_cells,
                              std::string vtu_filename) const
{
  // Open file
  std::ofstream file(vtu_filename.c_str(), std::ios::app);
  if (!file.is_open())
  {
    dolfin_error("VTKFile.cpp",
                 "write data to VTK file",
                 "Unable to open file \"%s\"", filename.c_str());
  }

  // Figure out endianness of machine
  std::string endianness = "";
  if (encode_string == "binary")
  {
    #if defined BOOST_LITTLE_ENDIAN
    endianness = "byte_order=\"LittleEndian\"";
    #elif defined BOOST_BIG_ENDIAN
    endianness = "byte_order=\"BigEndian\"";;
    #else
    dolfin_error("VTKFile.cpp",
                 "write data to VTK file",
                 "Unable to determine the endianness of the machine for VTK binary output");
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
  if (!file.is_open())
  {
    dolfin_error("VTKFile.cpp",
                 "write data to VTK file",
                 "Unable to open file \"%s\"", filename.c_str());
  }

  // Close headers
  file << "</Piece>" << std::endl << "</UnstructuredGrid>" << std::endl << "</VTKFile>";

  // Close file
  file.close();
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
template<typename T>
void VTKFile::mesh_function_write(T& meshfunction)
{
  const Mesh& mesh = meshfunction.mesh();
  const uint cell_dim = meshfunction.dim();

  // Throw error for MeshFunctions on vertices for interval elements
  if (mesh.topology().dim() == 1 && cell_dim == 0)
  {
    dolfin_error("VTKFile.cpp",
                 "write mesh function to VTK file",
                 "VTK output of mesh functions on interval facets is not supported");
  }

  if (cell_dim != mesh.topology().dim() && cell_dim != mesh.topology().dim() - 1)
  {
    dolfin_error("VTKFile.cpp",
                 "write mesh function to VTK file",
                 "VTK output of mesh functions is implemented for cell- and facet-based functions only");
  }

  // Update vtu file name and clear file
  std::string vtu_filename = init(mesh, cell_dim);

  // Write mesh
  VTKWriter::write_mesh(mesh, cell_dim, vtu_filename, binary, compress);

  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
  fp << "<CellData  Scalars=\"" << meshfunction.name() << "\">" << std::endl;
  fp << "<DataArray  type=\"Float32\"  Name=\"" << meshfunction.name() << "\"  format=\"ascii\">";
  for (MeshEntityIterator cell(mesh, cell_dim); !cell.end(); ++cell)
    fp << meshfunction[cell->index()] << " ";
  fp << "</DataArray>" << std::endl;
  fp << "</CellData>" << std::endl;

  // Close file
  fp.close();

  // Parallel-specfic files
  if (MPI::num_processes() > 1 && MPI::process_number() == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    pvtu_write_function(1, 0, "cell", meshfunction.name(), pvtu_filename);
    pvd_file_write(counter, counter, pvtu_filename);
  }
  else if (MPI::num_processes() == 1)
    pvd_file_write(counter, counter, vtu_filename);

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
  if (!_file.is_open())
  {
    dolfin_error("VTKFile.cpp",
                 "clear VTK file",
                 "Unable to open file \"%s\"", file.c_str());
  }
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
