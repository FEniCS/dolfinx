// Copyright (C) 2005-2009 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKFile.h"
#include "VTKWriter.h"
#include "pugixml.hpp"
#include <boost/cstdint.hpp>
#include <boost/detail/endian.hpp>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

using namespace dolfin;
using namespace dolfin::io;

//----------------------------------------------------------------------------
VTKFile::VTKFile(const std::string filename) : _filename(filename), counter(0)
{
  // Do nothing
}
//----------------------------------------------------------------------------
VTKFile::~VTKFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::Mesh& mesh) { write_mesh(mesh, counter); }
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<bool>& meshfunction)
{
  mesh_function_write(meshfunction, counter);
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<std::size_t>& meshfunction)
{
  mesh_function_write(meshfunction, counter);
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<int>& meshfunction)
{
  mesh_function_write(meshfunction, counter);
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<double>& meshfunction)
{
  mesh_function_write(meshfunction, counter);
}
//----------------------------------------------------------------------------
void VTKFile::write(const function::Function& u) { write_function(u, counter); }
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::Mesh& mesh, double time)
{
  write_mesh(mesh, time);
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<int>& mf, double time)
{
  mesh_function_write(mf, time);
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<std::size_t>& mf, double time)
{
  mesh_function_write(mf, time);
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<double>& mf, double time)
{
  mesh_function_write(mf, time);
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::MeshFunction<bool>& mf, double time)
{
  mesh_function_write(mf, time);
}
//----------------------------------------------------------------------------
void VTKFile::write(const function::Function& u, double time)
{
  write_function(u, time);
}
//----------------------------------------------------------------------------
void VTKFile::write_function(const function::Function& u, double time)
{
  assert(u.function_space()->mesh());
  const mesh::Mesh& mesh = *u.function_space()->mesh();

  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get vtu file name and initialise
  std::string vtu_filename = init(mesh, mesh.topology().dim());

  // Write mesh
  VTKWriter::write_mesh(mesh, mesh.topology().dim(), vtu_filename);

  // Write results
  results_write(u, vtu_filename);

  // Parallel-specific files
  const std::size_t num_processes = MPI::size(mpi_comm);
  if (num_processes > 1 && MPI::rank(mpi_comm) == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    pvtu_write(u, pvtu_filename);
    pvd_file_write(counter, time, pvtu_filename);
  }
  else if (num_processes == 1)
    pvd_file_write(counter, time, vtu_filename);

  // Finalise and write pvd files
  finalize(vtu_filename, time);

  log::log(TRACE, "Saved function %s (%s) to file %s in VTK format.",
           u.name().c_str(), u.name().c_str(), _filename.c_str());
}
//----------------------------------------------------------------------------
void VTKFile::write_mesh(const mesh::Mesh& mesh, double time)
{
  common::Timer t("Write mesh to PVD/VTK file");

  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get vtu file name and initialise out files
  std::string vtu_filename = init(mesh, mesh.topology().dim());

  // Write local mesh to vtu file
  VTKWriter::write_mesh(mesh, mesh.topology().dim(), vtu_filename);

  // Parallel-specific files
  const std::size_t num_processes = MPI::size(mpi_comm);
  if (num_processes > 1 && MPI::rank(mpi_comm) == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    pvtu_write_mesh(pvtu_filename, num_processes);
    pvd_file_write(counter, time, pvtu_filename);
  }
  else if (num_processes == 1)
    pvd_file_write(counter, time, vtu_filename);

  // Finalise
  finalize(vtu_filename, time);

  log::log(TRACE, "Saved mesh %s (%s) to file %s in VTK format.",
           mesh.name().c_str(), mesh.name().c_str(), _filename.c_str());
}
//----------------------------------------------------------------------------
std::string VTKFile::init(const mesh::Mesh& mesh, std::size_t cell_dim) const
{
  // Get MPI communicators
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get vtu file name and clear file
  std::string vtu_filename
      = vtu_name(MPI::rank(mpi_comm), MPI::size(mpi_comm), counter, ".vtu");
  clear_file(vtu_filename);

  // Number of cells and vertices
  const std::size_t num_cells = mesh.topology().ghost_offset(cell_dim);
  const std::size_t num_vertices = mesh.topology().ghost_offset(0);

  // Write headers
  vtk_header_open(num_vertices, num_cells, vtu_filename);

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
void VTKFile::results_write(const function::Function& u,
                            std::string vtu_filename) const
{
  // Get rank of function::Function
  const std::size_t rank = u.value_rank();
  if (rank > 2)
  {
    log::dolfin_error(
        "VTKFile.cpp", "write data to VTK file",
        "Only scalar, vector and tensor functions can be saved in VTK format");
  }

  // Get number of components
  const std::size_t dim = u.value_size();

  // Check that function type can be handled
  if (rank == 1)
  {
    if (!(dim == 2 || dim == 3))
    {
      log::dolfin_error(
          "VTKFile.cpp", "write data to VTK file",
          "Don't know how to handle vector function with dimension "
          "other than 2 or 3");
    }
  }
  else if (rank == 2)
  {
    if (!(dim == 4 || dim == 9))
    {
      log::dolfin_error(
          "VTKFile.cpp", "write data to VTK file",
          "Don't know how to handle tensor function with dimension "
          "other than 4 or 9");
    }
  }

  // Test for cell-based element type
  assert(u.function_space()->mesh());
  const mesh::Mesh& mesh = *u.function_space()->mesh();
  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();

  assert(u.function_space()->dofmap());
  const fem::GenericDofMap& dofmap = *u.function_space()->dofmap();
  if (dofmap.max_element_dofs() == cell_based_dim)
    VTKWriter::write_cell_data(u, vtu_filename);
  else
    write_point_data(u, mesh, vtu_filename);
}
//----------------------------------------------------------------------------
void VTKFile::write_point_data(const function::GenericFunction& u,
                               const mesh::Mesh& mesh,
                               std::string vtu_filename) const
{
  const std::size_t rank = u.value_rank();

  // Get number of components
  const std::size_t dim = u.value_size();

  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
  fp.precision(16);

  // Get function values at vertices
  auto values = u.compute_point_values(mesh);

  if (rank == 0)
  {
    fp << "<PointData  Scalars=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name() << "\"  format=\""
       << "ascii"
       << "\">";
  }
  else if (rank == 1)
  {
    fp << "<PointData  Vectors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"3\" format=\""
       << "ascii"
       << "\">";
  }
  else if (rank == 2)
  {
    fp << "<PointData  Tensors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float64\"  Name=\"" << u.name()
       << "\"  NumberOfComponents=\"9\" format=\""
       << "ascii"
       << "\">";
  }

  std::ostringstream ss;
  ss << std::scientific;
  ss << std::setprecision(16);
  for (auto& vertex : mesh::MeshRange<mesh::Vertex>(mesh))
  {
    if (rank == 1 && dim == 2)
    {
      // Append 0.0 to 2D vectors to make them 3D
      for (std::size_t i = 0; i < 2; i++)
        ss << values(vertex.index(), i) << " ";
      ss << 0.0 << "  ";
    }
    else if (rank == 2 && dim == 4)
    {
      // Pad 2D tensors with 0.0 to make them 3D
      for (std::size_t i = 0; i < 2; i++)
      {
        ss << values(vertex.index(), (2 * i + 0)) << " ";
        ss << values(vertex.index(), (2 * i + 1)) << " ";
        ss << 0.0 << " ";
      }
      ss << 0.0 << " ";
      ss << 0.0 << " ";
      ss << 0.0 << "  ";
    }
    else
    {
      // Write all components
      for (std::size_t i = 0; i < dim; i++)
        ss << values(vertex.index(), i) << " ";
      ss << " ";
    }
  }

  // Send to file
  fp << ss.str();

  fp << "</DataArray> " << std::endl;
  fp << "</PointData> " << std::endl;
}
//----------------------------------------------------------------------------
void VTKFile::pvd_file_write(std::size_t step, double time, std::string fname)
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
    pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
    if (!result)
    {
      log::dolfin_error("VTKFile.cpp", "write data to VTK file",
                        "XML parsing error when reading from existing file");
    }
  }

  // Remove directory path from name for pvd file
  const std::string fname_strip = strip_path(fname);

  // Get Collection node
  pugi::xml_node xml_collections = xml_doc.child("VTKFile").child("Collection");
  assert(xml_collections);

  // Append data set
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = fname_strip.c_str();

  // Save file
  xml_doc.save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_write_mesh(pugi::xml_node xml_node) const
{
  // mesh::Vertex data
  pugi::xml_node vertex_data_node = xml_node.append_child("PPoints");
  pugi::xml_node data_node = vertex_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
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
void VTKFile::pvtu_write_function(std::size_t dim, std::size_t rank,
                                  const std::string data_location,
                                  const std::string name,
                                  const std::string fname,
                                  std::size_t num_processes) const
{
  // Create xml doc
  pugi::xml_document xml_doc;
  pugi::xml_node vtk_node = xml_doc.append_child("VTKFile");
  vtk_node.append_attribute("type") = "PUnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
  grid_node.append_attribute("GhostLevel") = 0;

  // mesh::Mesh
  pvtu_write_mesh(grid_node);

  // Get type based on rank
  std::string rank_type;
  std::size_t num_components = 0;
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
      log::dolfin_error(
          "VTKFile.cpp", "write data to VTK file",
          "Don't know how to handle vector function with dimension "
          "other than 2 or 3");
    }
    num_components = 3;
  }
  else if (rank == 2)
  {
    rank_type = "Tensors";
    if (!(dim == 4 || dim == 9))
    {
      log::dolfin_error(
          "VTKFile.cpp", "write data to VTK file",
          "Don't know how to handle tensor function with dimension "
          "other than 4 or 9");
    }
    num_components = 9;
  }
  else
  {
    log::dolfin_error("VTKFile.cpp", "write data to VTK file",
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
  data_array_node.append_attribute("type") = "Float64";
  data_array_node.append_attribute("Name") = name.c_str();
  data_array_node.append_attribute("NumberOfComponents")
      = (unsigned int)num_components;

  // Write vtu file list
  for (std::size_t i = 0; i < num_processes; i++)
  {
    const std::string tmp_string
        = strip_path(vtu_name(i, num_processes, counter, ".vtu"));
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = tmp_string.c_str();
  }

  xml_doc.save_file(fname.c_str(), "  ");
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_write_mesh(const std::string fname,
                              const std::size_t num_processes) const
{
  // Create xml doc
  pugi::xml_document xml_doc;
  pugi::xml_node vtk_node = xml_doc.append_child("VTKFile");
  vtk_node.append_attribute("type") = "PUnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
  grid_node.append_attribute("GhostLevel") = 0;

  // mesh::Mesh
  pvtu_write_mesh(grid_node);

  // Write vtu file list
  for (std::size_t i = 0; i < num_processes; i++)
  {
    const std::string tmp_string
        = strip_path(vtu_name(i, num_processes, counter, ".vtu"));
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = tmp_string.c_str();
  }

  xml_doc.save_file(fname.c_str(), "  ");
}
//----------------------------------------------------------------------------
void VTKFile::pvtu_write(const function::Function& u,
                         const std::string fname) const
{
  assert(u.function_space()->element());
  const std::size_t rank = u.function_space()->element()->value_rank();
  if (rank > 2)
  {
    log::dolfin_error(
        "VTKFile.cpp", "write data to VTK file",
        "Only scalar, vector and tensor functions can be saved in VTK format");
  }

  // Get number of components
  const std::size_t dim = u.value_size();

  // Get mesh
  assert(u.function_space()->mesh());
  const mesh::Mesh& mesh = *(u.function_space()->mesh());

  // Test for cell-based element type
  std::string data_type = "point";
  std::size_t cell_based_dim = 1;
  assert(u.function_space()->dofmap());
  for (std::size_t i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();
  if (u.function_space()->dofmap()->max_element_dofs() == cell_based_dim)
    data_type = "cell";

  const std::size_t num_processes = MPI::size(mesh.mpi_comm());
  pvtu_write_function(dim, rank, data_type, u.name(), fname, num_processes);
}
//----------------------------------------------------------------------------
void VTKFile::vtk_header_open(std::size_t num_vertices, std::size_t num_cells,
                              std::string vtu_filename) const
{
  // Open file
  std::ofstream file(vtu_filename.c_str(), std::ios::app);
  file.precision(16);
  if (!file.is_open())
  {
    log::dolfin_error("VTKFile.cpp", "write data to VTK file",
                      "Unable to open file \"%s\"", _filename.c_str());
  }

  // Write headers
  file << "<?xml version=\"1.0\"?>" << std::endl;
  file << "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\" "
       << ">" << std::endl;
  file << "<UnstructuredGrid>" << std::endl;
  file << "<Piece  NumberOfPoints=\"" << num_vertices << "\" NumberOfCells=\""
       << num_cells << "\">" << std::endl;

  // Close file
  file.close();
}
//----------------------------------------------------------------------------
void VTKFile::vtk_header_close(std::string vtu_filename) const
{
  // Open file
  std::ofstream file(vtu_filename.c_str(), std::ios::app);
  file.precision(16);
  if (!file.is_open())
  {
    log::dolfin_error("VTKFile.cpp", "write data to VTK file",
                      "Unable to open file \"%s\"", _filename.c_str());
  }

  // Close headers
  file << "</Piece>" << std::endl
       << "</UnstructuredGrid>" << std::endl
       << "</VTKFile>";

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

  filestart.assign(_filename, 0, _filename.find_last_of("."));
  extension.assign(_filename, _filename.find_last_of("."), _filename.size());

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
template <typename T>
void VTKFile::mesh_function_write(T& meshfunction, double time)
{
  const mesh::Mesh& mesh = *meshfunction.mesh();
  const std::size_t cell_dim = meshfunction.dim();

  // Update vtu file name and clear file
  std::string vtu_filename = init(mesh, cell_dim);

  // Write mesh
  VTKWriter::write_mesh(mesh, cell_dim, vtu_filename);

  // Open file to write data
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);
  fp.precision(16);
  fp << "<CellData  Scalars=\"" << meshfunction.name() << "\">" << std::endl;
  fp << "<DataArray  type=\"Float64\"  Name=\"" << meshfunction.name()
     << "\"  format=\"ascii\">";

  // Write data
  for (auto& cell : mesh::MeshRange<mesh::MeshEntity>(mesh, cell_dim))
    fp << meshfunction[cell.index()] << " ";

  // Write footers
  fp << "</DataArray>" << std::endl;
  fp << "</CellData>" << std::endl;

  // Close file
  fp.close();

  // Parallel-specific files
  const std::size_t num_processes = MPI::size(mesh.mpi_comm());
  const std::size_t process_number = MPI::rank(mesh.mpi_comm());
  if (num_processes > 1 && process_number == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, ".pvtu");
    pvtu_write_function(1, 0, "cell", meshfunction.name(), pvtu_filename,
                        num_processes);
    pvd_file_write(counter, time, pvtu_filename);
  }
  else if (num_processes == 1)
    pvd_file_write(counter, time, vtu_filename);

  // Write pvd files
  finalize(vtu_filename, time);

  log::log(TRACE, "Saved mesh function %s (%s) to file %s in VTK format.",
           mesh.name().c_str(), mesh.name().c_str(), _filename.c_str());
}
//----------------------------------------------------------------------------
void VTKFile::clear_file(std::string file) const
{
  // Open file and clear
  std::ofstream _file(file.c_str(), std::ios::trunc);
  if (!_file.is_open())
  {
    log::dolfin_error("VTKFile.cpp", "clear VTK file",
                      "Unable to open file \"%s\"", file.c_str());
  }
  _file.close();
}
//----------------------------------------------------------------------------
std::string VTKFile::strip_path(std::string file) const
{
  std::string fname;
  fname.assign(file, _filename.find_last_of("/") + 1, file.size());
  return fname;
}
//----------------------------------------------------------------------------
