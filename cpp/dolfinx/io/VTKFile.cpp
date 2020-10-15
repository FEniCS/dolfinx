// Copyright (C) 2005-2020 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKFile.h"
#include "VTKWriter.h"
#include "pugixml.hpp"
#include <boost/cstdint.hpp>
#include <boost/filesystem.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
void write_function(const function::Function<PetscScalar>& u,
                    const std::string filename, const std::size_t counter,
                    double time);
void write_mesh(const mesh::Mesh& mesh, const std::string filename,
                const std::size_t counter, double time);
std::string init(const mesh::Mesh& mesh, const std::string filename,
                 const std::size_t counter, std::size_t dim);
void results_write(const function::Function<PetscScalar>& u, std::string file);
void write_point_data(const function::Function<PetscScalar>& u,
                      const mesh::Mesh& mesh, std::string file);
void pvd_file_write(std::size_t step, double time, const std::string filename,
                    std::string file);
void pvtu_write_function(std::size_t dim, std::size_t rank,
                         const std::string data_location,
                         const std::string name, const std::string filename,
                         const std::string fname, const std::size_t counter,
                         std::size_t num_processes);
void pvtu_write_mesh(const std::string filename,
                     const std::string pvtu_filename, const std::size_t counter,
                     const std::size_t num_processes);
void pvtu_write(const function::Function<PetscScalar>& u,
                const std::string filename, const std::string pvtu_filename,
                const std::size_t counter);
void create_vtk_header(std::size_t num_vertices, std::size_t num_cells,
                       const std::string vtu_filename);
std::string vtu_name(const int process, const int num_processes,
                     const int counter, const std::string filename,
                     const std::string ext);
std::string strip_path(const std::string filename, const std::string file);
void pvtu_write_mesh(pugi::xml_node xml_node);

//----------------------------------------------------------------------------
void create_vtk_header(std::size_t num_vertices, std::size_t num_cells,
                       const std::string vtu_filename)
{
  // Overwrite existing file
  pugi::xml_document vtk_header;

  // Create main node
  pugi::xml_node node = vtk_header.append_child("VTKFile");
  assert(node);
  // Create sub nodes with information
  node.append_attribute("type") = "UnstructuredGrid";
  node.append_attribute("version") = "0.1";
  pugi::xml_node grid_node = node.append_child("UnstructuredGrid");
  assert(grid_node);
  pugi::xml_node piece_node = grid_node.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = num_vertices;
  piece_node.append_attribute("NumberOfCells") = num_cells;
  assert(piece_node);
  vtk_header.save_file(vtu_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
std::string vtu_name(const int process, const int num_processes,
                     const int counter, const std::string filename,
                     const std::string ext)
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(filename, 0, filename.find_last_of('.'));
  extension.assign(filename, filename.find_last_of('.'), filename.size());

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
std::string strip_path(const std::string filename, const std::string file)
{
  std::string fname;
  fname.assign(file, filename.find_last_of('/') + 1, file.size());
  return fname;
}
//----------------------------------------------------------------------------
std::string init(const mesh::Mesh& mesh, const std::string filename,
                 const std::size_t counter, std::size_t cell_dim)
{
  // Get MPI communicators
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get vtu file name and clear file
  std::string vtu_filename
      = vtu_name(dolfinx::MPI::rank(mpi_comm), dolfinx::MPI::size(mpi_comm),
                 counter, filename, ".vtu");

  // Number of cells
  const std::size_t num_cells
      = mesh.topology().index_map(cell_dim)->size_local();

  // Number of points in mesh (can be more than the number of vertices)
  const int num_nodes = mesh.geometry().x().rows();

  // Write headers
  create_vtk_header(num_nodes, num_cells, vtu_filename);

  return vtu_filename;
}
//----------------------------------------------------------------------------
void write_function(const function::Function<PetscScalar>& u,
                    const std::string filename, const std::size_t counter,
                    double time)
{
  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);

  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh->mpi_comm();

  // Get vtu file name and initialise
  std::string vtu_filename
      = init(*mesh, filename, counter, mesh->topology().dim());

  // Write mesh
  VTKWriter::write_mesh(*mesh, mesh->topology().dim(), vtu_filename);

  // Write results
  results_write(u, vtu_filename);

  // Parallel-specific files
  const std::size_t num_processes = dolfinx::MPI::size(mpi_comm);
  if (num_processes > 1 and dolfinx::MPI::rank(mpi_comm) == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, filename, ".pvtu");
    pvtu_write(u, filename, pvtu_filename, counter);
    pvd_file_write(counter, time, filename, pvtu_filename);
  }
  else if (num_processes == 1)
    pvd_file_write(counter, time, filename, vtu_filename);

  DLOG(INFO) << "Saved function \""
             << "u"
             << "\" to file \"" << filename << "\" in VTK format.";
}
//----------------------------------------------------------------------------
void write_mesh(const mesh::Mesh& mesh, const std::string filename,
                const std::size_t counter, double time)
{
  common::Timer t("Write mesh to PVD/VTK file");

  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get vtu file name and initialise out files
  std::string vtu_filename
      = init(mesh, filename, counter, mesh.topology().dim());

  // Write local mesh to vtu file
  VTKWriter::write_mesh(mesh, mesh.topology().dim(), vtu_filename);

  // Parallel-specific files
  const std::size_t num_processes = dolfinx::MPI::size(mpi_comm);
  if (num_processes > 1 and dolfinx::MPI::rank(mpi_comm) == 0)
  {
    std::string pvtu_filename = vtu_name(0, 0, counter, filename, ".pvtu");
    pvtu_write_mesh(filename, pvtu_filename, counter, num_processes);
    pvd_file_write(counter, time, filename, pvtu_filename);
  }
  else if (num_processes == 1)
    pvd_file_write(counter, time, filename, vtu_filename);

  DLOG(INFO) << "Saved mesh in VTK format to file:" << filename;
}
//----------------------------------------------------------------------------
void results_write(const function::Function<PetscScalar>& u,
                   std::string vtu_filename)
{
  // Get rank of function::Function
  const int rank = u.function_space()->element()->value_rank();
  if (rank > 2)
  {
    throw std::runtime_error(
        "Cannot write data to VTK file. "
        "Only scalar, vector and tensor functions can be saved in VTK format");
  }

  // Get number of components
  const int dim = u.function_space()->element()->value_size();

  // Check that function type can be handled
  if (rank == 1)
  {
    if (!(dim == 2 || dim == 3))
    {
      throw std::runtime_error("Cannot write data to VTK file. "
                               "Don't know how to handle vector function with "
                               "dimension other than 2 or 3");
    }
  }
  else if (rank == 2)
  {
    if (!(dim == 4 || dim == 9))
    {
      throw std::runtime_error(
          "Cannot write data to VTK file. "
          "Don't know how to handle tensor function with dimension "
          "other than 4 or 9");
    }
  }

  // Test for cell-based element type
  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);
  int cell_based_dim = 1;
  for (int i = 0; i < rank; i++)
    cell_based_dim *= mesh->topology().dim();

  std::shared_ptr<const fem::DofMap> dofmap = u.function_space()->dofmap();
  assert(dofmap);
  assert(dofmap->element_dof_layout);
  if (dofmap->element_dof_layout->num_dofs() == cell_based_dim)
    VTKWriter::write_cell_data(u, vtu_filename);
  else
    write_point_data(u, *mesh, vtu_filename);
}
//----------------------------------------------------------------------------
void write_point_data(const function::Function<PetscScalar>& u,
                      const mesh::Mesh& mesh, std::string vtu_filename)
{
  const int rank = u.function_space()->element()->value_rank();

  // Get number of components
  const int dim = u.function_space()->element()->value_size();

  // Open file

  if (!boost::filesystem::exists(vtu_filename))
    throw std::runtime_error("File " + vtu_filename + " does not exist");
  pugi::xml_document file;
  pugi::xml_parse_result result = file.load_file(vtu_filename.c_str());
  assert(result);

  // Select mesh node, Note: Could be done with xpath in the future
  pugi::xml_node node
      = file.select_node("/VTKFile/UnstructuredGrid/Piece").node();
  if (!node)
    throw std::runtime_error("XML node VTKFile/Unstructured/Piece not found.");

  pugi::xml_node pd_node = node.append_child("PointData");
  if (rank == 0)
    pd_node.append_attribute("Scalars") = u.name.c_str();
  else if (rank == 1)
    pd_node.append_attribute("Vectors") = u.name.c_str();
  else if (rank == 2)
    pd_node.append_attribute("Tensors") = u.name.c_str();

#ifdef PETSC_USE_COMPLEX
  const std::vector<std::string> components = {"real", "imag"};
#else
  const std::vector<std::string> components = {""};
#endif

  for (const auto& component : components)
  {
    std::string attr_name;
    if (component.empty())
      attr_name = u.name;
    else
      attr_name = component + "_" + u.name;

    pugi::xml_node data_node = pd_node.append_child("DataArray");

    // Set common attributes
    data_node.append_attribute("Name") = attr_name.c_str();
    data_node.append_attribute("type") = "Float64";
    data_node.append_attribute("format") = "ascii";
    if (rank == 1)
      data_node.append_attribute("NumberOfComponents") = 3;
    else if (rank == 2)
      data_node.append_attribute("NumberOfComponents") = 9;

    // Get function values at the nodes of the mesh
    Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        values = u.compute_point_values();

    const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points
        = mesh.geometry().x();

    // Create flattened point data padded to 3D
    std::vector<double> point_data(points.rows() * std::pow(3, rank), 0);
    std::int32_t k = 0;
    for (int i = 0; i < points.rows(); ++i)
    {
#ifdef PETSC_USE_COMPLEX
      if (component == "real")
      {

        if (rank == 1 and dim == 2)
        {
          // Append 0.0 to 2D vectors to make them 3D
          for (int j = 0; j < 2; j++)
            point_data[k++] = values(i, j).real();
          k++;
        }
        else if (rank == 2 and dim == 4)
        {
          // Pad 2D tensors with 0.0 to make them 3D
          for (int j = 0; j < 2; j++)
          {
            point_data[k++] = values(i, (2 * j + 0)).real();
            point_data[k++] = values(i, (2 * j + 1)).real();
            k++;
          }
          k += 3;
        }
        else
        {
          // Write all components
          for (int j = 0; j < dim; j++)
            point_data[k++] = values(i, j).real();
        }
      }
      else if (component == "imag")
      {

        if (rank == 1 and dim == 2)
        {
          // Append 0.0 to 2D vectors to make them 3D
          for (int j = 0; j < 2; j++)
            point_data[k++] = values(i, j).imag();
          k++;
        }
        else if (rank == 2 and dim == 4)
        {
          // Pad 2D tensors with 0.0 to make them 3D
          for (int j = 0; j < 2; j++)
          {
            point_data[k++] = values(i, (2 * j + 0)).imag();
            point_data[k++] = values(i, (2 * j + 1)).imag();
            k++;
          }
          k += 3;
        }
        else
        {
          // Write all components
          for (int j = 0; j < dim; j++)
            point_data[k++] = values(i, j).imag();
        }
      }
#else

      if (rank == 1 and dim == 2)
      {
        // Append 0.0 to 2D vectors to make them 3D
        for (int j = 0; j < 2; j++)
          point_data[k++] = values(i, j);
        k++;
      }
      else if (rank == 2 and dim == 4)
      {
        // Pad 2D tensors with 0.0 to make them 3D
        for (int j = 0; j < 2; j++)
        {
          point_data[k++] = values(i, (2 * j + 0));
          point_data[k++] = values(i, (2 * j + 1));
          k++;
        }
        k += 3;
      }
      else
      {
        // Write all components
        for (int j = 0; j < dim; j++)
          point_data[k++] = values(i, j);
      }
#endif
    }
    // namespace
    std::int32_t linebreak = rank > 0 ? std::pow(3, rank) : 0;
    data_node.append_child(pugi::node_pcdata)
        .set_value(common::container_to_string(point_data, " ", 16, linebreak)
                       .c_str());
  }
  file.save_file(vtu_filename.c_str(), "  ");
} // namespace
//----------------------------------------------------------------------------
void pvd_file_write(std::size_t step, double time, const std::string filename,
                    std::string fname)
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
      throw std::runtime_error(
          "Cannot write data to VTK file. "
          "XML parsing error when reading from existing file");
    }
  }

  // Remove directory path from name for pvd file
  const std::string fname_strip = strip_path(filename, fname);

  // Get Collection node
  pugi::xml_node xml_collections = xml_doc.child("VTKFile").child("Collection");
  assert(xml_collections);

  // Append data set
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = fname_strip.c_str();

  // Save file
  xml_doc.save_file(filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void pvtu_write_mesh(pugi::xml_node xml_node)
{
  // mesh::Vertex data
  pugi::xml_node vertex_data_node = xml_node.append_child("PPoints");
  pugi::xml_node data_node = vertex_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
  data_node.append_attribute("NumberOfComponents") = "3";

  // Cell data
  pugi::xml_node cell_data_node = xml_node.append_child("PCellData");

  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int32";
  data_node.append_attribute("Name") = "connectivity";

  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int32";
  data_node.append_attribute("Name") = "offsets";

  data_node = cell_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Int8";
  data_node.append_attribute("Name") = "types";
}
//----------------------------------------------------------------------------
void pvtu_write_function(std::size_t dim, std::size_t rank,
                         const std::string data_location,
                         const std::string name, const std::string filename,
                         const std::string fname, const std::size_t counter,
                         std::size_t num_processes)
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
      throw std::runtime_error("Cannot write data to VTK file. "
                               "Don't know how to handle vector function with "
                               "dimension other than 2 or 3");
    }
    num_components = 3;
  }
  else if (rank == 2)
  {
    rank_type = "Tensors";
    if (!(dim == 4 || dim == 9))
    {
      throw std::runtime_error(
          "Cannot write data to VTK file. "
          "Don't know how to handle tensor function with dimension "
          "other than 4 or 9");
    }
    num_components = 9;
  }
  else
  {
    throw std::runtime_error("Cannot handle XML output of rank "
                             + std::to_string(rank));
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
        = vtu_name(i, num_processes, counter, filename, ".vtu");
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = tmp_string.c_str();
  }

  xml_doc.save_file(fname.c_str(), "  ");
}
//----------------------------------------------------------------------------
void pvtu_write_mesh(const std::string filename, const std::string fname,
                     const std::size_t counter, const std::size_t num_processes)
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
        = vtu_name(i, num_processes, counter, filename, ".vtu");
    pugi::xml_node piece_node = grid_node.append_child("Piece");
    piece_node.append_attribute("Source") = tmp_string.c_str();
  }

  xml_doc.save_file(fname.c_str(), "  ");
}
//----------------------------------------------------------------------------
void pvtu_write(const function::Function<PetscScalar>& u,
                const std::string filename, const std::string fname,
                const std::size_t counter)
{
  assert(u.function_space()->element());
  const int rank = u.function_space()->element()->value_rank();
  if (rank > 2)
  {
    throw std::runtime_error("Only scalar, vector and tensor functions can "
                             "be saved in VTK format");
  }

  // Get number of components
  const int dim = u.function_space()->element()->value_size();

  // Get mesh
  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);

  // Test for cell-based element type
  std::string data_type = "point";
  int cell_based_dim = 1;
  assert(u.function_space()->dofmap());
  for (int i = 0; i < rank; i++)
    cell_based_dim *= mesh->topology().dim();
  assert(u.function_space()->dofmap());
  assert(u.function_space()->dofmap()->element_dof_layout);
  if (u.function_space()->dofmap()->element_dof_layout->num_dofs()
      == cell_based_dim)
  {
    data_type = "cell";
  }

  const int num_processes = dolfinx::MPI::size(mesh->mpi_comm());
  pvtu_write_function(dim, rank, data_type, u.name.c_str(), filename, fname,
                      counter, num_processes);
}
//----------------------------------------------------------------------------

} // namespace

//----------------------------------------------------------------------------
VTKFile::VTKFile(const std::string filename) : _filename(filename), _counter(0)
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::Mesh& mesh)
{
  write_mesh(mesh, _filename, _counter, _counter);
  ++_counter;
}
//----------------------------------------------------------------------------
void VTKFile::write(const function::Function<PetscScalar>& u)
{
  write_function(u, _filename, _counter, _counter);
  ++_counter;
}
//----------------------------------------------------------------------------
void VTKFile::write(const mesh::Mesh& mesh, double time)
{
  write_mesh(mesh, _filename, _counter, time);
  ++_counter;
}
//----------------------------------------------------------------------------
void VTKFile::write(const function::Function<PetscScalar>& u, double time)
{
  write_function(u, _filename, _counter, time);
  ++_counter;
}
//----------------------------------------------------------------------------
