// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-07-19
// Last changed:

#include <fstream>
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
#include <dolfin/la/LinearAlgebraFactory.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include "Encoder.h"
#include "VTKWriter.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void VTKWriter::write_mesh(const Mesh& mesh, std::string filename,
                           bool binary, bool compress)
{
  if (binary)
    write_base64_mesh(mesh, filename, compress);
  else
    write_ascii_mesh(mesh, filename);
}
//----------------------------------------------------------------------------
void VTKWriter::write_cell_data(const Function& u, std::string filename,
                                bool binary, bool compress)
{
  std::string encode_string;
  if (!binary)
    encode_string = "ascii";
  else
    encode_string = "binary";

  // For brevity
  const Mesh& mesh = u.function_space().mesh();
  const GenericDofMap& dofmap = u.function_space().dofmap();

  // Get rank of Function
  const uint rank = u.value_rank();
  if(rank > 2)
    error("Only scalar, vector and tensor functions can be saved in VTK format.");

  // Get number of components
  uint dim = 1;
  for (uint i = 0; i < rank; i++)
    dim *= u.value_dimension(i);

  // Open file
  std::ofstream fp(filename.c_str(), std::ios_base::app);

  // Write headers
  if (rank == 0)
  {
    fp << "<CellData  Scalars=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  format=\""<< encode_string <<"\">" << std::endl;
  }
  else if (rank == 1)
  {
    fp << "<CellData  Vectors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"3\" format=\""<< encode_string <<"\">" << std::endl;
  }
  else if (rank == 2)
  {
    fp << "<CellData  Tensors=\"" << u.name() << "\"> " << std::endl;
    fp << "<DataArray  type=\"Float32\"  Name=\"" << u.name() << "\"  NumberOfComponents=\"9\" format=\""<< encode_string <<"\">" << std::endl;
  }

  // Allocate memory for function values at cell centres
  const uint size = mesh.num_cells()*dim;
  std::vector<uint> dofs(dofmap.max_local_dimension());

  // Build lists of dofs and create map
  UFCCell ufc_cell(mesh);
  std::vector<uint> dof_set;
  std::vector<uint> offset(size+1);
  std::vector<uint>::iterator cell_offset = offset.begin();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Tabulate dofs
    ufc_cell.update(*cell);
    dofmap.tabulate_dofs(&dofs[0], ufc_cell, cell->index());
    for(uint i = 0; i < dofmap.local_dimension(ufc_cell); ++i)
      dof_set.push_back(dofs[i]);

    // Add local dimension to cell offset and increment
    *(cell_offset + 1) = *(cell_offset) + dofmap.local_dimension(ufc_cell);
    ++cell_offset;
  }

  // Get  values
  std::vector<double> values(dof_set.size());
  u.vector().get(&values[0], dof_set.size(), &dof_set[0]);

  // Get cell data
  if (!binary)
    fp << ascii_cell_data(mesh, offset, values, dim, rank);
  else
    fp << base64_cell_data(mesh, offset, values, dim, rank, compress) << std::endl;

  fp << "</DataArray> " << std::endl;
  fp << "</CellData> " << std::endl;
}
//----------------------------------------------------------------------------
std::string VTKWriter::ascii_cell_data(const Mesh& mesh,
                                       const std::vector<uint>& offset,
                                       const std::vector<double>& values,
                                       uint dim, uint rank)
{
  std::ostringstream ss;
  ss << std::scientific;
  std::vector<uint>::const_iterator cell_offset = offset.begin();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (rank == 1 && dim == 2)
    {
      // Append 0.0 to 2D vectors to make them 3D
      ss << " " << values[*cell_offset];
      ss << " " << values[*cell_offset+1];
      ss << " " << 0.0;
    }
    else if (rank == 2 && dim == 4)
    {
      // Pad with 0.0 to 2D tensors to make them 3D
      for(uint i = 0; i < 2; i++)
      {
        ss << " " << values[*cell_offset + 2*i];
        ss << " " << values[*cell_offset + 2*i+1];
        ss << " " << 0.0;
      }
      ss << " " << 0.0;
      ss << " " << 0.0;
      ss << " " << 0.0;
    }
    else
    {
      // Write all components
      for (uint i = 0; i < dim; i++)
        ss << " " << values[*cell_offset + i];
    }
    ss << std::endl;
    ++cell_offset;
  }

  return ss.str();
}
//----------------------------------------------------------------------------
std::string VTKWriter::base64_cell_data(const Mesh& mesh,
                                        const std::vector<uint>& offset,
                                        const std::vector<double>& values,
                                        uint dim, uint rank, bool compress)
{
  const uint size = mesh.num_cells()*dim;

  std::vector<float> data;
  if (rank == 1 && dim == 2)
    data.resize(size + size/2);
  else if (rank == 2 && dim == 4)
    data.resize(size + 4*size/5);
  else
    data.resize(size);

  std::vector<uint>::const_iterator cell_offset = offset.begin();
  std::vector<float>::iterator entry = data.begin();
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (rank == 1 && dim == 2)
    {
      // Append 0.0 to 2D vectors to make them 3D
      *entry++ = values[*cell_offset];
      *entry++ = values[*cell_offset + 1];
      *entry++ = 0.0;
    }
    else if (rank == 2 && dim == 4)
    {
      // Pad with 0.0 to 2D tensors to make them 3D
      for(uint i = 0; i < 2; i++)
      {
        *entry++ = values[*cell_offset + 2*i];
        *entry++ = values[*cell_offset + 2*i+1];
        *entry++ = 0.0;
      }
      *entry++ = 0.0;
      *entry++ = 0.0;
      *entry++ = 0.0;
    }
    else
    {
      // Write all components
      for (uint i = 0; i < dim; i++)
        *entry++ = values[*cell_offset + i];
    }
    ++cell_offset;
  }

  return encode_stream(data, compress);
}
//----------------------------------------------------------------------------
void VTKWriter::write_ascii_mesh(const Mesh& mesh, std::string filename)
{
  // Open file
  std::ofstream file(filename.c_str(), std::ios::app);
  if ( !file.is_open() )
    error("Unable to open file %s", filename.c_str());

  // Write vertex positions
  file << "<Points>" << std::endl;
  file << "<DataArray  type=\"Float32\"  NumberOfComponents=\"3\"  format=\"" << "ascii" << "\">" << std::endl;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    file << p.x() << " " << p.y() << " " <<  p.z() << std::endl;
  }
  file << "</DataArray>" << std::endl <<  "</Points>" << std::endl;

  // Write cell connectivity
  file << "<Cells>" << std::endl;
  file << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\"" << "ascii" << "\">" << std::endl;
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (VertexIterator v(*c); !v.end(); ++v)
      file << v->index() << " ";
    file << std::endl;
  }
  file << "</DataArray>" << std::endl;

  // Write offset into connectivity array for the end of each cell
  file << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\"" << "ascii" << "\">" << std::endl;
  const uint num_cell_vertices = mesh.type().num_entities(0);
  for (uint offsets = 1; offsets <= mesh.num_cells(); offsets++)
    file << " " << offsets*num_cell_vertices << "  "  << std::endl;
  file << "</DataArray>" << std::endl;

  // Write cell type
  file << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"" << "ascii" << "\">" << std::endl;
  uint vtk_cell_type = 0;
  if (mesh.type().cell_type() == CellType::tetrahedron)
    vtk_cell_type = 10;
  if (mesh.type().cell_type() == CellType::triangle)
    vtk_cell_type = 5;
  if (mesh.type().cell_type() == CellType::interval)
    vtk_cell_type = 3;

  for (uint types = 0; types < mesh.num_cells(); types++)
    file << " " << vtk_cell_type << std::endl;
  file  << "</DataArray>" << std::endl;
  file  << "</Cells>" << std::endl;

  // Close file
  file.close();
}
//-----------------------------------------------------------------------------
void VTKWriter::write_base64_mesh(const Mesh& mesh, std::string filename,
                                 bool compress)
{
  // Open file
  std::ofstream file(filename.c_str(), std::ios::app);
  if ( !file.is_open() )
    error("Unable to open file %s", filename.c_str());

  // Write vertex positions
  file << "<Points>" << std::endl;
  file << "<DataArray  type=\"Float32\"  NumberOfComponents=\"3\"  format=\"" << "binary" << "\">" << std::endl;
  std::vector<float> vertex_data(3*mesh.num_vertices());
  std::vector<float>::iterator vertex_entry = vertex_data.begin();
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    *vertex_entry++ = p.x();
    *vertex_entry++ = p.y();
    *vertex_entry++ = p.z();
  }
  // Create encoded stream
  file <<  encode_stream(vertex_data, compress) << std::endl;
  file << "</DataArray>" << std::endl <<  "</Points>" << std::endl;

  // Write cell connectivity
  file << "<Cells>" << std::endl;
  file << "<DataArray  type=\"UInt32\"  Name=\"connectivity\"  format=\"" << "binary" << "\">" << std::endl;
  const int size = mesh.num_cells()*mesh.type().num_entities(0);
  std::vector<boost::uint32_t> cell_data(size);
  std::vector<boost::uint32_t>::iterator cell_entry = cell_data.begin();
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (VertexIterator v(*c); !v.end(); ++v)
      *cell_entry++ = v->index();
  }

  // Create encoded stream
  file << encode_stream(cell_data, compress) << std::endl;
  file << "</DataArray>" << std::endl;

  // Write offset into connectivity array for the end of each cell
  file << "<DataArray  type=\"UInt32\"  Name=\"offsets\"  format=\"" << "binary" << "\">" << std::endl;
  const uint num_cell_vertices = mesh.type().num_entities(0);
  std::vector<boost::uint32_t> offset_data(mesh.num_cells()*num_cell_vertices);
  std::vector<boost::uint32_t>::iterator offset_entry = offset_data.begin();
  for (uint offsets = 1; offsets <= mesh.num_cells(); offsets++)
    *offset_entry++ = offsets*num_cell_vertices;

  // Create encoded stream
  file << encode_stream(offset_data, compress) << std::endl;
  file << "</DataArray>" << std::endl;

  // Write cell type
  file << "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"" << "binary" << "\">" << std::endl;
  boost::uint8_t vtk_cell_type = 0;
  if (mesh.type().cell_type() == CellType::tetrahedron)
    vtk_cell_type = boost::uint8_t(10);
  if (mesh.type().cell_type() == CellType::triangle)
    vtk_cell_type = boost::uint8_t(5);
  if (mesh.type().cell_type() == CellType::interval)
    vtk_cell_type = boost::uint8_t(3);

  std::vector<boost::uint8_t> type_data(mesh.num_cells());
  std::vector<boost::uint8_t>::iterator type_entry = type_data.begin();
  for (uint types = 0; types < mesh.num_cells(); types++)
    *type_entry++ = vtk_cell_type;

  // Create encoded stream
  file << encode_stream(type_data, compress) << std::endl;

  file  << "</DataArray>" << std::endl;
  file  << "</Cells>" << std::endl;

  // Close file
  file.close();
}
//----------------------------------------------------------------------------
template<typename T>
std::string VTKWriter::encode_stream(const std::vector<T>& data,
                                           bool compress)
{
  std::stringstream stream;

  if (compress)
  {
  #ifdef HAS_ZLIB
    return encode_inline_compressed_base64(data);
  #else
    warning("zlib must be configured to enable compressed VTK output. Using uncompressed base64 encoding instead.");
    return encode_inline_base64(data);
  #endif
  }
  else
    return encode_inline_base64(data);
}
//----------------------------------------------------------------------------
template<typename T>
std::string VTKWriter::encode_inline_base64(const std::vector<T>& data)
{
  std::stringstream stream;

  const boost::uint32_t size = data.size()*sizeof(T);
  Encoder::encode_base64(&size, 1, stream);
  Encoder::encode_base64(data, stream);

  return stream.str();
}
//----------------------------------------------------------------------------
#ifdef HAS_ZLIB
template<typename T>
std::string VTKWriter::encode_inline_compressed_base64(const std::vector<T>& data)
{
  std::stringstream stream;

  boost::uint32_t header[4];
  header[0] = 1;
  header[1] = data.size()*sizeof(T);
  header[2] = 0;

  // Compress data
  std::pair<boost::shared_array<unsigned char>, dolfin::uint> compressed_data = Encoder::compress_data(data);

  // Length of compressed data
  header[3] = compressed_data.second;

  // Encode header
  Encoder::encode_base64(&header[0], 4, stream);

  // Encode data
  Encoder::encode_base64(compressed_data.first.get(), compressed_data.second, stream);

  return stream.str();
}
#endif
//----------------------------------------------------------------------------
