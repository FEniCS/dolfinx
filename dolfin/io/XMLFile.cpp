// Copyright (C) 2009-2011 Ola Skavhaug and Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2009-03-03
// Last changed: 2013-05-21

#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "pugixml.hpp"

#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/LocalMeshValueCollection.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "XMLFunctionData.h"
#include "XMLMesh.h"
#include "XMLMeshFunction.h"
#include "XMLMeshValueCollection.h"
#include "XMLParameters.h"
#include "XMLTable.h"
#include "XMLVector.h"
#include "XMLFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFile::XMLFile(MPI_Comm mpi_comm, const std::string filename)
  : GenericFile(filename, "XML"), _mpi_comm(mpi_comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLFile::XMLFile(std::ostream& s) : GenericFile("", "XML"),
                                    outstream(&s, NoDeleter()),
                                    _mpi_comm(MPI_COMM_SELF)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLFile::~XMLFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Mesh& input_mesh)
{
  if (MPI::rank(input_mesh.mpi_comm()) == 0)
  {
    // Create XML doc and get DOLFIN node
    pugi::xml_document xml_doc;
    load_xml_doc(xml_doc);
    pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);

    // Read mesh
    XMLMesh::read(input_mesh, dolfin_node);
  }

  if (MPI::size(input_mesh.mpi_comm()) > 1)
  {
    // Distribute local data
    input_mesh.domains().clear();
    LocalMeshData local_mesh_data(input_mesh);

    // FIXME: To be removed when advanced parallel XML IO is removed
    // Add mesh domain data
    if (MPI::rank(input_mesh.mpi_comm()) == 0)
    {
      // Create XML doc and get DOLFIN node
      pugi::xml_document xml_doc;
      load_xml_doc(xml_doc);
      pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);
      XMLMesh::read_domain_data(local_mesh_data, dolfin_node);
    }

    // Partition and build mesh
    const std::string ghost_mode = dolfin::parameters["ghost_mode"];
    MeshPartitioning::build_distributed_mesh(input_mesh, local_mesh_data,
                                             ghost_mode);
  }
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Mesh& output_mesh)
{
  if (MPI::size(output_mesh.mpi_comm()) > 1)
  {
    dolfin_error("XMLFile.cpp",
                 "write mesh to XML file in parallel",
                 "Parallel XML mesh output is not supported. Use HDF5 format instead");
  }

  pugi::xml_document doc;
  pugi::xml_node node = write_dolfin(doc);
  XMLMesh::write(output_mesh, node);
  save_xml_doc(doc);
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (GenericVector& input)
{
  pugi::xml_document xml_doc;
  pugi::xml_node dolfin_node(0);

  // Read vector size
  std::size_t size = 0;
  if (MPI::rank(_mpi_comm) == 0)
  {
    load_xml_doc(xml_doc);
    dolfin_node = get_dolfin_xml_node(xml_doc);
    size = XMLVector::read_size(dolfin_node);
  }
  MPI::broadcast(_mpi_comm, size);

  // Resize if necessary
  const std::size_t input_vector_size = input.size();
  const std::size_t num_proc = MPI::size(_mpi_comm);
  if (num_proc > 1 && input_vector_size != size)
  {
    warning("Resizing parallel vector. Default partitioning will be used. \
To control distribution, initialize vector size before reading from file.");
  }
  if (input.size() != size)
    input.init(size);

  // Read vector on root process
  if (MPI::rank(_mpi_comm) == 0)
  {
    dolfin_assert(dolfin_node);
    XMLVector::read(input, dolfin_node);
  }

  // Finalise
  input.apply("insert");
}
//-----------------------------------------------------------------------------
void XMLFile::read_vector(std::vector<double>& input,
                          std::vector<dolfin::la_index>& indices)
{
  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  load_xml_doc(xml_doc);
  const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);

  // Read parameters
  XMLVector::read(input, indices, dolfin_node);
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const GenericVector& output)
{
  // Open file on process 0 for distributed objects and on all
  // processes for local objects
  if (MPI::rank(_mpi_comm) == 0)
  {
    pugi::xml_document doc;
    pugi::xml_node node = write_dolfin(doc);
    XMLVector::write(output, node, true);
    save_xml_doc(doc);
  }
  else
  {
    pugi::xml_node node(0);
    XMLVector::write(output, node, false);
  }
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Parameters& input)
{
  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  load_xml_doc(xml_doc);
  const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);

  // Read parameters
  XMLParameters::read(input, dolfin_node);
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Parameters& output)
{
  if (MPI::rank(_mpi_comm) == 0)
  {
    pugi::xml_document doc;
    pugi::xml_node node = write_dolfin(doc);
    XMLParameters::write(output, node);
    save_xml_doc(doc);
  }
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Table& input)
{
  if (MPI::size(_mpi_comm) > 1)
    dolfin_error("XMLFile.cpp",
                 "read table into XML file",
                 "XMLTable is not colletive. Use separate XMLFile with "
                 "MPI_COMM_SELF on each process or single process only");

  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  load_xml_doc(xml_doc);
  const pugi::xml_node node = get_dolfin_xml_node(xml_doc);

  // Read into Table 'input'
  XMLTable::read(input, node);
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Table& output)
{
  if (MPI::size(_mpi_comm) > 1)
    dolfin_error("XMLFile.cpp",
                 "write table to XML file",
                 "XMLTable is not colletive. Use separate XMLFile with "
                 "MPI_COMM_SELF on each process or single process only");
  pugi::xml_document doc;
  load_xml_doc(doc);
  pugi::xml_node node = write_dolfin(doc);
  XMLTable::write(output, node);
  save_xml_doc(doc);
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Function& input)
{
  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  pugi::xml_node dolfin_node(0);
  if (MPI::rank(_mpi_comm) == 0)
  {
    load_xml_doc(xml_doc);
    dolfin_node = get_dolfin_xml_node(xml_doc);
  }

  // Read data
  XMLFunctionData::read(input, dolfin_node);
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Function& output)
{
  if (MPI::rank(_mpi_comm) == 0)
  {
    pugi::xml_document doc;
    pugi::xml_node node = write_dolfin(doc);
    XMLFunctionData::write(output, node);
    save_xml_doc(doc);
  }
  else
  {
    pugi::xml_node node(0);
    XMLFunctionData::write(output, node);
  }
}
//-----------------------------------------------------------------------------
template<typename T>
void XMLFile::read_mesh_function(MeshFunction<T>& t,
                                 const std::string type) const
{
  if (MPI::size(_mpi_comm) == 1)
  {
    pugi::xml_document xml_doc;
    load_xml_doc(xml_doc);
    pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);
    XMLMeshFunction::read(t, type, dolfin_node);
  }
  else
  {
    // Read a MeshValueCollection on process 0, then communicate to
    // other procs
    std::size_t dim = 0;
    MeshValueCollection<T> mvc(t.mesh());
    if (MPI::rank(_mpi_comm) == 0)
    {
      pugi::xml_document xml_doc;
      load_xml_doc(xml_doc);
      pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);
      XMLMeshFunction::read(mvc, type, dolfin_node);
      dim = mvc.dim();
      MPI::broadcast(_mpi_comm, dim);
    }
    else
    {
      MPI::broadcast(_mpi_comm, dim);
      mvc.init(dim);
    }

    // Broadcast and set dimension

    // Build local data
    LocalMeshValueCollection<T> local_data(_mpi_comm, mvc, dim);

    // Distribute MeshValueCollection
    MeshPartitioning::build_distributed_value_collection<T>(mvc, local_data,
                                                            *t.mesh());

    // Assign collection to mesh function (this is a local operation)
    t = mvc;
  }
}
//-----------------------------------------------------------------------------
template<typename T>
void XMLFile::write_mesh_function(const MeshFunction<T>& t,
                                  const std::string type)
{
  not_working_in_parallel("MeshFunction XML output in parallel not yet supported.");

  pugi::xml_document xml_doc;
  pugi::xml_node node = write_dolfin(xml_doc);
  XMLMeshFunction::write(t, type, node, false);
  save_xml_doc(xml_doc);
}
//-----------------------------------------------------------------------------
template<typename T>
void XMLFile::read_mesh_value_collection(MeshValueCollection<T>& t,
                                         const std::string type) const
{
  if (MPI::size(_mpi_comm) == 1)
  {
    pugi::xml_document xml_doc;
    load_xml_doc(xml_doc);
    const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);
    XMLMeshValueCollection::read(t, type, dolfin_node);
  }
  else
  {
    // Read file on process 0
    MeshValueCollection<T> tmp_collection(t.mesh());
    if (MPI::rank(_mpi_comm) == 0)
    {
      pugi::xml_document xml_doc;
      load_xml_doc(xml_doc);
      const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc);
      XMLMeshValueCollection::read(tmp_collection, type, dolfin_node);
      std::size_t dim = (tmp_collection.dim());
      MPI::broadcast(_mpi_comm, dim);
    }
    else
    {
      std::size_t dim = 0;
      MPI::broadcast(_mpi_comm, dim);
      tmp_collection.init(dim);
    }

    // Create local data and build value collection
    LocalMeshValueCollection<T> local_data(_mpi_comm, tmp_collection,
                                           tmp_collection.dim());

    // Build mesh value collection
    dolfin_assert(t.mesh());
    t.init(tmp_collection.dim());
    MeshPartitioning::build_distributed_value_collection(t, local_data,
                                                         *t.mesh());
  }
}
//-----------------------------------------------------------------------------
template<typename T>
void XMLFile::write_mesh_value_collection(const MeshValueCollection<T>& t,
                                          const std::string type)
{
  not_working_in_parallel("MeshValueCollection XML output in parallel not yet supported.");

  pugi::xml_document xml_doc;
  pugi::xml_node node = write_dolfin(xml_doc);
  XMLMeshValueCollection::write(t, type, node);
  save_xml_doc(xml_doc);
}
//-----------------------------------------------------------------------------
void XMLFile::load_xml_doc(pugi::xml_document& xml_doc) const
{
  // Create XML parser result
  pugi::xml_parse_result result;

  // Get file path and extension
  const boost::filesystem::path path(_filename);
  const std::string extension = boost::filesystem::extension(path);

  // Check that file exists
  if (!boost::filesystem::is_regular_file(_filename))
  {
    dolfin_error("XMLFile.cpp",
                 "read data from XML file",
                 "Unable to open file \"%s\"", _filename.c_str());
  }

  // Get file size if running in parallel
  if (dolfin::MPI::size(_mpi_comm) > 1)
  {
    const double size = boost::filesystem::file_size(path)/(1024.0*1024.0);

    // Print warning if file size is greater than threshold
    const std::size_t warning_size
      = dolfin::parameters["warn_on_xml_file_size"];
    if(size >= warning_size)
    {
      warning("XML file '%s' is very large. XML files are parsed in serial, \
which is not scalable. Use XMDF/HDF5 for scalable IO in parallel",
              path.filename().c_str());
    }
  }

  // Load xml file (unzip if necessary) into parser
  if (extension == ".gz")
  {
    // Decompress file
    std::ifstream
      file(_filename.c_str(), std::ios_base::in|std::ios_base::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(file);

    // FIXME: Is this the best way to do it?
    std::stringstream dst;
    boost::iostreams::copy(in, dst);

    // Load data
    result = xml_doc.load(dst);
  }
  else
    result = xml_doc.load_file(_filename.c_str());

  // Check the XML file was opened successfully, allow empty file
  if (!result && result.status != pugi::status_no_document_element)
  {
    dolfin_error("XMLFile.cpp",
                 "read data from XML file",
                 "Error while parsing XML with status \"%s\"",
                 result.description());
  }
}
//-----------------------------------------------------------------------------
void XMLFile::save_xml_doc(const pugi::xml_document& xml_doc) const
{
  if (outstream)
    xml_doc.save(*outstream, "  ");
  else
  {
    // Compress if filename has extension '.gz'
    const boost::filesystem::path path(_filename);
    const std::string extension = boost::filesystem::extension(path);
    if (extension == ".gz")
    {
      std::stringstream xml_stream;
      xml_doc.save(xml_stream, "  ");

      std::ofstream
        file(_filename.c_str(), std::ios_base::out | std::ios_base::binary);
      boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
      out.push(boost::iostreams::gzip_compressor());
      out.push(file);
      boost::iostreams::copy(xml_stream, out);
    }
    else
      xml_doc.save_file(_filename.c_str(), "  ");
  }
}
//-----------------------------------------------------------------------------
const pugi::xml_node
XMLFile::get_dolfin_xml_node(pugi::xml_document& xml_doc) const
{
  // Check that we have a DOLFIN XML file
  const pugi::xml_node dolfin_node = xml_doc.child("dolfin");
  if (!dolfin_node)
  {
    dolfin_error("XMLFile.cpp",
                 "read data from XML file",
                 "Not a DOLFIN XML file");
  }
  return dolfin_node;
}
//-----------------------------------------------------------------------------
pugi::xml_node XMLFile::write_dolfin(pugi::xml_document& xml_doc)
{
  pugi::xml_node node = xml_doc.child("dolfin");
  if (!node)
  {
    node = xml_doc.append_child("dolfin");
    node.append_attribute("xmlns:dolfin") = "http://fenicsproject.org";
  }
  return node;
}
//-----------------------------------------------------------------------------
