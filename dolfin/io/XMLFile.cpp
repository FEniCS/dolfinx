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
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-03
// Last changed: 2011-06-30

#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "pugixml.hpp"

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include "XMLFunctionPlotData.h"
#include "XMLLocalMeshData.h"
#include "XMLLocalMeshDataDistributed.h"
#include "XMLMesh.h"
#include "XMLMeshFunction.h"
#include "XMLParameters.h"
#include "XMLVector.h"
#include "XMLFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFile::XMLFile(const std::string filename) : GenericFile(filename),
                                               outstream(new std::ofstream)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLFile::XMLFile(std::ostream& s) : GenericFile(""), outstream(&s, NoDeleter())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Mesh& input_mesh)
{
  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc, filename);

  // Read mesh
  XMLMesh::read(input_mesh, dolfin_node);
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Mesh& output_mesh)
{
  if (MPI::num_processes() > 1)
    error("Mesh XML output in parallel not yet supported");

  // Open file on process 0 for distributed objects and on all processes
  // for local objects
  open_file();

  // Note: 'write' is being called on all processes since collective MPI
  // calls might be used.
  XMLMesh::write(output_mesh, *outstream, 1);

  // Close file
  close_file();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (LocalMeshData& input_data)
{
  XMLLocalMeshDataDistributed xml_object(input_data, filename);
  xml_object.read();

  /*
  std::cout << "Read local mesh data" << std::endl;
  if (MPI::process_number() == 0)
  {
    pugi::xml_document xml_doc;
    const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc, filename);
    XMLLocalMeshData::read(input_data, dolfin_node);
  }
  else
  {
    const pugi::xml_node dolfin_node(0);
    XMLLocalMeshData::read(input_data, dolfin_node);
  }
  */
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const LocalMeshData& output_data)
{
  error("Output of LocalMeshData not supported");
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (GenericVector& input)
{
  pugi::xml_document xml_doc;
  pugi::xml_node dolfin_node(0);

  // Read vector size
  uint size = 0;
  if (MPI::process_number() == 0)
  {
    dolfin_node = get_dolfin_xml_node(xml_doc, filename);
    size = XMLVector::read_size(dolfin_node);
  }
  MPI::broadcast(size);

  // Resize if necessary
  if (MPI::num_processes() > 1 && input.size() != size)
    warning("Resizing parallel vector. Default partitioning will be used. To control distribution, initialize vector size before reading from file.");
  if (input.size() != size)
    input.resize(size);

  // Read vector on root process
  if (MPI::process_number() == 0)
  {
    assert(dolfin_node);
    XMLVector::read(input, dolfin_node);
  }

  // Finalise
  input.apply("insert");
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const GenericVector& output)
{
  // Open file on process 0 for distributed objects and on all processes
  // for local objects
  if (MPI::process_number() == 0)
    open_file();

  // Note: 'write' is being called on all processes since collective MPI
  // calls might be used.
  XMLVector::write(output, *outstream, 1);

  // Close file
  if (MPI::process_number() == 0)
    close_file();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Parameters& input)
{
  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc, filename);

  // Read parameters
  XMLParameters::read(input, dolfin_node);
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Parameters& output)
{
  if (MPI::process_number() == 0)
    open_file();

  XMLParameters::write(output, *outstream, 1);

  if (MPI::process_number() == 0)
    close_file();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (FunctionPlotData& input)
{
  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc, filename);

  // Read plot data
  XMLFunctionPlotData::read(input, dolfin_node);
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const FunctionPlotData& output)
{
  if (MPI::process_number() == 0)
    open_file();

  XMLFunctionPlotData::write(output, *outstream, 1);

  if (MPI::process_number() == 0)
    close_file();
}
//-----------------------------------------------------------------------------
void XMLFile::write_start(std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl << std::endl;
  outfile << indent() << "<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLFile::write_end(std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "</dolfin>" << std::endl;
}
//-----------------------------------------------------------------------------
template<class T> void XMLFile::read_mesh_function(MeshFunction<T>& t,
                                                  const std::string type) const
{
  // Create XML doc and get DOLFIN node
  pugi::xml_document xml_doc;
  const pugi::xml_node dolfin_node = get_dolfin_xml_node(xml_doc, filename);

  // Read MeshFunction
  XMLMeshFunction::read(t, type, dolfin_node);
}
//-----------------------------------------------------------------------------
template<class T> void XMLFile::write_mesh_function(const MeshFunction<T>& t,
                                                  const std::string type)
{
  if (MPI::process_number() == 0)
    open_file();

  XMLMeshFunction::write(t, type, *outstream, 1);

  if (MPI::process_number() == 0)
    close_file();
}
//-----------------------------------------------------------------------------
const pugi::xml_node XMLFile::get_dolfin_xml_node(pugi::xml_document& xml_doc,
                                                  const std::string filename) const
{
  // Create XML parser result
  pugi::xml_parse_result result;

  // Get file path and extension
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);

  // FIXME: Check that file exists
  if (!boost::filesystem::is_regular_file(filename))
    error("File \"%s\" does not exist or is not a regular file. Cannot be read by XML parser.", filename.c_str());

  // Load xml file (unzip if necessary) into parser
  if (extension == ".gz")
  {
    // Decompress file
    std::ifstream file(filename.c_str(), std::ios_base::in|std::ios_base::binary);
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
    result = xml_doc.load_file(filename.c_str());

  // Check that we have a DOLFIN XML file
  const pugi::xml_node dolfin_node = xml_doc.child("dolfin");
  if (!dolfin_node)
    error("Not a DOLFIN XML file");

  return dolfin_node;
}
//-----------------------------------------------------------------------------
void XMLFile::open_file()
{
  // Convert to ofstream
  std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream.get());
  if (outfile)
  {
    // Open file
    outfile->open(filename.c_str());

    // Go to end of file
    outfile->seekp(0, std::ios::end);
  }
  XMLFile::write_start(*outstream);
}
//-----------------------------------------------------------------------------
void XMLFile::close_file()
{
  XMLFile::write_end(*outstream);

  // Get file path and extension
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);
  if (extension == ".gz")
  {
    error("Compressed XML output not yet supported.");
  }
  else
  {
    // Convert to ofstream
    std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream.get());
    if (outfile)
      outfile->close();
  }
}
//-----------------------------------------------------------------------------
