// Copyright (C) 2009 Ola Skavhaug
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
// First added:  2009-03-09
// Last changed: 2011-03-28

#include "dolfin/mesh/MeshData.h"
#include "XMLMeshFunction.h"
#include "XMLIndent.h"
#include "XMLArray.h"
#include "XMLMap.h"
#include "XMLMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMeshData::XMLMeshData(MeshData& data, XMLFile& parser, bool inside)
  : XMLHandler(parser), data(data), state(OUTSIDE), type(UNSET), entity_name(""),
    xml_array(0), xml_map(0), xml_mesh_function(0),
    im(0), um(0), dm(0), iam(0), uam(0), dam(0)
{
  if (inside)
    state = INSIDE_DATA;
}
//-----------------------------------------------------------------------------
XMLMeshData::~XMLMeshData()
{
  delete im;
  delete um;
  delete dm;
  delete iam;
  delete uam;
  delete dam;
  delete xml_array;
  delete xml_map;
  delete xml_mesh_function;
}
//-----------------------------------------------------------------------------
void XMLMeshData::start_element (const xmlChar* name, const xmlChar** attrs)
{
  switch (state)
  {
  case OUTSIDE:

    if (xmlStrcasecmp(name, (xmlChar *) "data") == 0)
      state = INSIDE_DATA;

    break;

  case INSIDE_DATA:

    if (xmlStrcasecmp(name, (xmlChar *) "data_entry") == 0)
    {
      state = INSIDE_DATA_ENTRY;
      read_data_entry(name, attrs);
    }

    break;

  case INSIDE_DATA_ENTRY:

    if (xmlStrcasecmp(name, (xmlChar *) "array") == 0)
      read_array(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar *) "map") == 0 )
      read_map(name, attrs);
    else if (xmlStrcasecmp(name, (xmlChar *) "meshfunction") == 0 )
      read_mesh_function(name, attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMeshData::end_element (const xmlChar* name)
{
  switch (state)
  {
  case INSIDE_DATA:

    if (xmlStrcasecmp(name, (xmlChar *) "data") == 0)
    {
      state = DONE;
      release();
    }

    break;

  case INSIDE_DATA_ENTRY:

    if (xmlStrcasecmp(name, (xmlChar *) "data_entry") == 0)
    {
      state = INSIDE_DATA;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMeshData::write(const MeshData& data, std::ostream& outfile,
                        uint indentation_level)
{
  if (data.mesh_functions.size() > 0 || data.arrays.size() > 0)
  {
    XMLIndent indent(indentation_level);

    // Write mesh data header
    outfile << indent();
    outfile << "<data>" << std::endl;

    // Increment level for data_entries
    ++indent;

    // Write mesh functions
    typedef std::map<std::string,  boost::shared_ptr<MeshFunction<unsigned int> > >::const_iterator mf_iter;
    for (mf_iter it = data.mesh_functions.begin(); it != data.mesh_functions.end(); ++it)
    {
      // Write data entry header
      outfile << indent();
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write mesh function (omit mesh)
      ++indent;
      XMLMeshFunction::write(*(it->second), outfile, indent.level(), false);
      --indent;

      // Write data entry footer
      outfile << indent();
      outfile << "</data_entry>" << std::endl;
    }

    typedef std::map<std::string, std::vector<uint>*>::const_iterator arr_iter;
    for (arr_iter it = data.arrays.begin(); it != data.arrays.end(); ++it)
    {
      // Write data entry header
      outfile << indent();
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write array
      ++indent;
      XMLArray::write(*(it->second), 0, outfile, indent.level());
      --indent;

      // Write data entry footer
      outfile << indent();
      outfile << "</data_entry>" << std::endl;
    }

    typedef std::map<std::string, std::map<uint,uint>* >::const_iterator map_iter;
    for (map_iter it = data.mappings.begin(); it != data.mappings.end(); ++it)
    {
      // Write data entry header
      outfile << indent();
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write array
      ++indent;
      XMLMap::write(*(it->second), outfile, indent.level());
      --indent;

      // Write data entry footer
      outfile << indent();
      outfile << "</data_entry>" << std::endl;
    }

    typedef std::map<std::string, std::map<uint, std::vector<uint> >* >::const_iterator vec_map_iter;
    for (vec_map_iter it = data.vector_mappings.begin(); it != data.vector_mappings.end(); ++it)
    {
      // Write data entry header
      outfile << indent();
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write array
      ++indent;
      XMLMap::write(*(it->second), outfile, indent.level());
      --indent;

      // Write data entry footer
      outfile << indent();
      outfile << "</data_entry>" << std::endl;
    }

    // Done with entries, decrement level
    --indent;

    // Write mesh data footer
    outfile << indent();
    outfile << "</data>" << std::endl;
  }
}
//-----------------------------------------------------------------------------
void XMLMeshData::read_data_entry(const xmlChar* name, const xmlChar** attrs)
{
  entity_name = parse_string(name, attrs, "name");
}
//-----------------------------------------------------------------------------
void XMLMeshData::read_array(const xmlChar* name, const xmlChar** attrs)
{
  std::string array_type = parse_string(name, attrs, "type");
  uint size = parse_uint(name, attrs, "size");
  if (array_type.compare("int") == 0)
  {
    // FIXME: Add support for more types in MeshData?
    std::vector<int>* ux = new std::vector<int>();
    delete xml_array;
    xml_array = new XMLArray(*ux, parser, size);
    xml_array->handle();
  }
  else if (array_type.compare("uint") == 0)
  {
    std::vector<uint>* array = data.create_array(entity_name, size);
    delete xml_array;
    xml_array = new XMLArray(*array, parser, size);
    xml_array->handle();
  }
  else if (array_type.compare("double") == 0)
  {
    // FIXME: Add support for more types in MeshData?
    std::vector<double>* dx = new std::vector<double>();
    delete xml_array;
    xml_array = new XMLArray(*dx, parser, size);
    xml_array->handle();
  }
}
//-----------------------------------------------------------------------------
void XMLMeshData::read_map(const xmlChar* name, const xmlChar** attrs)
{
  std::string key_type = parse_string(name, attrs, "key_type");
  std::string value_type = parse_string(name, attrs, "value_type");

  if (! key_type.compare("uint") == 0)
    error("Key type in mapping must be uint.");
  if (value_type.compare("uint") == 0)
  {
    delete xml_map;
    std::map<uint, uint>* map = data.create_mapping(entity_name);
    xml_map = new XMLMap(*map,  parser);
    xml_map->handle();
  } else if (value_type.compare("array") == 0)
    warning("Add support for array maps, needed in parallel assembly.");
  else
    error("Unknown map type '%s'.", value_type.c_str());
}
//-----------------------------------------------------------------------------
void XMLMeshData::read_mesh_function(const xmlChar* name, const xmlChar** attrs)
{
  std::string mf_type = parse_string(name, attrs, "type");
  uint dim = parse_uint(name, attrs, "dim");
  uint size = parse_uint(name, attrs, "size");
  if (mf_type.compare("uint") != 0)
    error("Only MeshFunctions of type 'uint' supported as mesh data. Found '%s'.", mf_type.c_str());
  delete xml_mesh_function;
   boost::shared_ptr<MeshFunction<unsigned int> > mf = data.create_mesh_function(entity_name);
  xml_mesh_function = new XMLMeshFunction(*mf, parser, size, dim);
  xml_mesh_function->handle();
}
//-----------------------------------------------------------------------------
