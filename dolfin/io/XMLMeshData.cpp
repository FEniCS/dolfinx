// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added: 2009-03-09
// Last changed: 2009-03-09


#include "NewXMLMeshFunction.h"
#include "XMLArray.h"
#include "XMLMap.h"
#include "XMLMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMeshData::XMLMeshData(MeshData& data, NewXMLFile& parser, bool inside)
  : XMLHandler(parser), data(data), state(OUTSIDE), type(UNSET), entity_name(""),
    im(0), um(0), dm(0), iam(0), uam(0), dam(0), imf(0), umf(0), dmf(0)
{
  if ( inside )
    state = INSIDE_DATA;
  // Do nothing
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
  delete imf;
  delete umf;
  delete dmf;
}
//-----------------------------------------------------------------------------
void XMLMeshData::start_element (const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
    case OUTSIDE:
      if ( xmlStrcasecmp(name, (xmlChar *) "data") == 0 )
        state = INSIDE_DATA;

      break;

    case INSIDE_DATA:
      if ( xmlStrcasecmp(name, (xmlChar *) "data_entry") == 0 )
      {
        state = INSIDE_DATA_ENTRY;
        read_data_entry(name, attrs);
      }

      break;

    case INSIDE_DATA_ENTRY:
      if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
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
  switch ( state )
  {
    case INSIDE_DATA:
      if ( xmlStrcasecmp(name, (xmlChar *) "data") == 0 )
      {
        state = DONE;
        release();
      }

      break;

    case INSIDE_DATA_ENTRY:
      if ( xmlStrcasecmp(name, (xmlChar *) "data_entry") == 0 )
      {
        state = INSIDE_DATA;
      }

      break;

    default:
      ;
  }
}
//-----------------------------------------------------------------------------
void XMLMeshData::write(const MeshData& data, std::ofstream& outfile, uint indentation_level)
{
  if ( data.mesh_functions.size() > 0 || data.arrays.size() > 0)
  {
    uint curr_indent = indentation_level;

    // Write mesh data header
    outfile << std::setw(curr_indent) << "";
    outfile << "<data>" << std::endl;


    // Write mesh functions
    typedef std::map<std::string, MeshFunction<uint>*>::const_iterator mf_iter;
    for (mf_iter it = data.mesh_functions.begin(); it != data.mesh_functions.end(); ++it)
    {
      // Write data entry header
      curr_indent = indentation_level + 2;
      outfile << std::setw(curr_indent) << "";
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write array
      NewXMLMeshFunction::write(*(it->second), outfile, indentation_level + 4);

      // Write data entry footer 
      curr_indent = indentation_level + 2;
      outfile << std::setw(curr_indent) << "";
      outfile << "</data_entry>" << std::endl;
    }

    typedef std::map<std::string, std::vector<uint>*>::const_iterator arr_iter;
    for (arr_iter it = data.arrays.begin(); it != data.arrays.end(); ++it)
    {
      // Write data entry header
      curr_indent = indentation_level + 2;
      outfile << std::setw(curr_indent) << "";
      outfile << "<data_entry name=\"" << it->first << "\">" << std::endl;

      // Write array
      XMLArray::write(*(it->second), outfile, indentation_level + 4);

      // Write data entry footer 
      curr_indent = indentation_level + 2;
      outfile << std::setw(curr_indent) << "";
      outfile << "</data_entry>" << std::endl;
    }

    // Write mesh data footer
    curr_indent = indentation_level;
    outfile << std::setw(curr_indent) << "";
    outfile << "<data>" << std::endl;

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
  if ( array_type.compare("int") == 0 )
  {
    // FIXME: Add support for more types in MeshData?
    std::vector<int>* ux = new std::vector<int>();
    delete xml_array;
    xml_array = new XMLArray(*ux, parser, size);
    xml_array->handle();
  }
  else if ( array_type.compare("uint") == 0 )
  {
    data.create_array(entity_name, size);
    delete xml_array;
    xml_array = new XMLArray(*(data.array(entity_name)), parser, size);
    xml_array->handle();
  }
  else if ( array_type.compare("double") == 0 )
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

  if ( ! key_type.compare("uint") == 0 )
    error("Key type in mapping must be uint.");
  if ( value_type.compare("uint") == 0 )
  {
    delete xml_map;
    data.create_mapping(entity_name);
    xml_map = new XMLMap(*(data.mapping(entity_name)), parser);
    xml_map->handle();
  } else if ( value_type.compare("array") == 0 )
    warning("Add support for array maps, needed in parallel assembly.");
  else
    error("Unknown map type '%s'.", value_type.c_str());
}
//-----------------------------------------------------------------------------
void XMLMeshData::read_mesh_function(const xmlChar* name, const xmlChar** attrs)
{
}
//-----------------------------------------------------------------------------
