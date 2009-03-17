// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-02
// Last changed: 2009-03-17

#include <dolfin/log/dolfin_log.h>
#include "XMLIndent.h"
#include "NewXMLFile.h"
#include "NewXMLMeshFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLMeshFunction::NewXMLMeshFunction(MeshFunction<int>& imf, NewXMLFile& parser)
  : XMLHandler(parser), imf(&imf), umf(0), dmf(0), state(OUTSIDE_MESHFUNCTION), mf_type(INT), size(0), dim(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewXMLMeshFunction::NewXMLMeshFunction(MeshFunction<uint>& umf, NewXMLFile& parser)
  : XMLHandler(parser), imf(0), umf(&umf), dmf(0), state(OUTSIDE_MESHFUNCTION), mf_type(UINT), size(0), dim(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewXMLMeshFunction::NewXMLMeshFunction(MeshFunction<double>& dmf, NewXMLFile& parser)
  : XMLHandler(parser), imf(0), umf(0), dmf(&dmf), state(OUTSIDE_MESHFUNCTION), mf_type(DOUBLE), size(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewXMLMeshFunction::NewXMLMeshFunction(MeshFunction<int>& imf, NewXMLFile& parser, uint size, uint dim)
  : XMLHandler(parser), imf(&imf), umf(0), dmf(0), state(INSIDE_MESHFUNCTION), mf_type(INT), size(size), dim(dim)
{
  // Initialize mesh function
  this->imf->init(dim);

  // Set all values to zero
  *(this->imf) = 0;
}
//-----------------------------------------------------------------------------
NewXMLMeshFunction::NewXMLMeshFunction(MeshFunction<uint>& umf, NewXMLFile& parser, uint size, uint dim)
  : XMLHandler(parser), imf(0), umf(&umf), dmf(0), state(INSIDE_MESHFUNCTION), mf_type(UINT), size(size), dim(dim)
{
  // Initialize mesh function
  this->umf->init(dim);

  // Set all values to zero
  *(this->umf) = 0;
}
//-----------------------------------------------------------------------------
NewXMLMeshFunction::NewXMLMeshFunction(MeshFunction<double>& dmf, NewXMLFile& parser, uint size, uint dim)
  : XMLHandler(parser), imf(0), umf(0), dmf(&dmf), state(INSIDE_MESHFUNCTION), mf_type(DOUBLE), size(size), dim(dim)
{
  // Initialize mesh function
  this->dmf->init(dim);

  // Set all values to zero
  *(this->dmf) = 0;
}

//-----------------------------------------------------------------------------
void NewXMLMeshFunction::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE_MESHFUNCTION:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "meshfunction") == 0 )
    {
      start_mesh_function(name, attrs);
      state = INSIDE_MESHFUNCTION;
    }
    
    break;
    
  case INSIDE_MESHFUNCTION:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "entity") == 0 )
      read_entity(name, attrs);
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLMeshFunction::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_MESHFUNCTION:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "meshfunction") == 0 )
    {
      state = DONE;
      release();
    }
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLMeshFunction::write(const MeshFunction<int>& mf, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent();
  outfile << "<meshfunction type=\"int\" dim=\"" << mf.dim() << "\" size=\"" << mf.size() << "\">" << std::endl;

  ++indent;
  for (uint i = 0; i < mf.size(); ++i)
  {
    outfile << indent();
    outfile << "<entity index=\"" << i << "\" value=\"" << mf.get(i) << "\"/>" << std::endl;
  }
  --indent;
  outfile << indent() << "</meshfunction>" << std::endl;
}
//-----------------------------------------------------------------------------
void NewXMLMeshFunction::write(const MeshFunction<uint>& mf, std::ostream& outfile, uint indentation_level)
{  
  XMLIndent indent(indentation_level);
  outfile << indent();
  outfile << "<meshfunction type=\"uint\" dim=\"" << mf.dim() << "\" size=\"" << mf.size() << "\">" << std::endl;

  ++indent;
  for (uint i = 0; i < mf.size(); ++i)
  {
    outfile << indent();
    outfile << "<entity index=\"" << i << "\" value=\"" << mf.get(i) << "\"/>" << std::endl;
  }
  --indent;
  outfile << indent() << "</meshfunction>" << std::endl;
}

//-----------------------------------------------------------------------------
void NewXMLMeshFunction::write(const MeshFunction<double>& mf, std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent();
  outfile << "<meshfunction type=\"double\" dim=\"" << mf.dim() << "\" size=\"" << mf.size() << "\">" << std::endl;

  ++indent;
  for (uint i = 0; i < mf.size(); ++i)
  {
    outfile << indent();
    outfile << "<entity index=\"" << i << "\" value=\"" << mf.get(i) << "\"/>" << std::endl;
  }
  --indent;
  outfile << indent() << "</meshfunction>" << std::endl;
}

//-----------------------------------------------------------------------------
void NewXMLMeshFunction::start_mesh_function(const xmlChar *name, const xmlChar **attrs)
{
  // Parse size of mesh function
  size = parse_uint(name, attrs, "size");

  // Parse type of mesh function 
  std::string _type = parse_string(name, attrs, "type");

  // Parse dimension of mesh function

  uint dim = parse_uint(name, attrs, "dim");
  
  // Initialize mesh function
  switch ( mf_type )
  {
    case INT:
      dolfin_assert(imf);
      if ( _type.compare("int") != 0 )
        error("MeshFunction file of type '%s', expected 'int'.", _type.c_str());
      imf->init(dim);
      
      break;

    case UINT:
      dolfin_assert(umf);
      if ( _type.compare("uint") != 0 )
        error("MeshFunction file of type '%s', expected 'uint'.", _type.c_str());
      umf->init(dim);

      break;

    case DOUBLE:
      dolfin_assert(dmf);
      if ( _type.compare("double") != 0 )
        error("MeshFunction file of type '%s', expected 'double'.", _type.c_str());
      dmf->init(dim);

      break;

    default:
      ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLMeshFunction::read_entity(const xmlChar *name, const xmlChar **attrs)
{
  // Parse index 
  uint index = parse_uint(name, attrs, "index");
  
  // Check values
  if (index >= size)
    error("Illegal XML data for MeshFunction: row index %d out of range (0 - %d)",
          index, size - 1);
  
  // Parse value and insert in array
  switch ( mf_type )
  {
    case INT:
      dolfin_assert(imf);
      imf->set(index, parse_int(name, attrs, "value"));

      break;

     case UINT:
      dolfin_assert(umf);
      umf->set(index, parse_uint(name, attrs, "value"));

      break;

     case DOUBLE:
      dolfin_assert(dmf);
      dmf->set(index, parse_float(name, attrs, "value"));

      break;
          
     default:
      ;
  }
}
//-----------------------------------------------------------------------------
