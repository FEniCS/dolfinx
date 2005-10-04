// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-02
// Last changed: 2005-10-03

#include <dolfin/dolfin_log.h>
#include <dolfin/BLASFormData.h>
#include <dolfin/XMLBLASFormData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLBLASFormData::XMLBLASFormData(BLASFormData& blas) :
  XMLObject(), blas(blas), mi(0), ni(0), mb(0), nb(0), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::startElement(const xmlChar *name, const xmlChar **attrs)
{
  //dolfin_debug1("Found start of element \"%s\"", (const char *) name);

  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "form") == 0 )
    {
      readForm(name, attrs);
      state = INSIDE_FORM;
    }
    
    break;

  case INSIDE_FORM:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "interior") == 0 )
    {
      readInterior(name, attrs);
      state = INSIDE_INTERIOR;
    }
    else if ( xmlStrcasecmp(name,(xmlChar *) "boundary") == 0 )
    {
      readBoundary(name, attrs);
      state = INSIDE_BOUNDARY;
    }
    
    break;
    
  case INSIDE_INTERIOR:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "term") == 0 )
    {
      readTerm(name,attrs);
      state = INSIDE_INTERIOR_TERM;
    }

    break;

  case INSIDE_BOUNDARY:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "term") == 0 )
    {
      readTerm(name,attrs);
      state = INSIDE_BOUNDARY_TERM;
    }
    
    break;
    
  case INSIDE_INTERIOR_TERM:

    if ( xmlStrcasecmp(name,(xmlChar *) "geometrytensor") == 0 )
    {
      readGeoTensor(name, attrs);
      state = INSIDE_INTERIOR_GEOTENSOR;
    }
    else if ( xmlStrcasecmp(name,(xmlChar *) "referencetensor") == 0 )
    {
      readRefTensor(name, attrs);
      state = INSIDE_INTERIOR_REFTENSOR;
    }

    break;

  case INSIDE_BOUNDARY_TERM:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "referencetensor") == 0 )
    {
      readRefTensor(name, attrs);
      state = INSIDE_BOUNDARY_REFTENSOR;
    }

    break;

  case INSIDE_INTERIOR_GEOTENSOR:
    break;

  case INSIDE_BOUNDARY_GEOTENSOR:
    break;

  case INSIDE_INTERIOR_REFTENSOR:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "entry") == 0 )
    {
      readEntry(name, attrs);
    }

    break;

  case INSIDE_BOUNDARY_REFTENSOR:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "entry") == 0 )
    {
      readEntry(name, attrs);
    }

    break;
   
  default:
    ;
  }
  
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::endElement(const xmlChar *name)
{
  //dolfin_debug1("Found end of element \"%s\"", (const char *) name);

  switch ( state )
  {
  case INSIDE_FORM:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "form") == 0 )
    {
      initForm();
      ok = true;
      data_interior.clear();
      data_interior.clear();
      state = DONE;
    }
    
    break;
    
  case INSIDE_INTERIOR:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "interior") == 0 )
      state = INSIDE_FORM;
    
    break;

  case INSIDE_BOUNDARY:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "boundary") == 0 )
      state = INSIDE_FORM;
    
    break;

  case INSIDE_INTERIOR_TERM:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "term") == 0 )
      state = INSIDE_INTERIOR;
    
    break;

  case INSIDE_BOUNDARY_TERM:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "term") == 0 )
      state = INSIDE_BOUNDARY;
    
    break;

  case INSIDE_INTERIOR_GEOTENSOR:
	 
    if ( xmlStrcasecmp(name,(xmlChar *) "geometrytensor") == 0 )
      state = INSIDE_INTERIOR_TERM;
    
    break;

  case INSIDE_BOUNDARY_GEOTENSOR:
	 
    if ( xmlStrcasecmp(name,(xmlChar *) "geometrytensor") == 0 )
      state = INSIDE_BOUNDARY_TERM;
    
    break;
    
  case INSIDE_INTERIOR_REFTENSOR:
	 
    if ( xmlStrcasecmp(name,(xmlChar *) "referencetensor") == 0 )
      state = INSIDE_INTERIOR_TERM;
    
    break;

  case INSIDE_BOUNDARY_REFTENSOR:
	 
    if ( xmlStrcasecmp(name,(xmlChar *) "referencetensor") == 0 )
      state = INSIDE_BOUNDARY_TERM;
    
    break;
    
  default:
    ;
  }
  
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::reading(std::string filename)
{
  cout << "Reading form data from file \"" << filename << "\"." << endl;
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::done()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::readForm(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading form..." << endl;

  // Reset data
  data_interior.clear();
  data_boundary.clear();
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::readInterior(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading interior..." << endl;
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::readBoundary(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading boundary..." << endl;
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::readTerm(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading term..." << endl;

  Array<real> tensor;

  // Add new term and read number of rows
  switch ( state )
  {
  case INSIDE_INTERIOR:
    data_interior.push_back(tensor);
    parseIntegerRequired(name, attrs, "size", mi);
    break;
  case INSIDE_BOUNDARY:
    data_boundary.push_back(tensor);
    parseIntegerRequired(name, attrs, "size", mb);
    break;
  default:
    cout << state << endl;
    dolfin_error("Inconsistent state while reading XML form data.");
  }

}
//-----------------------------------------------------------------------------
void XMLBLASFormData::readGeoTensor(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading geometry tensor..." << endl;

  // Read number of columns
  switch ( state )
  {
  case INSIDE_INTERIOR_TERM:
    parseIntegerRequired(name, attrs, "size", ni);
    break;
  case INSIDE_BOUNDARY_TERM:
    parseIntegerRequired(name, attrs, "size", nb);
    break;
  default:
    dolfin_error("Inconsistent state while reading XML form data.");
  }
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::readRefTensor(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading reference tensor..." << endl;
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::readEntry(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading entry..." << endl;

  // Read value
  real value = 0.0;
  parseRealRequired(name, attrs, "value", value);

  // Append value to tensor
  switch ( state )
  {
  case INSIDE_INTERIOR_REFTENSOR:
    data_interior.back().push_back(value);
    break;
  case INSIDE_BOUNDARY_REFTENSOR:
    data_boundary.back().push_back(value);
    break;
  default:
    dolfin_error("Inconsistent state while reading XML data.");
  }
}
//-----------------------------------------------------------------------------
void XMLBLASFormData::initForm()
{
  // Give data to form
  blas.init(mi, ni, data_interior, mb, nb, data_boundary);
}
//-----------------------------------------------------------------------------
