// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-02
// Last changed: 2005-10-02

#include <dolfin/dolfin_log.h>
#include <dolfin/XMLForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLForm::XMLForm(Form& form) : XMLObject(), form(form)
{
  state = OUTSIDE;
}
//-----------------------------------------------------------------------------
void XMLForm::startElement(const xmlChar *name, const xmlChar **attrs)
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
    
    if ( xmlStrcasecmp(name,(xmlChar *) "referencetensor") == 0 )
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
void XMLForm::endElement(const xmlChar *name)
{
  //dolfin_debug1("Found end of element \"%s\"", (const char *) name);

  switch ( state )
  {
  case INSIDE_FORM:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "form") == 0 )
    {
      initForm();
      ok = true;
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
void XMLForm::reading(std::string filename)
{
  cout << "Reading form data from file \"" << filename << "\"." << endl;
}
//-----------------------------------------------------------------------------
void XMLForm::done()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLForm::readForm(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading form..." << endl;
}
//-----------------------------------------------------------------------------
void XMLForm::readInterior(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading interior..." << endl;
}
//-----------------------------------------------------------------------------
void XMLForm::readBoundary(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading boundary..." << endl;
}
//-----------------------------------------------------------------------------
void XMLForm::readTerm(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading term..." << endl;
}
//-----------------------------------------------------------------------------
void XMLForm::readRefTensor(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading reference tensor..." << endl;
}
//-----------------------------------------------------------------------------
void XMLForm::readEntry(const xmlChar *name, const xmlChar **attrs)
{
  cout << "Reading entry..." << endl;
}
//-----------------------------------------------------------------------------
void XMLForm::initForm()
{
  // Clear form data
}
//-----------------------------------------------------------------------------
