// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-02
// Last changed: 2005-10-03

#ifndef __XML_BLAS_FORM_DATA_H
#define __XML_BLAS_FORM_DATA_H

#include <dolfin/Array.h>
#include <dolfin/XMLObject.h>

namespace dolfin
{

  class BLASFormData;
  
  class XMLBLASFormData : public XMLObject
  {
  public:

    XMLBLASFormData(BLASFormData& blas);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
    void reading(std::string filename);
    void done();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_FORM,
		       INSIDE_INTERIOR, INSIDE_BOUNDARY,
		       INSIDE_INTERIOR_TERM, INSIDE_BOUNDARY_TERM,
		       INSIDE_INTERIOR_GEOTENSOR, INSIDE_BOUNDARY_GEOTENSOR,
		       INSIDE_INTERIOR_REFTENSOR, INSIDE_BOUNDARY_REFTENSOR,
		       DONE };
    
    void readForm        (const xmlChar *name, const xmlChar **attrs);
    void readInterior    (const xmlChar *name, const xmlChar **attrs);
    void readBoundary    (const xmlChar *name, const xmlChar **attrs);
    void readTerm        (const xmlChar *name, const xmlChar **attrs);
    void readGeoTensor   (const xmlChar *name, const xmlChar **attrs);
    void readRefTensor   (const xmlChar *name, const xmlChar **attrs);
    void readEntry       (const xmlChar *name, const xmlChar **attrs);
    
    void initForm();

    BLASFormData& blas;

    Array<Array<real> > data_interior;
    Array<Array<real> > data_boundary;
    int mi, ni, mb, nb;

    ParserState state;
    
  };
  
}

#endif
