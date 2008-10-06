// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-10-02
// Last changed: 2006-10-23

#ifndef __XML_BLAS_FORM_DATA_H
#define __XML_BLAS_FORM_DATA_H

#include <dolfin/common/Array.h>
#include "XMLObject.h"

namespace dolfin
{

  class BLASFormData;
  
  class XMLBLASFormData : public XMLObject
  {
  public:

    XMLBLASFormData(BLASFormData& blas);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_FORM,
		       INSIDE_INTERIOR, INSIDE_BOUNDARY,
		       INSIDE_INTERIOR_TERM, INSIDE_BOUNDARY_TERM,
		       INSIDE_INTERIOR_GEOTENSOR, INSIDE_BOUNDARY_GEOTENSOR,
		       INSIDE_INTERIOR_REFTENSOR, INSIDE_BOUNDARY_REFTENSOR,
		       DONE };
    
    void readForm     (const xmlChar *name, const xmlChar **attrs);
    void readInterior (const xmlChar *name, const xmlChar **attrs);
    void readBoundary (const xmlChar *name, const xmlChar **attrs);
    void readTerm     (const xmlChar *name, const xmlChar **attrs);
    void readGeoTensor(const xmlChar *name, const xmlChar **attrs);
    void readRefTensor(const xmlChar *name, const xmlChar **attrs);
    void readEntry    (const xmlChar *name, const xmlChar **attrs);
    
    void initForm();

    BLASFormData& blas;

    Array<Array<double> > data_interior;
    Array<Array<double> > data_boundary;
    int mi, ni, mb, nb;

    ParserState state;
    
  };
  
}

#endif
