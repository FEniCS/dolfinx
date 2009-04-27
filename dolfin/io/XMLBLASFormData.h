// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-10-02
// Last changed: 2006-10-23

#ifndef __XML_BLAS_FORM_DATA_H
#define __XML_BLAS_FORM_DATA_H

#include "XMLObject.h"

namespace dolfin
{

  class BLASFormData;
  
  class XMLBLASFormData : public XMLObject
  {
  public:

    XMLBLASFormData(BLASFormData& blas);
    
    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_FORM,
		       INSIDE_INTERIOR, INSIDE_BOUNDARY,
		       INSIDE_INTERIOR_TERM, INSIDE_BOUNDARY_TERM,
		       INSIDE_INTERIOR_GEOTENSOR, INSIDE_BOUNDARY_GEOTENSOR,
		       INSIDE_INTERIOR_REFTENSOR, INSIDE_BOUNDARY_REFTENSOR,
		       DONE };
    
    void read_form     (const xmlChar *name, const xmlChar **attrs);
    void read_interior (const xmlChar *name, const xmlChar **attrs);
    void read_boundary (const xmlChar *name, const xmlChar **attrs);
    void read_term     (const xmlChar *name, const xmlChar **attrs);
    void readGeoTensor(const xmlChar *name, const xmlChar **attrs);
    void readRefTensor(const xmlChar *name, const xmlChar **attrs);
    void read_entry    (const xmlChar *name, const xmlChar **attrs);
    
    void init_form();

    BLASFormData& blas;

    std::vector<std::vector<double> > data_interior;
    std::vector<std::vector<double> > data_boundary;
    int mi, ni, mb, nb;

    ParserState state;
    
  };
  
}

#endif
