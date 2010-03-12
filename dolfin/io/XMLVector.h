// Copyright (C) 2003-2007 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-06
// Last changed: 2009-06-15

#ifndef __XML_VECTOR_H
#define __XML_VECTOR_H

#include <memory>
#include <boost/scoped_ptr.hpp>
#include "XMLArray.h"
#include "XMLHandler.h"

namespace dolfin
{

  class GenericVector;


  class XMLVector : public XMLHandler
  {
  public:

    XMLVector(GenericVector& vector, XMLFile& parser);
    ~XMLVector();

    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);

    static void write(const GenericVector& vector, std::ostream& outfile, uint indentation_level=0);

    // Read the vector begin tag
    void read_vector_tag(const xmlChar *name, const xmlChar **attrs);

  private:

    enum parser_state { OUTSIDE, INSIDE_VECTOR, DONE };

    void end_vector();
    void read_array_tag(const xmlChar *name, const xmlChar **attrs);

    GenericVector& x;
    parser_state state;

    std::vector<double> values;
    boost::scoped_ptr<XMLArray> xml_array;

  };

}

#endif
