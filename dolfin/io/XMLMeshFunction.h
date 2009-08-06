// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-10
// Last changed: 2009-03-11

#ifndef __XMLMESHFUNCTION_H
#define __XMLMESHFUNCTION_H

#include <dolfin/mesh/MeshFunction.h>
#include "XMLHandler.h"

namespace dolfin
{

  class XMLSkipper;

  class XMLMeshFunction : public XMLHandler
  {
  public:

    XMLMeshFunction(MeshFunction<int>& imf, XMLFile& parser);
    XMLMeshFunction(MeshFunction<int>& imf, XMLFile& parser, uint size, uint dim);

    XMLMeshFunction(MeshFunction<uint>& umf, XMLFile& parser);
    XMLMeshFunction(MeshFunction<uint>& umf, XMLFile& parser, uint size, uint dim);

    XMLMeshFunction(MeshFunction<double>& dmf, XMLFile& parser);
    XMLMeshFunction(MeshFunction<double>& dmf, XMLFile& parser, uint size, uint dim);

    /// Destructor
    ~XMLMeshFunction();

    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);

    /// Write to file
    static void write(const MeshFunction<int>& mf, std::ostream& outfile, uint indentation_level=0, bool write_mesh=true);
    static void write(const MeshFunction<uint>& mf, std::ostream& outfile, uint indentation_level=0, bool write_mesh=true);
    static void write(const MeshFunction<double>& mf, std::ostream& outfile, uint indentation_level=0, bool write_mesh=true);

  private:

    enum parser_state { OUTSIDE_MESHFUNCTION, INSIDE_MESHFUNCTION, DONE };
    enum mesh_function_type { INT, UINT, DOUBLE, UNSET };

    void start_mesh_function(const xmlChar *name, const xmlChar **attrs);
    void read_entity (const xmlChar *name, const xmlChar **attrs);

    MeshFunction<int>*  imf;
    MeshFunction<uint>* umf;
    MeshFunction<double>* dmf;
    XMLSkipper* xml_skipper;

    parser_state state;
    mesh_function_type mf_type;

    uint size;
    uint dim;

  };

}

#endif
