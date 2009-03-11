// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-10
// Last changed: 2009-03-11

#ifndef __NEWXMLMESHFUNCTION_H
#define __NEWXMLMESHFUNCTION_H

#include <dolfin/mesh/MeshFunction.h>
#include "XMLHandler.h"

namespace dolfin
{

  class NewXMLMeshFunction : public XMLHandler
  {
  public:

    NewXMLMeshFunction(MeshFunction<int>& imf, NewXMLFile& parser);
    NewXMLMeshFunction(MeshFunction<int>& imf, NewXMLFile& parser, uint size, uint dim);

    NewXMLMeshFunction(MeshFunction<uint>& umf, NewXMLFile& parser);
    NewXMLMeshFunction(MeshFunction<uint>& umf, NewXMLFile& parser, uint size, uint dim);

    NewXMLMeshFunction(MeshFunction<double>& dmf, NewXMLFile& parser);
    NewXMLMeshFunction(MeshFunction<double>& dmf, NewXMLFile& parser, uint size, uint dim);
     
    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);
    
    /// Write to file
    static void write(const MeshFunction<int>& mf, std::ostream& outfile, uint indentation_level=0);
    static void write(const MeshFunction<uint>& mf, std::ostream& outfile, uint indentation_level=0);
    static void write(const MeshFunction<double>& mf, std::ostream& outfile, uint indentation_level=0);

  private:
    
    enum parser_state { OUTSIDE_MESHFUNCTION, INSIDE_MESHFUNCTION, DONE };
    enum mesh_function_type { INT, UINT, DOUBLE, UNSET };
    
    void start_mesh_function(const xmlChar *name, const xmlChar **attrs);
    void read_entity (const xmlChar *name, const xmlChar **attrs);
    
    MeshFunction<int>*  imf;
    MeshFunction<uint>* umf;
    MeshFunction<double>* dmf;
    parser_state state;
    mesh_function_type mf_type;

    uint size;
    uint dim;

  };
  
}

#endif
