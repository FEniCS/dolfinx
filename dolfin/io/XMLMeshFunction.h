// Copyright (C) 2009 Ola Skavhaug
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-03-10
// Last changed: 2009-09-08

#ifndef __XMLMESHFUNCTION_H
#define __XMLMESHFUNCTION_H

#include <iostream>
#include <map>
#include "dolfin/common/types.h"
#include "XMLHandler.h"

namespace dolfin
{

  class Mesh;
  template<class T> class MeshFunction;
  class XMLSkipper;

  class XMLMeshFunction : public XMLHandler
  {
  public:

    XMLMeshFunction(MeshFunction<int>& imf, XMLFile& parser);
    XMLMeshFunction(MeshFunction<int>& imf, XMLFile& parser, uint size, uint dim);

    XMLMeshFunction(MeshFunction<unsigned int>& umf, XMLFile& parser);
    XMLMeshFunction(MeshFunction<unsigned int>& umf, XMLFile& parser, uint size, uint dim);

    XMLMeshFunction(MeshFunction<double>& dmf, XMLFile& parser);
    XMLMeshFunction(MeshFunction<double>& dmf, XMLFile& parser, uint size, uint dim);

    /// Destructor
    ~XMLMeshFunction();

    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);

    /// Write to file
    static void write(const MeshFunction<int>& mf, std::ostream& outfile, uint indentation_level=0, bool write_mesh=true);
    static void write(const MeshFunction<unsigned int>& mf, std::ostream& outfile, uint indentation_level=0, bool write_mesh=true);
    static void write(const MeshFunction<double>& mf, std::ostream& outfile, uint indentation_level=0, bool write_mesh=true);

  private:

    enum parser_state { OUTSIDE_MESHFUNCTION, INSIDE_MESHFUNCTION, DONE };
    enum mesh_function_type { INT, UINT, DOUBLE, UNSET };

    void start_mesh_function(const xmlChar *name, const xmlChar **attrs);
    void read_entity (const xmlChar *name, const xmlChar **attrs);

    void build_mapping(uint entity_dimension);

    MeshFunction<int>*  imf;
    MeshFunction<uint>* umf;
    MeshFunction<double>* dmf;
    XMLSkipper* xml_skipper;
    const Mesh& mesh;
    std::map<uint, uint> glob2loc; // In parallel, use this global to local mapping to assign values


    parser_state state;
    mesh_function_type mf_type;

    uint size;
    uint dim;

  };

}

#endif
