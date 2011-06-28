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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2011.
//
// First added:  2009-03-03
// Last changed: 2011-03-31

#ifndef __SAXHANDLER_H
#define __SAXHANDLER_H

#include <fstream>
#include <map>
#include <string>
#include <stack>
#include <vector>

#include <libxml/parser.h>

#include <dolfin/common/MPI.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/plot/FunctionPlotData.h>
#include "GenericFile.h"
#include "XMLLocalMeshDataDistributed.h"
#include "XMLFunctionPlotData.h"

namespace dolfin
{

  class GenericVector;
  class Mesh;
  class Parameters;

  class SaxHandler
  {
  public:


  private:

    //xmlSAXHandler sax;

  };

  /*
  // Callback functions for the SAX interface
  void sax_start_document (void *ctx);
  void sax_end_document   (void *ctx);
  void sax_start_element  (void *ctx, const xmlChar *name, const xmlChar **attrs);
  void sax_end_element    (void *ctx, const xmlChar *name);

  void sax_warning     (void *ctx, const char *msg, ...);
  void sax_error       (void *ctx, const char *msg, ...);
  void sax_fatal_error (void *ctx, const char *msg, ...);

  // Callback functions for Relax-NG Schema
  void rng_parser_error(void *user_data, xmlErrorPtr error);
  void rng_valid_error (void *user_data, xmlErrorPtr error);
  */

}
#endif
