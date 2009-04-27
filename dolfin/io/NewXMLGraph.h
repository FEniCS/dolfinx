// Copyright (C) 2009 Ola Skavhaug and Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// This file is a port of Magnus Vikstrom's previous implementation
//
// First added:  2009-03-11
// Last changed: 2000-03-11

#ifndef __NEWXMLGRAPH_H
#define __NEWXMLGRAPH_H

#include <dolfin/common/types.h>
#include <dolfin/graph/GraphEditor.h>
#include "XMLHandler.h"

namespace dolfin
{
  class Graph;
  class NewXMLFile;

  class NewXMLGraph : public XMLHandler
  {
  public:

    NewXMLGraph(Graph& graph, NewXMLFile& parser);
    ~NewXMLGraph();

    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);

    static void write(const Graph& graph, std::ostream& outfile, uint indentation_level=0);

  private:

    enum parser_state { OUTSIDE, INSIDE_GRAPH, INSIDE_VERTICES,
                       INSIDE_EDGES, DONE };

    void read_graph       (const xmlChar* name, const xmlChar** attrs);
    void read_vertex      (const xmlChar* name, const xmlChar** attrs);
    void read_vertices    (const xmlChar* name, const xmlChar** attrs);
    void read_edge        (const xmlChar* name, const xmlChar** attrs);
    void read_edges       (const xmlChar* name, const xmlChar** attrs);

    void close_graph();

    Graph& graph;
    parser_state state;
    GraphEditor editor;

  };

}

#endif
