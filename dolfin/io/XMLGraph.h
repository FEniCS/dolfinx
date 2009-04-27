// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-02-12
// Last changed: 2007-03-21

#ifndef __XML_GRAPH_H
#define __XML_GRAPH_H

#include <dolfin/graph/GraphEditor.h>
#include <dolfin/graph/Graph.h>
#include "XMLObject.h"

namespace dolfin
{

  class Graph;

  class XMLGraph : public XMLObject
  {
  public:

    XMLGraph(Graph& graph);
    ~XMLGraph();

    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);

    void open(std::string filename);
    bool close();

  private:

    enum ParserState { OUTSIDE, INSIDE_GRAPH, INSIDE_VERTICES,
                       INSIDE_EDGES, DONE };

    void read_graph       (const xmlChar* name, const xmlChar** attrs);
    void read_vertex      (const xmlChar* name, const xmlChar** attrs);
    void read_vertices    (const xmlChar* name, const xmlChar** attrs);
	void read_edge        (const xmlChar* name, const xmlChar** attrs);
	void read_edges       (const xmlChar* name, const xmlChar** attrs);

    void close_graph();

    Graph& _graph;
    ParserState state;
    GraphEditor editor;
    uint current_vertex;

  };

}

#endif
