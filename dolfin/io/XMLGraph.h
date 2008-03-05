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
    
    void startElement (const xmlChar* name, const xmlChar** attrs);
    void endElement   (const xmlChar* name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_GRAPH, INSIDE_VERTICES, 
                       INSIDE_EDGES, DONE };
    
    void readGraph       (const xmlChar* name, const xmlChar** attrs);
    void readVertex      (const xmlChar* name, const xmlChar** attrs);
    void readVertices    (const xmlChar* name, const xmlChar** attrs);
	void readEdge        (const xmlChar* name, const xmlChar** attrs);
	void readEdges       (const xmlChar* name, const xmlChar** attrs);
    
    void closeGraph();

    Graph& _graph;
    ParserState state;
    GraphEditor editor;
    uint currentVertex;
    
  };
  
}

#endif
