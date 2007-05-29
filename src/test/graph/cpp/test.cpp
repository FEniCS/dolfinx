// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1. 
//
// First added:  2007-05-29
// Last changed: 2007-05-29
//
// Unit tests for the graph library 

#include <dolfin.h>
#include <dolfin/unittest.h>

using namespace dolfin;

class Editor : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Editor);
  CPPUNIT_TEST(testUndirected);
  CPPUNIT_TEST(testDirected);
  CPPUNIT_TEST_SUITE_END();

public: 

  void testUndirected()
  {
    // Create undirected graph with edges added out of order (should pass)
    Graph graph;
    GraphEditor editor;
    editor.open(graph, "undirected");
    editor.initVertices(4);
    editor.addVertex(0, 2);
    editor.addVertex(1, 3);
    editor.addVertex(2, 2);
    editor.addVertex(3, 3);
    editor.initEdges(5);
    editor.addEdge(0, 1);
    editor.addEdge(1, 2);
    editor.addEdge(2, 3);
    editor.addEdge(1, 3);
    editor.addEdge(0, 3);
    editor.close();

    CPPUNIT_ASSERT(graph.numVertices() == 4);
    CPPUNIT_ASSERT(graph.numEdges() == 5);
    CPPUNIT_ASSERT(graph.numArches() == 10);
  }
   
  void testDirected()
  {
    // Create directed graph with edges added out of order (should pass)
    Graph graph;
    GraphEditor editor;
    editor.open(graph, "directed");
    editor.initVertices(4);
    editor.addVertex(0, 2);
    editor.addVertex(1, 1);
    editor.addVertex(2, 1);
    editor.addVertex(3, 1);
    editor.initEdges(5);
    editor.addEdge(0, 1);
    editor.addEdge(1, 2);
    editor.addEdge(2, 3);
    editor.addEdge(3, 1);
    editor.addEdge(0, 3);
    editor.close();

    CPPUNIT_ASSERT(graph.numVertices() == 4);
    CPPUNIT_ASSERT(graph.numEdges() == 5);
    CPPUNIT_ASSERT(graph.numArches() == 5);
  }
};

class InputOutput : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(InputOutput);
  CPPUNIT_TEST(testUndirectedGraphXML);
  CPPUNIT_TEST(testDirectedGraphXML);
  CPPUNIT_TEST(testMetisGraphConvertion);
  CPPUNIT_TEST(testScotchGraphConvertion);
  CPPUNIT_TEST_SUITE_END();

public:
  void testUndirectedGraphXML()
  {
    // Write and read undirected graph to/from file

    UndirectedClique graph_out(4);

    // Change to temp files
    File file("undirectedgraph.xml");
    file << graph_out;

    Graph graph_in;
    file >> graph_in;

    CPPUNIT_ASSERT(graph_out.numVertices() == graph_in.numVertices());
    CPPUNIT_ASSERT(graph_out.numEdges() == graph_in.numEdges());
  }

  void testDirectedGraphXML()
  {
    // Write and read directed graph to/from file

    DirectedClique graph_out(4);

    // Change to temp files
    File file("directedgraph.xml");
    file << graph_out;

    Graph graph_in;
    file >> graph_in;

    CPPUNIT_ASSERT(graph_out.numVertices() == graph_in.numVertices());
    CPPUNIT_ASSERT(graph_out.numEdges() == graph_in.numEdges());
  }

  void testMetisGraphConvertion()
  {
    // Create metis graph file, convert to dolfin xml and read xml file

  }

  void testScotchGraphConvertion()
  {
    // Create scotch graph file, convert to dolfin xml and read xml file

  }
};

class Partitioning : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(Partitioning);
  CPPUNIT_TEST(testPartition);
  CPPUNIT_TEST_SUITE_END();

public: 

  void testPartition()
  {
    // Create a graph and partition it

    /*  Example graph:
	            0 -- 1 -- 2
	                 |  /
	                 | /
		         4 -- 3 -- 6
               |    |    |
               |    |    |
               7    5 -- 8
    */
    dolfin::uint nn = 9;
    dolfin::uint num_part = 2;
    Graph graph;
    GraphEditor editor;
    editor.open(graph, "undirected");
    editor.initVertices(nn);
    editor.addVertex(0, 1);
    editor.addVertex(1, 3);
    editor.addVertex(2, 2);
    editor.addVertex(3, 5);
    editor.addVertex(4, 2);
    editor.addVertex(5, 2);
    editor.addVertex(6, 2);
    editor.addVertex(7, 1);
    editor.addVertex(8, 2);
    editor.initEdges(10);
    editor.addEdge(0, 1);
    editor.addEdge(1, 2);
    editor.addEdge(1, 3);
    editor.addEdge(2, 3);
    editor.addEdge(3, 4);
    editor.addEdge(3, 5);
    editor.addEdge(3, 6);
    editor.addEdge(4, 7);
    editor.addEdge(5, 8);
    editor.addEdge(6, 8);
    editor.close();

    dolfin::uint* parts = new dolfin::uint[nn];

    GraphPartition::partition(graph, num_part, parts);
    GraphPartition::eval(graph, num_part, parts);
    GraphPartition::disp(graph, num_part, parts);
    GraphPartition::check(graph, num_part, parts);
    dolfin::uint edgecut = GraphPartition::edgecut(graph, num_part, parts);

    // Simple graph partitioning should give edge-cut: 2
    CPPUNIT_ASSERT(edgecut == 2);
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(Editor);
CPPUNIT_TEST_SUITE_REGISTRATION(InputOutput);
CPPUNIT_TEST_SUITE_REGISTRATION(Partitioning);

int main()
{
  DOLFIN_TEST;
}
