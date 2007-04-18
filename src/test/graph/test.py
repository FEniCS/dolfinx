"""Unit test for the graph library"""

__author__ = "Gustav Magnus Vikstrom (gustavv@ifi.uio.no)"
__date__ = "2007-02-12 -- 2007-03-21"
__copyright__ = "Copyright (C) 2007 Gustav Magnus Vikstrom"
__license__  = "GNU GPL Version 2"

import unittest
import os
import tempfile
import numpy
from dolfin import *

class Editor(unittest.TestCase):

    def testGraphEditorUndirected(self):
        """Create undirected graph with edges added out of order (should pass)"""
        graph = Graph()
        editor = GraphEditor()
        editor.open(graph, "undirected")
        editor.initVertices(4)
        editor.addVertex(0, 2)
        editor.addVertex(1, 3)
        editor.addVertex(2, 2)
        editor.addVertex(3, 3)
        editor.initEdges(5)
        editor.addEdge(0, 1)
        editor.addEdge(1, 2)
        editor.addEdge(2, 3)
        editor.addEdge(1, 3)
        editor.addEdge(0, 3)
        editor.close()

        self.assertEqual(graph.numVertices(), 4)
        self.assertEqual(graph.numEdges(), 5)
        self.assertEqual(graph.numArches(), 10)

    def testGraphEditorDirected(self):
        """Create directed graph with edges added out of order (should pass)"""
        graph = Graph()
        editor = GraphEditor()
        editor.open(graph, "directed")
        editor.initVertices(4)
        editor.addVertex(0, 2)
        editor.addVertex(1, 1)
        editor.addVertex(2, 1)
        editor.addVertex(3, 1)
        editor.initEdges(5)
        editor.addEdge(0, 1)
        editor.addEdge(1, 2)
        editor.addEdge(2, 3)
        editor.addEdge(3, 1)
        editor.addEdge(0, 3)
        editor.close()

        self.assertEqual(graph.numVertices(), 4)
        self.assertEqual(graph.numEdges(), 5)
        self.assertEqual(graph.numArches(), 5)

class InputOutput(unittest.TestCase):

    def testUndirectedGraphXML(self):
        """Write and read undirected graph to/from file"""
        graph = UndirectedClique(4)

        # Create temp file from graph, read file and delete tempfile
        fd, tmp_name = tempfile.mkstemp(suffix=".xml")
        file = File(tmp_name)
        file << graph
        graph2 = Graph()
        file >> graph2
        os.remove(tmp_name)

        graph2.disp()
        self.assertEqual(graph.numVertices(), graph2.numVertices())
        self.assertEqual(graph.numEdges(), graph2.numEdges())

    def testDirectedGraphXML(self):
        """Write and read directed graph to/from file"""
        graph = DirectedClique(4)

        # Create temp file from graph, read file and delete tempfile
        fd, tmp_name = tempfile.mkstemp(suffix=".xml")
        file = File(tmp_name)
        file << graph
        graph2 = Graph()
        file >> graph2
        os.remove(tmp_name)

        graph2.disp()
        self.assertEqual(graph.numVertices(), graph2.numVertices())
        self.assertEqual(graph.numEdges(), graph2.numEdges())

    def testMetisGraphConvertion(self):
        """Create metis graph file, convert to dolfin xml and read xml file"""

        fd, tmp_name = tempfile.mkstemp(suffix=".gra")
        mfile = open(tmp_name, "w")
        mfile.write("3 3\n")
        mfile.write(" 1 2\n")
        mfile.write(" 0 2\n")
        mfile.write(" 0 1\n")
        
        mfile.close()

        # Create temp dolfin xml file from metis graph
        fd, tmp_name = tempfile.mkstemp(suffix=".xml")
        os.system('dolfin-convert %s %s' % (mfile.name, tmp_name))

        # Read dolfin xml file
        graph = Graph()
        file = File(tmp_name)
        file >> graph

        # Delete tempfiles
        os.remove(tmp_name)
        os.remove(mfile.name)
        
        self.assertEqual(graph.numVertices(), 3)
        self.assertEqual(graph.numEdges(), 3)

    def testScotchGraphConvertion(self):
        """Create scotch graph file, convert to dolfin xml and read xml file"""
        fd, tmp_name = tempfile.mkstemp(suffix=".grf")
        sfile = open(tmp_name, "w")
        sfile.write("0\n")
        sfile.write("3 6\n")
        sfile.write("1 000\n")
        sfile.write("2 1 2\n")
        sfile.write("2 0 2\n")
        sfile.write("2 0 1\n")

        sfile.close()

        # Create temp dolfin xml file from scotch graph
        fd, tmp_name = tempfile.mkstemp(suffix=".xml")
        os.system('dolfin-convert %s %s' % (sfile.name, tmp_name))

        # Read dolfin xml file
        graph = Graph()
        file = File(tmp_name)
        file >> graph
        
        # Delete tempfiles
        os.remove(tmp_name)
        os.remove(sfile.name)

        self.assertEqual(graph.numVertices(), 3)
        self.assertEqual(graph.numEdges(), 3)

class Partitioning(unittest.TestCase):

    def testPartition(self):
        """Create a graph and partition it"""

        graph = UndirectedClique(100)
        #parts = numpy.array('I')
        #parts_ptr = parts.c_ptr()
        #parts = GraphPartition.create(100)
        parts = numpy.array('I')

        GraphPartition.partition(graph, 10, parts)

if __name__ == "__main__":
    unittest.main()
