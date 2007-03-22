"""Unit test for the graph library"""

__author__ = "Gustav Magnus Vikstrom (gustavv@ifi.uio.no)"
__date__ = "2007-02-12 -- 2007-03-21"
__copyright__ = "Copyright (C) 2007 Gustav Magnus Vikstrom"
__license__  = "GNU GPL Version 2"

import unittest
from dolfin import *

class Editor(unittest.TestCase):

""" Testing error conditions, currently does not work because editor exits on error
    def testGraphEditorVertexOrderError(self):
        \"""Create graph with vertices in wrong order (should fail)\"""
        graph = Graph()
        editor = GraphEditor()
        editor.open(graph)
        editor.initVertices(4)
        editor.initEdges(5)
        editor.addVertex(0, 2)
        try:
            editor.addVertex(3, 3)
            self.fail("Should not be allowed to add vertex 3 after vertex 0")
        except SystemExit:
            print "ok"

    def testGraphEditorUndefinedVertexError(self):
        \"""Create graph with edges out of undefined vertices (should fail)\"""
        graph = Graph()
        editor = GraphEditor()
        editor.open(graph)
        editor.initVertices(4)
        editor.initEdges(5)
        editor.addVertex(0, 2)
        try:
            editor.addEdge(3, 3)
            self.fail("Should not be allowed to add edge to undefined vertex")
        except SystemExit:
            print "ok"

    def testGraphEditorCloseError(self):
        \"""Create graph with undefined vertices/edges (should fail)\"""
        graph = Graph()
        editor = GraphEditor()
        editor.open(graph)
        editor.initVertices(4)
        editor.initEdges(5)
        editor.addVertex(0, 2)
        editor.addVertex(1, 2)
        editor.addVertex(2, 2)
        editor.addEdge(0, 2)
        editor.addEdge(1, 2)
        try:
            editor.close()
            self.fail("Should not be allowed to close with undefined vertices/edges")
        except SystemExit:
            print "ok"

"""

if __name__ == "__main__":
    unittest.main()
