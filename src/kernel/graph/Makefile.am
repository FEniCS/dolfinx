SUBDIRS                   = dolfin
nobase_include_HEADERS    = dolfin/dolfin_graph.h \
                            dolfin/GraphEditor.h \
                            dolfin/GraphPartition.h \
                            dolfin/Graph.h \
                            dolfin/UndirectedClique.h \
                            dolfin/DirectedClique.h
INCLUDES                  = @KERNEL_CFLAGS@
lib_LTLIBRARIES           = libdolfin-graph.la
libdolfin_graph_la_SOURCES = GraphEditor.cpp \
                            GraphPartition.cpp \
                            Graph.cpp \
                            UndirectedClique.cpp \
                            DirectedClique.cpp
libdolfin_graph_la_LDFLAGS = -static
