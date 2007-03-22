SUBDIRS                   = dolfin
nobase_include_HEADERS    = dolfin/GraphEditor.h \
                            dolfin/Graph.h \
                            dolfin/UndirectedClique.h \
                            dolfin/DirectedClique.h
INCLUDES                  = @KERNEL_CFLAGS@
lib_LTLIBRARIES           = libdolfin-graph.la
libdolfin_graph_la_SOURCES = GraphEditor.cpp \
                            Graph.cpp \
                            UndirectedClique.cpp \
                            DirectedClique.cpp
libdolfin_graph_la_LDFLAGS = -static
