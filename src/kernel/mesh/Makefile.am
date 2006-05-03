SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_mesh.h \
                           dolfin/Boundary.h \
                           dolfin/BoundaryData.h \
                           dolfin/BoundaryFacetIterator.h \
                           dolfin/BoundaryInit.h \
                           dolfin/Cell.h \
                           dolfin/CellIterator.h \
                           dolfin/CellRefData.h \
                           dolfin/Edge.h \
                           dolfin/EdgeIterator.h \
                           dolfin/EdgeRefData.h \
                           dolfin/Face.h \
                           dolfin/FaceIterator.h \
                           dolfin/GenericCell.h \
                           dolfin/Mesh.h \
                           dolfin/MeshData.h \
                           dolfin/MeshHierarchy.h \
                           dolfin/MeshInit.h \
                           dolfin/MeshIterator.h \
                           dolfin/MeshRefinement.h \
                           dolfin/Point.h \
                           dolfin/TetMeshRefinement.h \
                           dolfin/Tetrahedron.h \
                           dolfin/TriMeshRefinement.h \
                           dolfin/Triangle.h \
                           dolfin/UnitCube.h \
                           dolfin/UnitSquare.h \
                           dolfin/Vertex.h \
                           dolfin/VertexIterator.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES            = libdolfin-mesh.la
libdolfin_mesh_la_SOURCES = Boundary.cpp \
                           BoundaryData.cpp \
                           BoundaryInit.cpp \
                           Cell.cpp \
                           CellIterator.cpp \
                           Edge.cpp \
                           EdgeIterator.cpp \
                           EdgeRefData.cpp \
                           Face.cpp \
                           FaceIterator.cpp \
                           GenericCell.cpp \
                           Mesh.cpp \
                           MeshData.cpp \
                           MeshHierarchy.cpp \
                           MeshInit.cpp \
                           MeshIterator.cpp \
                           MeshRefinement.cpp \
                           Point.cpp \
                           TetMeshRefinement.cpp \
                           Tetrahedron.cpp \
                           TriMeshRefinement.cpp \
                           Triangle.cpp \
                           UnitCube.cpp \
                           UnitSquare.cpp \
                           Vertex.cpp \
                           VertexIterator.cpp
libdolfin_mesh_la_LDFLAGS = -static
