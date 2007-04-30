SUBDIRS                   = dolfin
nobase_include_HEADERS    = dolfin/dolfin_mesh.h \
                            dolfin/UnitCube.h \
                            dolfin/UnitSquare.h \
                            dolfin/MeshConnectivity.h \
                            dolfin/MeshEditor.h \
                            dolfin/MeshEntity.h \
                            dolfin/MeshEntityIterator.h \
                            dolfin/MeshFunction.h \
                            dolfin/MeshGeometry.h \
                            dolfin/MeshHierarchy.h \
                            dolfin/MeshHierarchyAlgorithms.h \
                            dolfin/MeshOrdering.h \
                            dolfin/MeshPartition.h \
                            dolfin/MeshTopology.h \
                            dolfin/Mesh.h \
                            dolfin/MeshData.h \
                            dolfin/Vertex.h \
                            dolfin/Edge.h \
                            dolfin/Face.h \
                            dolfin/Facet.h \
                            dolfin/Cell.h \
                            dolfin/TopologyComputation.h \
                            dolfin/CellType.h \
                            dolfin/Interval.h \
                            dolfin/SubDomain.h \
                            dolfin/Triangle.h \
                            dolfin/Tetrahedron.h \
                            dolfin/UniformMeshRefinement.h \
                            dolfin/LocalMeshRefinement.h \
                            dolfin/LocalMeshCoarsening.h \
                            dolfin/Point.h \
                            dolfin/BoundaryComputation.h \
                            dolfin/BoundaryMesh.h \
                            dolfin/GeometricPredicates.h \
                            dolfin/GTSInterface.h \
                            dolfin/IntersectionDetector.h 
INCLUDES                  = @KERNEL_CFLAGS@
lib_LTLIBRARIES           = libdolfin-mesh.la
libdolfin_mesh_la_SOURCES = UnitCube.cpp \
                            UnitSquare.cpp \
                            MeshConnectivity.cpp \
                            MeshEditor.cpp \
                            MeshEntity.cpp \
                            MeshEntityIterator.cpp \
                            MeshGeometry.cpp \
                            MeshHierarchy.cpp \
                            MeshHierarchyAlgorithms.cpp \
                            MeshOrdering.cpp \
                            MeshPartition.cpp \
                            MeshTopology.cpp \
                            Mesh.cpp \
                            MeshData.cpp \
                            Vertex.cpp \
                            Edge.cpp \
                            Face.cpp \
                            Facet.cpp \
                            Cell.cpp \
                            TopologyComputation.cpp \
                            Interval.cpp \
                            Triangle.cpp \
                            Tetrahedron.cpp \
                            CellType.cpp \
                            UniformMeshRefinement.cpp \
                            LocalMeshRefinement.cpp \
                            LocalMeshCoarsening.cpp \
                            Point.cpp \
                            BoundaryComputation.cpp \
                            BoundaryMesh.cpp \
                            GeometricPredicates.cpp \
                            GTSInterface.cpp \
                            IntersectionDetector.cpp
libdolfin_mesh_la_LDFLAGS = -static
