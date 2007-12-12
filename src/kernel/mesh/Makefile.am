SUBDIRS                   = dolfin
nobase_include_HEADERS    = dolfin/dolfin_mesh.h \
                            dolfin/BoundaryComputation.h \
                            dolfin/BoundaryMesh.h \
                            dolfin/Cell.h \
                            dolfin/CellType.h \
                            dolfin/Edge.h \
                            dolfin/Face.h \
                            dolfin/Facet.h \
                            dolfin/GeometricPredicates.h \
                            dolfin/GTSInterface.h \
                            dolfin/IntersectionDetector.h \
                            dolfin/IntervalCell.h \
                            dolfin/LocalMeshCoarsening.h \
                            dolfin/LocalMeshRefinement.h \
                            dolfin/MeshConnectivity.h \
                            dolfin/MeshData.h \
                            dolfin/MeshEditor.h \
                            dolfin/MeshEntity.h \
                            dolfin/MeshEntityIterator.h \
                            dolfin/MeshFunction.h \
                            dolfin/MeshGeometry.h \
                            dolfin/MeshPartition.h \
                            dolfin/Mesh.h \
                            dolfin/MeshHierarchyAlgorithms.h \
                            dolfin/MeshHierarchy.h \
                            dolfin/MeshOrdering.h \
                            dolfin/MeshTopology.h \
                            dolfin/MPIManager.h \
                            dolfin/MPIMeshCommunicator.h \
                            dolfin/Point.h \
                            dolfin/PointCell.h \
                            dolfin/SubDomain.h \
                            dolfin/TetrahedronCell.h \
                            dolfin/TopologyComputation.h \
                            dolfin/TriangleCell.h \
                            dolfin/UniformMeshRefinement.h \
                            dolfin/UnitCube.h \
                            dolfin/UnitInterval.h \
                            dolfin/UnitSquare.h \
                            dolfin/Vertex.h
INCLUDES                  = @KERNEL_CFLAGS@
lib_LTLIBRARIES           = libdolfin-mesh.la
libdolfin_mesh_la_SOURCES = BoundaryComputation.cpp \
                            BoundaryMesh.cpp \
                            Cell.cpp \
                            CellType.cpp \
                            Edge.cpp \
                            Face.cpp \
                            Facet.cpp \
                            GeometricPredicates.cpp \
                            GTSInterface.cpp \
                            IntersectionDetector.cpp \
                            IntervalCell.cpp \
                            LocalMeshCoarsening.cpp \
                            LocalMeshRefinement.cpp \
                            MeshConnectivity.cpp \
                            Mesh.cpp \
                            MeshData.cpp \
                            MeshEditor.cpp \
                            MeshEntity.cpp \
                            MeshEntityIterator.cpp \
                            MeshGeometry.cpp \
                            MeshHierarchyAlgorithms.cpp \
                            MeshHierarchy.cpp \
                            MeshOrdering.cpp \
                            MeshPartition.cpp \
                            MeshTopology.cpp \
                            MPIManager.cpp \
                            MPIMeshCommunicator.cpp \
                            Point.cpp \
                            PointCell.cpp \
                            SubDomain.cpp \
                            TetrahedronCell.cpp \
                            TopologyComputation.cpp \
                            TriangleCell.cpp \
                            UniformMeshRefinement.cpp \
                            UnitCube.cpp \
                            UnitInterval.cpp \
                            UnitSquare.cpp \
                            Vertex.cpp
libdolfin_mesh_la_LDFLAGS = -static
