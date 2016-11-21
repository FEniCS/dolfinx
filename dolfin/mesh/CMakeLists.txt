set(DIR mesh)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryComputation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Cell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CellType.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DistributedMeshTools.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DomainBoundary.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DynamicMeshEditor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Edge.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Face.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FacetCell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Facet.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HexahedronCell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IntervalCell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalMeshData.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalMeshValueCollection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshColoring.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshConnectivity.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshData.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshDomains.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshEditor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshEntity.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshEntityIteratorBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshEntityIterator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshGeometry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshHierarchy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshOrdering.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshPartitioning.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshQuality.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshRelation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshRenumbering.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshSmoothing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshTopology.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshTransformation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshValueCollection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PeriodicBoundaryComputation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PointCell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/QuadrilateralCell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SubDomain.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SubMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SubsetIterator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/TetrahedronCell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/TopologyComputation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/TriangleCell.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Vertex.h
  )

# Install headers
install(FILES ${DOLFIN_${DIR}_HEADERS}
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR}
  COMPONENT Development)

# Append headers
set(DOLFIN_HEADERS ${DOLFIN_HEADERS} ${DOLFIN_${DIR}_HEADERS} PARENT_SCOPE)

# Append sources
set(DOLFIN_SOURCES
  ${DOLFIN_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryComputation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CellType.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DistributedMeshTools.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DynamicMeshEditor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Edge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Face.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FacetCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Facet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HexahedronCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IntervalCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalMeshData.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshColoring.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshConnectivity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshData.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshDomains.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshEditor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshEntity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshGeometry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshHierarchy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshOrdering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshPartitioning.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshQuality.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshRenumbering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshSmoothing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshTopology.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshTransformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PeriodicBoundaryComputation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PointCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/QuadrilateralCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SubDomain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SubMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TetrahedronCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TopologyComputation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TriangleCell.cpp
  PARENT_SCOPE)
