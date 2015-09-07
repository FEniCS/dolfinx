#ifndef __DOLFIN_MESH_H
#define __DOLFIN_MESH_H

// DOLFIN mesh interface

#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshDomains.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshEntityIteratorBase.h>
#include <dolfin/mesh/SubsetIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/FacetCell.h>
#include <dolfin/mesh/MeshConnectivity.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/DynamicMeshEditor.h>
#include <dolfin/mesh/LocalMeshValueCollection.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/MeshColoring.h>
#include <dolfin/mesh/MeshRenumbering.h>
#include <dolfin/mesh/MeshTransformation.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/SubMesh.h>
#include <dolfin/mesh/DomainBoundary.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/MeshQuality.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/MeshHierarchy.h>
#include <dolfin/mesh/MeshPartitioning.h>

#endif
