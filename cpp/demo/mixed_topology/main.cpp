// # Mixed topology
//
// Experimental demo.

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <iostream>
#include <vector>

using namespace dolfinx;

// Note: this demo is not currently intended to provide a fully
// functional example of using a mixed-topology mesh, but shows only the
// basic construction. Experimental.

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // Number of square cell in x-direction
  constexpr int nx_s = 2;
  // Number of triangle cells in x-direction
  constexpr int nx_t = 2;
  // Number of cells in y-direction
  constexpr int ny = 4;

  constexpr int num_s = nx_s * ny;
  constexpr int num_t = 2 * nx_t * ny;

  std::vector<double> x;
  for (int i = 0; i < nx_s + nx_t + 1; ++i)
  {
    for (int j = 0; j < ny + 1; ++j)
    {
      x.push_back(i);
      x.push_back(j);
    }
  }

  std::vector<std::int64_t> cells;
  std::vector<std::int32_t> offsets{0};
  for (int i = 0; i < nx_s + nx_t; ++i)
  {
    for (int j = 0; j < ny; ++j)
    {
      const int v_0 = j + i * (ny + 1);
      const int v_1 = v_0 + 1;
      const int v_2 = v_0 + ny + 1;
      const int v_3 = v_0 + ny + 2;
      if (i < nx_s)
      {
        cells.push_back(v_0);
        cells.push_back(v_1);
        cells.push_back(v_3);
        cells.push_back(v_2);
        offsets.push_back(cells.size());
      }
      else
      {
        cells.push_back(v_0);
        cells.push_back(v_1);
        cells.push_back(v_2);
        offsets.push_back(cells.size());

        cells.push_back(v_1);
        cells.push_back(v_2);
        cells.push_back(v_3);
        offsets.push_back(cells.size());
      }
    }
  }

  graph::AdjacencyList<std::int64_t> cells_list(cells, offsets);
  std::vector<std::int64_t> original_global_index(num_s + num_t);
  std::iota(original_global_index.begin(), original_global_index.end(), 0);
  std::vector<int> ghost_owners;
  std::vector<std::int32_t> cell_group_offsets{0, num_s, num_s + num_t,
                                               num_s + num_t, num_s + num_t};
  std::vector<std::int64_t> boundary_vertices;
  for (int j = 0; j < ny + 1; ++j)
  {
    boundary_vertices.push_back(j);
    boundary_vertices.push_back(j + (nx_s + nx_t + 1) * ny);
  }
  for (int i = 0; i < nx_s + nx_t + 1; ++i)
  {
    boundary_vertices.push_back((ny + 1) * i);
    boundary_vertices.push_back(ny + (ny + 1) * i);
  }

  std::ranges::sort(boundary_vertices);
  boundary_vertices.erase(
      std::unique(boundary_vertices.begin(), boundary_vertices.end()),
      boundary_vertices.end());

  std::vector<mesh::CellType> cell_types{mesh::CellType::quadrilateral,
                                         mesh::CellType::triangle};
  std::vector<fem::CoordinateElement<double>> elements;
  for (auto ct : cell_types)
    elements.push_back(fem::CoordinateElement<double>(ct, 1));

  {
    auto topo = std::make_shared<mesh::Topology>(mesh::create_topology(
        MPI_COMM_WORLD, cells_list, original_global_index, ghost_owners,
        cell_types, cell_group_offsets, boundary_vertices));

    auto topo_cells = topo->connectivity(2, 0);

    std::cout << "cells\n------\n";
    for (int i = 0; i < topo_cells->num_nodes(); ++i)
    {
      std::cout << i << " [";
      for (auto q : topo_cells->links(i))
        std::cout << q << " ";
      std::cout << "]\n";
    }

    topo->create_connectivity(1, 0);

    std::cout << "facets\n------\n";
    auto topo_facets = topo->connectivity(1, 0);
    for (int i = 0; i < topo_facets->num_nodes(); ++i)
    {
      std::cout << i << " [";
      for (auto q : topo_facets->links(i))
        std::cout << q << " ";
      std::cout << "]\n";
    }

    mesh::Geometry geom = mesh::create_geometry(MPI_COMM_WORLD, *topo, elements,
                                                cells_list, x, 2);

    mesh::Mesh<double> mesh(MPI_COMM_WORLD, topo, geom);
    std::cout << "num cells = " << mesh.topology()->index_map(2)->size_local()
              << "\n";
    for (auto q : mesh.topology()->entity_group_offsets(2))
      std::cout << q << " ";
    std::cout << "\n";
  }

  MPI_Finalize();

  return 0;
}
