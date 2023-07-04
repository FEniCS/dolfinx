#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <iostream>
#include <vector>

// Note: this demo is not currently intended to provide a fully functional
// example of using a mixed-topology mesh, but shows only the
// basic constrution. Experimental.

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  const int nx_s = 2;
  const int nx_t = 2;
  const int ny = 4;

  const int num_s = nx_s * ny;
  const int num_t = 2 * nx_t * ny;

  std::vector<double> x;
  for (int i = 0; i < nx_s + nx_t + 1; ++i)
  {
    for (int j = 0; j < ny + 1; ++j)
    {
      x.push_back(i);
      x.push_back(j);
      x.push_back(0);
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

  dolfinx::graph::AdjacencyList<std::int64_t> cells_list(cells, offsets);

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

  std::sort(boundary_vertices.begin(), boundary_vertices.end());
  boundary_vertices.erase(
      std::unique(boundary_vertices.begin(), boundary_vertices.end()),
      boundary_vertices.end());

  std::vector<dolfinx::mesh::CellType> cell_types{
      dolfinx::mesh::CellType::quadrilateral,
      dolfinx::mesh::CellType::triangle};
  std::vector<dolfinx::fem::CoordinateElement<double>> elements;
  for (auto ct : cell_types)
    elements.push_back(dolfinx::fem::CoordinateElement(ct, 1));

  {
    auto topo = dolfinx::mesh::create_topology(
        MPI_COMM_WORLD, cells_list, original_global_index, ghost_owners,
        cell_types, cell_group_offsets, boundary_vertices);

    auto topo_cells = topo.connectivity(2, 0);

    for (int i = 0; i < topo_cells->num_nodes(); ++i)
    {
      std::cout << i << " [";
      for (auto q : topo_cells->links(i))
        std::cout << q << " ";
      std::cout << "]\n";
    }

    topo.create_connectivity(1, 0);

    auto topo_facets = topo.connectivity(1, 0);
    for (int i = 0; i < topo_facets->num_nodes(); ++i)
    {
      std::cout << i << " [";
      for (auto q : topo_facets->links(i))
        std::cout << q << " ";
      std::cout << "]\n";
    }

    auto geom = dolfinx::mesh::create_geometry(MPI_COMM_WORLD, topo, elements,
                                               cells_list, x, 2);

    dolfinx::mesh::Mesh mesh(MPI_COMM_WORLD, topo, geom);
    std::cout << "num cells = " << mesh.topology().index_map(2)->size_local()
              << "\n";
    for (auto q : mesh.topology().entity_group_offsets(2))
      std::cout << q << " ";
    std::cout << "\n";
  }

  MPI_Finalize();

  return 0;
}
