#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  std::vector<std::int64_t> cells{0, 1, 2, 3, 2, 3, 4};
  std::vector<std::int32_t> offsets{0, 4, 7};
  dolfinx::graph::AdjacencyList<std::int64_t> cells_list(cells, offsets);

  std::vector<std::int64_t> original_global_index{0, 1};

  std::vector<int> ghost_owners;

  std::vector<std::int32_t> cell_group_offsets{0, 1, 1, 1};

  std::vector<std::int64_t> boundary_vertices = cells;

  std::vector<dolfinx::mesh::CellType> cell_types{
      dolfinx::mesh::CellType::quadrilateral,
      dolfinx::mesh::CellType::triangle};

  dolfinx::mesh::create_topology(
      MPI_COMM_WORLD, cells_list, original_global_index, ghost_owners,
      cell_types, cell_group_offsets, boundary_vertices);

  MPI_Finalize();

  return 0;
}
