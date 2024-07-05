// # Checkpointing
//

#include <dolfinx.h>
#include <dolfinx/io/checkpointing.h>
#include <mpi.h>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<dolfinx::mesh::Mesh<float>>(
      mesh::create_rectangle<float>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                    {4, 4}, mesh::CellType::quadrilateral,
                                    part));

  dolfinx::io::checkpointing::write(mesh->comm(), "mesh.bp", "mesh-write",
                                    mesh);
  dolfinx::io::checkpointing::test();
  MPI_Finalize();
  return 0;
}