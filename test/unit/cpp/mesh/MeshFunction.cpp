// Copyright (C) 2013 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-05-30
// Last changed:
//
// Unit tests for MeshFunction

#include <dolfin.h>
#include <gtest/gtest.h>

using namespace dolfin;

TEST(MeshFunctions, test_create_from_domains) {
  // Create mesh
  std::shared_ptr<Mesh> mesh(new UnitSquareMesh(3, 3));
  dolfin_assert(mesh);

  const std::size_t D = mesh->topology().dim();

  // Test setting all values
  for (std::size_t d = 0; d <= D; ++d)
  {
    // Create MeshDomains object
    MeshDomains mesh_domains;
    mesh_domains.init(D);

    mesh->init(d);

    // Build mesh domain
    std::map<std::size_t, std::size_t>& domain = mesh_domains.markers(d);
    for (std::size_t i = 0; i < mesh->num_entities(d); ++i)
      domain.insert(std::make_pair(i, i));

    // Create MeshFunction and test values
    MeshFunction<std::size_t> mf(mesh, d, mesh_domains);
    for (std::size_t i = 0; i < mf.size(); ++i)
      ASSERT_EQ(mf[i], i);
  }

  // Test setting some values only
  for (std::size_t d = 0; d <= D; ++d)
  {
    // Create MeshDomains object
    MeshDomains mesh_domains;
    mesh_domains.init(D);

    mesh->init(d);

    // Build mesh domain
    std::map<std::size_t, std::size_t>& domain = mesh_domains.markers(d);
    const std::size_t num_entities = mesh->num_entities(d);
    for (std::size_t i = num_entities/2; i < num_entities; ++i)
      domain.insert(std::make_pair(i, i));

    // Create MeshFunction and test values
    MeshFunction<std::size_t> mf(mesh, d, mesh_domains);
    for (std::size_t i = 0; i < num_entities/2; ++i)
      ASSERT_EQ(mf[i], std::numeric_limits<std::size_t>::max());
    for (std::size_t i = num_entities/2; i < mf.size(); ++i)
      ASSERT_EQ(mf[i], i);
  }
}

// Test all
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}