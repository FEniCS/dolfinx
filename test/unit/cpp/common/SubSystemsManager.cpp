// Copyright (C) 2015 Garth N. Wells
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
// First added:  2008-09-30
// Last changed: 2012-08-21
//
// Unit tests for SubSystemsManager

#include <dolfin.h>
#include <gtest/gtest.h>

using namespace dolfin;

// Test rewritten using Google Test
TEST(TestSubSystemsManager, test_petsc_user_init)
{
  // Test user initialisation of PETSc
#ifdef HAS_PETSC
  int argc = 0;
  char **argv = NULL;
  PetscInitialize(&argc, &argv, NULL, NULL);

  UnitSquareMesh(12, 12);
  PETScVector(MPI_COMM_WORLD, 30);
#endif
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
