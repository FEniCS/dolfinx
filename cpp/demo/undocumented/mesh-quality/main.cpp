// Copyright (C) 2013 Jan Blechta
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
// First added:  2013-11-20
// Last changed:
//
// This demo illustrates basic inspection of mesh quality.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // The output needs to be deterministic so that it can be piped to python
  parameters["std_out_all_processes"] = false;

  // Read mesh from file
  Mesh mesh("../dolfin_fine.xml.gz");

  // Print minimal and maximal radius ratio
  const std::pair<double, double> qminmax
    = MeshQuality::radius_ratio_min_max(mesh);
  if (dolfin::MPI::rank(mesh.mpi_comm()) == 0)
  {
    std::cout << "# Minimal radius ratio: " << qminmax.first  << std::endl;
    std::cout << "# Maximal radius ratio: " << qminmax.second << std::endl;
    std::cout <<                                                 std::endl;
  }

  // Print matplotlib code for generation of histogram
  const std::string hist = MeshQuality::radius_ratio_matplotlib_histogram(mesh);
  if (dolfin::MPI::rank(mesh.mpi_comm()) == 0)
  {
    std::cout << "# Execute following commands in python" << std::endl;
    std::cout << "# to get histogram of radius ratios:"   << std::endl;
    std::cout <<                                             std::endl;
    std::cout << "# ------------------------------------" << std::endl;
    std::cout << "# ------------------------------------" << std::endl;
    std::cout << hist                                     << std::endl;
    std::cout << "# ------------------------------------" << std::endl;
    std::cout << "# ------------------------------------" << std::endl;
    std::cout << "# or pass output of this program"       << std::endl;
    std::cout << "# to python, i.e.:"                     << std::endl;
    std::cout << "# $> ./demo_mesh-quality | python"      << std::endl;
  }

  // Write out mesh to XDMF.
  XDMFFile("mesh.xdmf").write(mesh);

  return 0;
}
