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
// First added:  2013-10-07
// Last changed:

#include <sstream>
#include <dolfin/common/MPI.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshFunction.h"
#include "MeshQuality.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::CellFunction<double>
MeshQuality::radius_ratios(boost::shared_ptr<const Mesh> mesh)
{
  // Create CellFunction
  CellFunction<double> cf(mesh, 0.0);

  // Compute radius ration
  for (CellIterator cell(*mesh); !cell.end(); ++cell)
    cf[*cell] = cell->radius_ratio();

  return cf;
}
//-----------------------------------------------------------------------------
std::pair<double, double> MeshQuality::radius_ratio_min_max(const Mesh& mesh)
{
  CellIterator cell(mesh);
  double qmin = cell->radius_ratio();
  double qmax = cell->radius_ratio();
  for (; !cell.end(); ++cell)
  {
    qmin = std::min(qmin, cell->radius_ratio());
    qmax = std::max(qmax, cell->radius_ratio());
  }

  qmin = MPI::min(qmin);
  qmax = MPI::min(qmax);
  return std::make_pair(qmin, qmax);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
MeshQuality::radius_ratio_histogram_data(const Mesh& mesh,
                                         std::size_t num_bins)
{
  std::vector<double> bins(num_bins), values(num_bins, 0.0);
  dolfin_assert(radius_ratio_min_max(mesh).second <= 1.0);

  const double interval = 1.0/static_cast<double>(num_bins);
  for (std::size_t i = 0; i < num_bins; ++i)
    bins[i] = static_cast<double>(i)*interval + interval/2.0;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const double ratio = cell->radius_ratio();
    const std::size_t slot = ratio/interval;
    values[slot] += 1.0;
  }

  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = MPI::sum(values[i]);

  return std::make_pair(bins, values);
}
//-----------------------------------------------------------------------------
std::string
MeshQuality::radius_ratio_matplolib_histogram(const Mesh& mesh,
                                              std::size_t num_intervals)
{
  // Compute data
  std::pair<std::vector<double>, std::vector<double> >
    data = radius_ratio_histogram_data(mesh, num_intervals);

  dolfin_assert(!data.first.empty());
  dolfin_assert(data.first.size() == data.second.size());

  // Create Matplotlib string
  std::stringstream matplotlib;
  matplotlib << "def plot_histogram():" << std::endl;
  matplotlib << "    import pylab" <<  std::endl;
  std::stringstream bins, values;
  bins   << "    bins = [" << data.first[0];
  values << "    values = [" << data.second[0];
  for (std::size_t i = 1; i < data.first.size(); ++i)
  {
    bins   << ", " << data.first[i];
    values << ", " << data.second[i];
  }
  bins << "]";
  values << "]";

  matplotlib << bins.str() << std::endl;
  matplotlib << values.str()  << std::endl;
  matplotlib << std::endl;

  matplotlib << "    pylab.xlim([0, 1])" <<  std::endl;
  matplotlib << "    width = 0.7*(bins[1] - bins[0])" << std::endl;
  matplotlib << "    pylab.xlabel('radius ratio')" << std::endl;
  matplotlib << "    pylab.ylabel('number of cells')" << std::endl;
  matplotlib << "    pylab.bar(bins, values, align='center', width=width)"
             << std::endl;
  matplotlib << "    pylab.show()" << std::endl;

  matplotlib << std::endl;
  matplotlib << "try:" << std::endl;
  matplotlib << "    import pylab"  << std::endl;
  matplotlib << "except ImportError:" << std::endl;
  matplotlib << "    print(\"Plotting mesh quality histogram requires Matplotlib\")" << std::endl;
  matplotlib << "else:" << std::endl;
  matplotlib << "    plot_histogram()" << std::endl;

  return matplotlib.str();
}
//-----------------------------------------------------------------------------
