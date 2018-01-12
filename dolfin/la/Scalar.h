// Copyright (C) 2007-2014 Anders Logg
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
// Modified by Garth N. Wells, 2007-2015.
// Modified by Ola Skavhaug, 2007.
// Modified by Martin Alnaes, 2014.

#pragma once

#include <dolfin/common/MPI.h>
#include <string>

namespace dolfin
{

/// This class represents a real-valued scalar quantity and
/// implements the GenericTensor interface for scalars.

class Scalar
{
public:
  /// Create zero scalar
  Scalar(MPI_Comm comm) : _value(0.0), _local_increment(0.0), _mpi_comm(comm) {}

  /// Destructor
  ~Scalar() {}

  /// Add to value
  void add(double x) { _local_increment += x; }

  /// Set all entries to zero
  void zero()
  {
    _value = 0.0;
    _local_increment = 0.0;
  }

  /// Finalize assembly
  void apply()
  {
    _value = _value + MPI::sum(_mpi_comm.comm(), _local_increment);
    _local_increment = 0.0;
  }

  /// Return MPI communicator
  MPI_Comm mpi_comm() const { return _mpi_comm.comm(); }

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const
  {
    std::stringstream s;
    s << "<Scalar value " << _value << ">";
    return s.str();
  }

  /// Get final value (assumes prior apply(), not part of
  /// GenericTensor interface)
  double value() const { return _value; }

private:
  // Value of scalar
  double _value;

  // Local intermediate value of scalar prior to apply call
  double _local_increment;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;
};
}
