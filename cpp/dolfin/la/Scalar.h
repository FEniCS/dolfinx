// Copyright (C) 2007-2014 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <string>

namespace dolfin
{
namespace la
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
}