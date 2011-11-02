// Copyright (C) 2010 Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2010-02-25
// Last changed: 2011-10-19

#ifndef __DOFLIN_TRILINOS_PRECONDITIONER_H
#define __DOFLIN_TRILINOS_PRECONDITIONER_H

#ifdef HAS_TRILINOS

#include <string>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/parameter/Parameters.h>

// Trilinos forward declarations
class Epetra_RowMatrix;
class Ifpack_Preconditioner;
namespace ML_Epetra
{
  class MultiLevelPreconditioner;
}

namespace dolfin
{

  // Forward declarations
  class EpetraKrylovSolver;
  class EpetraMatrix;

  /// This class is a wrapper for configuring Epetra preconditioners. It does
  /// not own a preconditioner. It can take a EpetraKrylovSolver and set the
  /// preconditioner type and parameters.

  class TrilinosPreconditioner : public Variable
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    explicit TrilinosPreconditioner(std::string method="default");

    /// Destructor
    virtual ~TrilinosPreconditioner();

    /// Set the precondtioner and matrix used in preconditioner
    virtual void set(EpetraKrylovSolver& solver, const EpetraMatrix& P);

    /// Return preconditioner name
    std::string name() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Setup the ML precondtioner
    void set_ml(AztecOO& solver, const Epetra_RowMatrix& P);

    /// Named preconditioner
    std::string preconditioner;

    // Available named preconditioners
    static const std::map<std::string, int> _preconditioners;

    boost::shared_ptr<Ifpack_Preconditioner> ifpack_preconditioner;
    boost::shared_ptr<ML_Epetra::MultiLevelPreconditioner> ml_preconditioner;
  };

}

#endif

#endif
