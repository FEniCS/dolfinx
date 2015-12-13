// Copyright (C) 2014 Chris Richardson
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

#ifndef __DOLFIN_MUELU_PRECONDITIONER_H
#define __DOLFIN_MUELU_PRECONDITIONER_H

#ifdef HAS_TRILINOS

//#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>

#include "BelosKrylovSolver.h"
#include "TrilinosPreconditioner.h"
#include "TpetraMatrix.h"
#include "TpetraVector.h"

namespace dolfin
{

  /// Forward declarations
  class BelosKrylovSolver;

  /// Implements Muelu preconditioner from Trilinos

  class MueluPreconditioner : public TrilinosPreconditioner, public Variable
  {

  public:

    /// Create a particular preconditioner object
    MueluPreconditioner();

    /// Destructor
    virtual ~MueluPreconditioner();

    /// Set the preconditioner on a solver
    virtual void set(BelosKrylovSolver& solver);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Initialise preconditioner based on Operator P
    virtual void init(std::shared_ptr<const TpetraMatrix> P);

    /// Default parameter values
    static Parameters default_parameters();

  private:

    typedef Tpetra::Operator<double, int, dolfin::la_index,
                             TpetraVector::node_type> op_type;

    typedef MueLu::TpetraOperator<double, int, dolfin::la_index,
                                  TpetraVector::node_type> prec_type;

    // Muelu preconditioner, to be constructed from a Tpetra Operator
    // or Matrix
    Teuchos::RCP<prec_type> _prec;

  };

}

#endif

#endif
