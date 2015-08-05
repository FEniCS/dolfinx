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

#ifndef __DOLFIN_IFPACK2_PRECONDITIONER_H
#define __DOLFIN_IFPACK2_PRECONDITIONER_H

#ifdef HAS_TRILINOS

#include <map>
#include <memory>
#include <Ifpack2_Factory.hpp>
#include <dolfin/common/types.h>
#include "TpetraMatrix.h"
#include "TpetraVector.h"
#include "TrilinosPreconditioner.h"

namespace dolfin
{

  /// Forward declarations
  class BelosKrylovSolver;

  /// Implements preconditioners using Ifpack2 from Trilinos

  class Ifpack2Preconditioner : public TrilinosPreconditioner, public Variable
  {

  public:

    /// Create a particular preconditioner object
    explicit Ifpack2Preconditioner(std::string type = "default");

    /// Destructor
    virtual ~Ifpack2Preconditioner();

    /// Set the preconditioner type on a solver
    virtual void set(BelosKrylovSolver& solver);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Initialise preconditioner based on Operator P
    virtual void init(std::shared_ptr<const TpetraMatrix> P);

    /// Return a list of available preconditioners
    static std::map<std::string, std::string> preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // name of preconditioner
    std::string _name;

    typedef Ifpack2::Preconditioner<double, int, dolfin::la_index,
                                    TpetraVector::node_type> prec_type;

    // Ifpack2 preconditioner, to be constructed from a
    // Tpetra Operator or Matrix
    Teuchos::RCP<prec_type> _prec;

  };

}

#endif

#endif
