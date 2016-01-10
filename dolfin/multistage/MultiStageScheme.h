// Copyright (C) 2013 Johan Hake
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
// First added:  2013-02-15
// Last changed: 2014-03-05

#ifndef __BUTCHERSCHEME_H
#define __BUTCHERSCHEME_H

#include <memory>
#include <vector>

#include <dolfin/common/Variable.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/Form.h>

namespace dolfin
{

  /// This class is a place holder for forms and solutions for a
  /// multi-stage Butcher tableau based method

  // Forward declarations
  class Form;
  class Function;
  class DirichletBC;
  class Constant;

  class MultiStageScheme : public Variable
  {
  public:

    /// Constructor
    MultiStageScheme(std::vector<std::vector<std::shared_ptr<const Form>>> stage_forms,
                     std::shared_ptr<const Form> last_stage,
                     std::vector<std::shared_ptr<Function> > stage_solutions,
                     std::shared_ptr<Function> u,
                     std::shared_ptr<Constant> t,
                     std::shared_ptr<Constant> dt,
                     std::vector<double> dt_stage_offset,
                     std::vector<int> jacobian_indices,
                     unsigned int order,
                     const std::string name,
                     const std::string human_form,
                     std::vector<std::shared_ptr<const DirichletBC>> bcs = {});

    /// Return the stages
    std::vector<std::vector<std::shared_ptr<const Form>>>& stage_forms();

    /// Return the last stage
    std::shared_ptr<const Form> last_stage();

    /// Return stage solutions
    std::vector<std::shared_ptr<Function> >& stage_solutions();

    /// Return solution variable
    std::shared_ptr<Function> solution();

    /// Return solution variable (const version)
    std::shared_ptr<const Function> solution() const;

    /// Return local time
    std::shared_ptr<Constant> t();

    /// Return local timestep
    std::shared_ptr<Constant> dt();

    /// Return local timestep
    const std::vector<double>& dt_stage_offset() const;

    /// Return the order of the scheme
    unsigned int order() const;

    /// Return boundary conditions
    std::vector<std::shared_ptr<const DirichletBC>> bcs() const;

    /// Return true if stage is implicit
    bool implicit(unsigned int stage) const;

    /// Return true if the whole scheme is implicit
    bool implicit() const;

    // Return a distinct jacobian index for a given stage if negative the
    // stage is explicit and hence no jacobian needed.
    int jacobian_index(unsigned int stage) const;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

  private:

    // Check sanity of arguments
    void _check_arguments();

    // Vector of forms for the different RK stages
    std::vector<std::vector<std::shared_ptr<const Form>>> _stage_forms;

    // A linear combination of solutions for the last stage
    std::shared_ptr<const Form> _last_stage;

    // Solutions for the different stages
    std::vector<std::shared_ptr<Function>> _stage_solutions;

    // The solution
    std::shared_ptr<Function> _u;

    // The local time
    std::shared_ptr<Constant> _t;

    // The local time step
    std::shared_ptr<Constant> _dt;

    // The time step offset. (c from the ButcherTableau)
    std::vector<double> _dt_stage_offset;

    // Map for distinct storage of jacobians
    std::vector<int> _jacobian_indices;

    // The order of the scheme
    unsigned int _order;

    // Is the scheme implicit
    bool _implicit;

    // A pretty print representation of the form
    std::string _human_form;

    // The boundary conditions
    std::vector<std::shared_ptr<const DirichletBC>> _bcs;

  };

}

#endif
