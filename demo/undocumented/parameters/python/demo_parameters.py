"""This demo demonstrates the DOLFIN parameter system.

Try running this demo with

  python demo.py --bar 1 --solver_parameters.max_iterations 1000
"""

# Copyright (C) 2009 Johan Hake and Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-09-06
# Last changed: 2009-12-08

from __future__ import print_function
from dolfin import *

#--- Demo of global DOLFIN parameters ---

# Set some global DOLFIN parameters
parameters["linear_algebra_backend"] = "Eigen"

# Print global DOLFIN parameters
info(parameters, True)
print()

# Read parameters from file
file = File("parameters.xml")
parameters = Parameters("parameters")
file >> parameters
info(parameters, True)
print()

#--- Demo of nested parameter sets ---

# Create an application parameter set
application_parameters = Parameters("application_parameters")

# Create application parameters
application_parameters.add("foo", 1.0)
application_parameters.add("bar", 100)
application_parameters.add("baz", False)
application_parameters.add("pc", "amg")

# Create a solver parameter set
solver_parameters = Parameters("solver_parameters")

# Create solver parameters
solver_parameters.add("max_iterations", 100)
solver_parameters.add("tolerance", 1e-16)
solver_parameters.add("relative_tolerance", 1e-16, 1e-16, 1.0)

# Set range for parameter
solver_parameters.set_range("max_iterations", 0, 1000)

# Set some parameter values
solver_parameters["max_iterations"] = 500
solver_parameters["relative_tolerance"] = 0.1

# Set solver parameters as nested parameters of application parameters
application_parameters.add(solver_parameters)

# Parse command-line options
application_parameters.parse()

# Access parameter values
foo = application_parameters["foo"]
bar = application_parameters["bar"]
tol = application_parameters["solver_parameters"]["tolerance"]

# Print parameter values
print("foo =", foo)
print("bar =", bar)
print("tol =", tol)
print()

# Print application parameters
info(application_parameters, True)
print()

#--- Demo of Krylov solver parameters ---

# Set a parameter for the Krylov solver
solver = KrylovSolver()
solver.parameters["relative_tolerance"] = 1e-20

# Print Krylov solver parameters
info(solver.parameters, True)
print()

#--- Demo of updating a parameter set ---

# Create a subset of the application parameter set
parameter_subset = Parameters("parameter_subset")
parameter_subset.add("foo", 3.0)
nested_subset = Parameters("solver_parameters")
nested_subset.add("max_iterations", 850)
parameter_subset.add(nested_subset)

# Update application parameters
application_parameters.update(parameter_subset)
info(application_parameters, True)

# Can also update using a dictionary
parameter_subset = dict(foo = 1.5, solver_parameters = dict(max_iterations = 50))
application_parameters.update(parameter_subset)

# Or like this
parameter_subset = {"foo": 1.5, "solver_parameters": {"max_iterations": 50}}
application_parameters.update(parameter_subset)

# Print command-line option string
print("\nCommand-line option string")
print(application_parameters.option_string())

# Demostrate access to full info of parameters
def print_data(par, indent=""):
    print("\n" + indent + "Content of:", par.name())
    for key, par_info in par.iterdata():
        if isinstance(par_info,Parameters):
            print_data(par_info, indent + "    ")
        else:
            print(indent + key, par_info)

print_data(application_parameters)

# Direct creation of application parameter set
new_application_parameters = Parameters(
    "application_parameters",
    foo = 1.0,
    bar = 100,
    pc = "amg",
    solver_parameters = Parameters(
        "solver",
        max_iterations = 100,
        tolerance = 1e-16,
        relative_tolerance = (1e-16, 1e-16, 1.0),
        pcs = ("ilu", ["ilu","amg","icc","sor"])
        )
    )
