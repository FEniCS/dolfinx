
# Based on original implementation by Martin Alnes and Anders Logg

__all__ = ["dolfin_tag", "stl_includes", "dolfin_includes", "snippets"]

dolfin_tag = "// DOLFIN wrappers"

stl_includes = """\
// Standard library includes
#include <string>
"""

dolfin_includes = """\
// DOLFIN includes
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Restriction.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/CoefficientAssigner.h>
#include <dolfin/adaptivity/ErrorControl.h>
#include <dolfin/adaptivity/GoalFunctional.h>"""
#-------------------------------------------------------------------------------
snippets = {"shared_ptr_space":
                ("boost::shared_ptr<const dolfin::FunctionSpace> %s",
                 "    _function_spaces[%d] = %s;"),
            "referenced_space":
                ("const dolfin::FunctionSpace& %s",
                 "    _function_spaces[%d] = reference_to_no_delete_pointer(%s);"),
            "shared_ptr_mesh":
                ("boost::shared_ptr<const dolfin::Mesh> mesh",
                 "    _mesh = mesh;"),
            "referenced_mesh":
                ("const dolfin::Mesh& mesh",
                 "    _mesh = reference_to_no_delete_pointer(mesh);"),
            "shared_ptr_coefficient":
                ("boost::shared_ptr<const dolfin::GenericFunction> %s",
                 "    this->%s = *%s;"),
            "referenced_coefficient":
                ("const dolfin::GenericFunction& %s",
                 "    this->%s = %s;"),
            "functionspace":
                ("TestSpace", "TrialSpace")
            }
#-------------------------------------------------------------------------------

