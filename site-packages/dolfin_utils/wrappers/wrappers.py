# Copyright (C) 2011 Marie E. Rognes
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
# Based on original implementation by Martin Alnes and Anders Logg
#
# Last changed: 2012-12-05

import includes as incl
from functionspace import *
from form import generate_form
from capsules import UFCElementNames

__all__ = ["generate_dolfin_code"]

# NB: generate_dolfin_namespace(...) assumes that if a coefficient has
# the same name in multiple forms, it is indeed the same coefficient:
parameters = {"use_common_coefficient_names": True}

#-------------------------------------------------------------------------------
def generate_dolfin_code(prefix, header, forms,
                         common_function_space=False, add_guards=False,
                         error_control=False):
    """Generate complete dolfin wrapper code with given generated names.

    @param prefix:
        String, prefix for all form names.
    @param header:
        Code that will be inserted at the top of the file.
    @param forms:
        List of UFCFormNames instances or single UFCElementNames.
    @param common_function_space:
        True if common function space, otherwise False
    @param add_guards:
        True iff guards (ifdefs) should be added
    @param error_control:
        True iff adaptivity typedefs (ifdefs) should be added
    """

    # Generate dolfin namespace
    namespace = generate_dolfin_namespace(prefix, forms, common_function_space,
                                          error_control)

    # Collect pieces of code
    code = [incl.dolfin_tag, header, incl.stl_includes, incl.dolfin_includes,
            namespace]

    # Add ifdefs/endifs if specified
    if add_guards:
        guard_name = ("%s_h" % prefix).upper()
        preguard = "#ifndef %s\n#define %s\n" % (guard_name, guard_name)
        postguard = "\n#endif\n\n"
        code = [preguard] + code + [postguard]

    # Return code
    return "\n".join(code)

#-------------------------------------------------------------------------------
def generate_dolfin_namespace(prefix, forms, common_function_space=False,
                              error_control=False):

    # Allow forms to represent a single space, and treat separately
    if isinstance(forms, UFCElementNames):
        return generate_single_function_space(prefix, forms)

    # Extract (common) coefficient spaces
    assert(parameters["use_common_coefficient_names"])
    spaces = extract_coefficient_spaces(forms)

    # Generate code for common coefficient spaces
    code = [apply_function_space_template(*space) for space in spaces]

    # Generate code for forms
    code += [generate_form(form, "Form_%s" % form.name) for form in forms]

    # Generate namespace typedefs (Bilinear/Linear & Test/Trial/Function)
    code += [generate_namespace_typedefs(forms, common_function_space,
                                         error_control)]

    # Wrap code in namespace block
    code = "\nnamespace %s\n{\n\n%s\n}" % (prefix, "\n".join(code))

    # Return code
    return code

#-------------------------------------------------------------------------------
def generate_single_function_space(prefix, space):
    code = apply_function_space_template("FunctionSpace",
                                         space.ufc_finite_element_classnames[0],
                                         space.ufc_dofmap_classnames[0])
    code = "\nnamespace %s\n{\n\n%s\n}" % (prefix, code)
    return code

#-------------------------------------------------------------------------------
def generate_namespace_typedefs(forms, common_function_space, error_control):

    # Generate typedefs as (fro, to) pairs of strings
    pairs = []

    # Add typedef for Functional/LinearForm/BilinearForm if only one
    # is present of each
    aliases = ["Functional", "LinearForm", "BilinearForm"]
    extra_aliases = {"LinearForm": "ResidualForm", "BilinearForm": "JacobianForm"}
    for rank in sorted(range(len(aliases)), reverse=True):
        forms_of_rank = [form for form in forms if form.rank == rank]
        if len(forms_of_rank) == 1:
            pairs += [("Form_%s" % forms_of_rank[0].name, aliases[rank])]
            if aliases[rank] in extra_aliases:
                extra_alias = extra_aliases[aliases[rank]]
                pairs += [("Form_%s" % forms_of_rank[0].name, extra_alias)]

    # Keepin' it simple: Add typedef for FunctionSpace if term applies
    if common_function_space:
        for i, form in enumerate(forms):
            if form.rank:
                pairs += [("Form_%s::TestSpace" % form.name, "FunctionSpace")]
                break

    # Add specialized typedefs when adding error control wrapppers
    if error_control:
        pairs += error_control_pairs(forms)

    # Combine data to typedef code
    typedefs = "\n".join("typedef %s %s;" % (to, fro) for (to, fro) in pairs)

    # Return typedefs or ""
    if not typedefs:
        return ""
    return "// Class typedefs\n" + typedefs + "\n"

#-------------------------------------------------------------------------------
def error_control_pairs(forms):
    assert (len(forms) == 11), "Expecting 11 error control forms"

    return [("Form_%s" % forms[8].name, "BilinearForm"),
            ("Form_%s" % forms[9].name, "LinearForm"),
            ("Form_%s" % forms[10].name, "GoalFunctional")]

