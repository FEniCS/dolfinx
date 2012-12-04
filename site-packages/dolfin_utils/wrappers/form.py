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
# Modified by Anders Logg 2011
#
# Last changed: 2011-11-15

from includes import snippets
from functionspace import *
from goalfunctional import generate_update_ec

__all__ = ["generate_form"]

#-------------------------------------------------------------------------------
def generate_form(form, classname):
    """Generate dolfin wrapper code associated with a form including
    code for function spaces used in form and typedefs

    @param form:
        A UFCFormNames instance
    @param classname
        Name of Form class.
    """

    # Generate code for Form_x_FunctionSpace_y subclasses
    wrap = apply_function_space_template
    blocks = [wrap("%s_FunctionSpace_%d" % (classname, i),
                   form.ufc_finite_element_classnames[i],
                   form.ufc_dofmap_classnames[i]) for i in range(form.rank)]

    # Add typedefs CoefficientSpace_z -> Form_x_FunctionSpace_y
    blocks += ["typedef CoefficientSpace_%s %s_FunctionSpace_%d;\n"
               % (form.coefficient_names[i], classname, form.rank + i)
               for i in range(form.num_coefficients)]

    # Generate Form subclass
    blocks += [generate_form_class(form, classname)]

    # Return code
    return "\n".join(blocks)
#-------------------------------------------------------------------------------
def generate_form_class(form, classname):
    "Generate dolfin wrapper code for a single Form class."

    # Generate constructors
    constructors = generate_form_constructors(form, classname)

    # Generate data for coefficient assignments
    (number, name) = generate_coefficient_map_data(form)

    # Generate typedefs for FunctionSpace subclasses for Coefficients
    typedefs = ["  // Typedefs", generate_typedefs(form, classname), ""]

    # Member variables for coefficients
    members = ["  dolfin::CoefficientAssigner %s;" % coefficient
               for coefficient in form.coefficient_names]
    if form.superclassname == "GoalFunctional":
        members += [generate_update_ec(form)]

    # Group typedefs and members together for inserting into template
    additionals = "\n".join(typedefs + ["  // Coefficients"] + members)

    # Wrap functions in class body
    code = apply_form_template(classname, constructors, number, name,
                               additionals, form.superclassname)

    # Return code
    return code
#-------------------------------------------------------------------------------
def generate_coefficient_map_data(form):
    """Generate data for code for the functions
    Form::coefficient_number and Form::coefficient_name."""

    # Write error if no coefficients
    if form.num_coefficients == 0:
        message = '''\
dolfin::dolfin_error("generated code for class %s",
                         "access coefficient data",
                         "There are no coefficients");''' % form.superclassname
        num  = "\n    %s\n    return 0;" % message
        name = '\n    %s\n    return "unnamed";' % message
        return (num, name)

    # Otherwise create switch
    ifstr = "if "
    num = ""
    name = '    switch (i)\n    {\n'
    for i, coeff in enumerate(form.coefficient_names):
        num += '    %s(name == "%s")\n      return %d;\n' % (ifstr, coeff, i)
        name += '    case %d:\n      return "%s";\n' % (i, coeff)
        ifstr = 'else if '

    # Create final return
    message = '''\
dolfin::dolfin_error("generated code for class %s",
                         "access coefficient data",
                         "Invalid coefficient");''' % form.superclassname
    num +=  "\n    %s\n    return 0;" % message
    name += '    }\n\n    %s\n    return "unnamed";' % message

    return (num, name)
#-------------------------------------------------------------------------------
def generate_form_constructors(form, classname):
    """Generate the dolfin::Form constructors for different
    combinations of references/shared pointers etc."""

    coeffs = ("referenced_coefficient", "shared_ptr_coefficient")
    spaces = ("referenced_space", "shared_ptr_space")

    # Treat functionals a little special
    if form.rank == 0:
        spaces = ("referenced_mesh", "shared_ptr_mesh")

    # Generate permutations of constructors
    constructors = []
    for space in spaces:
        constructors += [generate_constructor(form, classname, space)]
        if form.num_coefficients > 0:
            constructors += [generate_constructor(form, classname, space, coeff)
                             for coeff in coeffs]

    # Return joint constructor code
    return "\n\n".join(constructors)

#-------------------------------------------------------------------------------
def generate_constructor(form, classname, space_tag, coefficient_tag=None):
    "Generate a single Form constructor according to the given parameters."

    # Extract correct code snippets
    (argument, assign) = snippets[space_tag]

    # Construct list of arguments and function space assignments
    name = "V%d"
    if form.rank > 0:
        arguments = [argument % (name % i) for i in reversed(range(form.rank))]
        assignments = [assign % (i, name % i) for i in range(form.rank)]
    else:
        arguments = [argument]
        assignments = [assign]

    # Add coefficients to argument/assignment lists if specified
    if coefficient_tag is not None:
        (argument, assign) = snippets[coefficient_tag]
        arguments += [argument % name for name in form.coefficient_names]
        if form.rank > 0: # FIXME: To match old generated code only
            assignments += [""]
        assignments += [assign %(name, name) for name in form.coefficient_names]

    # Add assignment of _ufc_form variable
    line = "\n    _ufc_form = boost::shared_ptr<const ufc::form>(new %s());"
    # FIXME: To match old generated code only
    if form.rank == 0 and coefficient_tag is None:
        line =  "    _ufc_form = boost::shared_ptr<const ufc::form>(new %s());"
    assignments += [line % form.ufc_form_classname]

    # Construct list for initialization of Coefficient references
    initializers = ["%s(*this, %d)" % (name, number)
                    for (number, name) in enumerate(form.coefficient_names)]

    # Join lists together
    arguments = ", ".join(arguments)
    initializers = ", " + ", ".join(initializers) if initializers else ""
    body = "\n".join(assignments)

    # Wrap code into template
    args = {"classname": classname,
            "rank": form.rank,
            "num_coefficients": form.num_coefficients,
            "arguments": arguments,
            "initializers": initializers,
            "body": body,
            "superclass": form.superclassname
            }
    code = form_constructor_template % args
    return code
#-------------------------------------------------------------------------------
form_class_template = """\
class %(classname)s: public dolfin::%(superclass)s
{
public:

%(constructors)s

  // Destructor
  ~%(classname)s()
  {}

  /// Return the number of the coefficient with this name
  virtual std::size_t coefficient_number(const std::string& name) const
  {
%(coefficient_number)s
  }

  /// Return the name of the coefficient with this number
  virtual std::string coefficient_name(std::size_t i) const
  {
%(coefficient_name)s
  }

%(members)s
};
"""
#-------------------------------------------------------------------------------
# Template code for Form constructor
form_constructor_template = """\
  // Constructor
  %(classname)s(%(arguments)s):
    dolfin::%(superclass)s(%(rank)d, %(num_coefficients)d)%(initializers)s
  {
%(body)s
  }"""
#-------------------------------------------------------------------------------
def apply_form_template(classname, constructors, number, name, members,
                        superclass):
    args = {"classname": classname,
            "superclass": superclass,
            "constructors": constructors,
            "coefficient_number": number,
            "coefficient_name": name,
            "members": members}
    return form_class_template % args
#-------------------------------------------------------------------------------
