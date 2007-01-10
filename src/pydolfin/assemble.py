def assemble(form, mesh, bc):

    import dolfin
    import ffc
    from commands import getoutput

    formname = "MyForm"
    options = ffc.FFC_OPTIONS
    language = "dolfin"
    modulename = "myform"

    # Compile form with FFC
    compiled_form = ffc.compile(form, formname, language, options)
    
    # FIXME: dolfin-swig should be implemented as a Python module

    # Generate Python module using SWIG
    output = getoutput("dolfin-swig " + formname + ".h " + modulename)
    print output

    # Import generated Python module
    imported_form = __import__(modulename)

    # Assemble and apply boundary conditions
    if compiled_form.rank == 0:
        M = imported_form.MyFormFunctional()
        value = dolfin.FEM_assemble(M, mesh)
        return value
    elif compiled_form.rank == 1:
        L = imported_form.MyFormLinearForm()
        b = dolfin.Vector()
        dolfin.FEM_assemble(L, b, mesh)
        if bc:
            dolfin.FEM_applyBC(b, mesh, L.test(), bc)
        return b
    elif compiled_form.rank == 2:
        a = imported_form.MyFormBilinearForm()
        A = dolfin.Matrix()
        dolfin.FEM_assemble(a, A, mesh)
        if bc:
            dolfin.FEM_applyBC(A, mesh, a.test(), bc)
        return A
    else:
        raise RuntimeError, "Unable to handle form of rank %d" % compiled_form.rank
