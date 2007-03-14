def interpolate(f, K, mesh):

    from dolfin import *
    
    N = mesh.numVertices()
    x = Vector(N)

    for vi in vertices(mesh):
        id = vi.index()
        p = vi.point()
        x[id] = f(p)

    # Define a function from computed degrees of freedom
    pif = Function(x, mesh, K)

    # Indicate memory ownership
    pif.thisown = False
    x.thisown = False

    return pif
