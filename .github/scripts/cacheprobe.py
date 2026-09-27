"""Probe the FFCx JIT cache. Temporary, for the macOS wheel slowdown."""

import os
from pathlib import Path

from mpi4py import MPI

import basix
import ffcx
import ufl
from dolfinx.fem import Function, functionspace
from dolfinx.jit import get_options
from dolfinx.mesh import create_unit_cube

print(f"ufl={ufl.__version__} ffcx={ffcx.__version__} basix={basix.__version__}")
print(f"XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME')!r} HOME={os.environ.get('HOME')!r}")
print(f"cwd={Path.cwd()}")

opts = get_options()
cache_dir = opts["cache_dir"]
print(f"cache_dir={cache_dir!r} type={type(cache_dir).__name__} exists={Path(cache_dir).exists()}")

mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
V = functionspace(mesh, ("N1curl", 2))
Vv = functionspace(mesh, ("P", 1, (3,)))

sigs = []
for _ in range(10):
    v = Function(V)
    t = Function(Vv)
    sigs.append((ufl.inner(ufl.jump(v), t) * ufl.dS).signature())
print(f"UFL_SIGS calls={len(sigs)} distinct={len(set(sigs))} first={sigs[0][:16]}")

# Same question one level up, at the name FFCx actually caches on.
from ffcx.codegeneration.jit import _compilation_signature, _compute_option_signature  # noqa: E402
from ffcx.naming import compute_signature  # noqa: E402

p = ffcx.get_options({})
names = []
for _ in range(10):
    v = Function(V)
    t = Function(Vv)
    form = ufl.inner(ufl.jump(v), t) * ufl.dS
    names.append(
        compute_signature([form], _compute_option_signature(p) + _compilation_signature([], False))
    )
print(f"MODULE_NAMES calls={len(names)} distinct={len(set(names))} first={names[0][:16]}")
