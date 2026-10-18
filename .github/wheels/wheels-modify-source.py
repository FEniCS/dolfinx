"""Stamp dev versions, pin the stack and inject the MPI check for the wheels."""

import os
import pathlib
import re

from packaging.specifiers import SpecifierSet

stamp = os.environ["DEV_STAMP"]
projects = {"fenics-dolfinx": pathlib.Path("dolfinx/python/pyproject.toml")}
if os.environ["BUILD_STACK"] == "true":
    projects |= {
        name: pathlib.Path(name.removeprefix("fenics-"), "pyproject.toml")
        for name in ("fenics-ufl", "fenics-ffcx", "fenics-basix")
    }

versions = {}
for name, path in projects.items():
    text, n = re.subn(
        r'^version = "(.+)\.dev0"$',
        rf'version = "\1.dev{stamp}"',
        path.read_text(encoding="utf-8"),
        flags=re.M,
    )
    if n != 1:
        raise RuntimeError(f"{path} has no .dev0 version to stamp")
    path.write_text(text, encoding="utf-8")
    versions[name] = re.search(r'^version = "(.+)"$', text, flags=re.M).group(1)
    print(f"{name} {versions[name]}")


# PyPI rejects direct URL dependencies, so those are pinned too.
def _pin(match):
    name, spec = match.groups()
    if not spec.startswith("@") and versions[name] not in SpecifierSet(spec, prereleases=True):
        raise RuntimeError(f"{name} {versions[name]} does not satisfy {spec}")
    return f'"{name}=={versions[name]}"'


for name, path in projects.items():
    deps = "|".join(re.escape(d) for d in versions if d != name)
    if deps:
        text = re.sub(
            rf'"({deps})\s*(@[^";]*|[<>=!~][^";]*)"', _pin, path.read_text(encoding="utf-8")
        )
        path.write_text(text, encoding="utf-8")

init = pathlib.Path("dolfinx/python/dolfinx/__init__.py")
check = pathlib.Path("dolfinx/.github/wheels/wheels_mpi_check.py").read_text(encoding="utf-8")
text = init.read_text(encoding="utf-8")
if text.count("# WHEELMPICHECK\n") != 1:
    raise RuntimeError(f"{init} has no WHEELMPICHECK placeholder")
init.write_text(text.replace("# WHEELMPICHECK\n", check), encoding="utf-8")
