"""Stamp dev versions, pin the stack and inject the MPI check for the wheels."""

import argparse
import pathlib
import re

from packaging.specifiers import SpecifierSet

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dev-stamp", required=True, help="replaces 0 in each .dev0 version")
parser.add_argument("--build-stack", action="store_true", help="also prepare UFL, FFCx and Basix")
args = parser.parse_args()

stamp = args.dev_stamp
projects = {"fenics-dolfinx": pathlib.Path("dolfinx/python/pyproject.toml")}
if args.build_stack:
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
check = pathlib.Path("dolfinx/.github/wheels/wheels_mpi_check.py.in").read_text(encoding="utf-8")
text = init.read_text(encoding="utf-8")
if text.count("# WHEELMPICHECK\n") != 1:
    raise RuntimeError(f"{init} has no WHEELMPICHECK placeholder")
init.write_text(text.replace("# WHEELMPICHECK\n", check), encoding="utf-8")
