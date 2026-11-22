"""Print 'true' if a package autopush sends to local-buildcache was built from source.

Run with 'spack -e <env> python'. A package installed from a build cache
keeps the cache's '.spack/binary_distribution' file; one built from
source has none.
"""

import os

import spack.binary_distribution
import spack.environment
import spack.mirrors.mirror

mirror = spack.mirrors.mirror.MirrorCollection(binary=True)["local-buildcache"]
built = any(
    s.installed
    and not s.external
    and mirror.matches_binary(s, "push")
    and not os.path.exists(spack.binary_distribution.buildinfo_file_name(s.prefix))
    for s in spack.environment.active_environment().all_specs()
)
print("true" if built else "false")
