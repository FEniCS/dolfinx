"""Record FFCx JIT cache hits/misses and what the module name is built from.

Temporary, for the macOS wheel slowdown.
"""

import difflib

import ffcx.options

import ffcx.codegeneration.jit as fj

_orig_cached = fj.get_cached_module
_orig_cf = fj.compile_forms

lookups: list[tuple[str, bool]] = []
# (form signature, ffcx option signature, compilation signature)
parts: list[tuple[str, str, str]] = []


def _cached(module_name, object_names, cache_dir, timeout):
    obj, mod = _orig_cached(module_name, object_names, cache_dir, timeout)
    lookups.append((module_name, obj is not None))
    return obj, mod


def _cf(forms, options=None, **kwargs):
    options = options or {}
    p = ffcx.options.get_options(options)
    parts.append(
        (
            forms[0].signature(),
            fj._compute_option_signature(p),
            fj._compilation_signature(
                kwargs.get("cffi_extra_compile_args", []), kwargs.get("cffi_debug", False)
            ),
        )
    )
    return _orig_cf(forms, options=options, **kwargs)


fj.get_cached_module = _cached
fj.compile_forms = _cf


def _report(label, values):
    distinct = sorted(set(values))
    print(f"  {label}: calls={len(values)} distinct={len(distinct)}")
    if len(distinct) > 1:
        for line in list(difflib.unified_diff(distinct[0].split(", "), distinct[1].split(", ")))[:25]:
            print("     ", line.rstrip())


def pytest_sessionfinish(session, exitstatus):
    hits = sum(1 for _, h in lookups if h)
    names = [n for n, _ in lookups]
    print(
        f"\nJITLOG lookups={len(lookups)} hits={hits} misses={len(lookups) - hits} "
        f"distinct_names={len(set(names))}"
    )
    if parts:
        _report("form_signature", [a for a, _, _ in parts])
        _report("option_signature", [b for _, b, _ in parts])
        _report("compilation_signature", [c for _, _, c in parts])
