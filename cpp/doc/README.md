# DOLFINx C++ interface documentation

The C++ API documentation is generated from the source code using
[Doxygen](https://www.doxygen.nl/), and the Doxygen output is curated
and rendered using reStructured text with
[Sphinx](https://www.sphinx-doc.org/) and
[Breathe](https://breathe.readthedocs.io/).


## Requirements

- [Doxygen](https://www.doxygen.nl/)
- Python dependencies listed in `doc-requirements.txt`.

## Building

```bash
> doxygen Doxyfile
> make html
```

Sphinx/Breathe documentation is in `build/html`and the Doxygen generated
HTML is in `html/`.
