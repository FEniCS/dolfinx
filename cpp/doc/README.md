# DOLFINx C++ interface documentation

The C++ API documentation is generated from the source code using
[Doxygen](https://www.doxygen.nl/), and the Doxygen output is curated and
rendered using reStructured text with [Sphinx](https://www.sphinx-doc.org/) and
[Breathe](https://breathe.readthedocs.io/).


## Requirements

For the basic C++ documentation:

- [Doxygen](https://www.doxygen.nl/)

And then for the nicer-looking Sphinx documentation, Python with:

- [sphinx](https://www.sphinx-doc.org/)
- [breathe](https://breathe.readthedocs.io/)
- [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/)
- [myst-parser](https://myst-parser.readthedocs.io/en/latest/)

## Building

```bash
doxygen Doxyfile
python -m sphinx -b html source/ build/html/
```

Sphinx/Breathe documentation is in `build/html`and the Doxygen generated
HTML is in `html/`.
