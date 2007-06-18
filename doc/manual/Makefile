FILENAME='dolfin-user-manual'

all:
	latex $(FILENAME).tex

final:
	latex $(FILENAME).tex
	bibtex $(FILENAME)
	latex $(FILENAME).tex
	makeindex $(FILENAME)
	latex $(FILENAME)
	dvips -P pdf -o $(FILENAME).ps $(FILENAME).dvi
	ps2pdf $(FILENAME).ps $(FILENAME).pdf
