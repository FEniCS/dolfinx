FILENAME='dolfin_manual'

all:
	latex $(FILENAME).tex
	dvips -P pdf -o ../ps/$(FILENAME).ps $(FILENAME).dvi
	ps2pdf ../ps/$(FILENAME).ps ../pdf/$(FILENAME).pdf
