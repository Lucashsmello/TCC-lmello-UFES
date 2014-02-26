DOC:=monografia
TMPDIR:=.tmpdir

LATEX:=xelatex

all: $(TMPDIR) $(DOC).tex
	$(LATEX) -output-directory=$(TMPDIR) $(DOC).tex
	bibtex $(TMPDIR)/$(DOC)
	mv $(TMPDIR)/$(DOC).pdf .
#	dvips $(TMPDIR)/$(DOC).dvi -o $(TMPDIR)/$(DOC).ps
#	ps2pdf $(TMPDIR)/$(DOC).ps


$(TMPDIR):
	mkdir .tmpdir

clean:
	rm -rf $(TMPDIR)

#latex
#latex
#dvips
#ps2pdf
