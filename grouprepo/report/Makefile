report.pdf: report.tex *.tex *.bib
	pdflatex --shell-escape report.tex
	bibtex report
	pdflatex --shell-escape report.tex
	pdflatex --shell-escape report.tex
