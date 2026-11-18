.PHONY: test verify figures paper docx supplementary refs all
test:
	python -m unittest discover -s tests -v
verify:
	python scripts/verify_reported_results.py
supplementary:
	python scripts/run_supplementary_experiments.py
refs:
	python scripts/check_references.py
figures:
	python scripts/make_figures.py --formats pdf png --dpi 600
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode NeuroPulse_manuscript.tex
docx: paper
	python scripts/make_docx.py
all: test verify figures paper docx
