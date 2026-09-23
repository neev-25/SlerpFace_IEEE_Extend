# GuardFace IEEE report

`main.tex` is the IEEE conference-format report. It uses the six PNG files in
`figs/`. All six are reproducible from the project code and synthetic inputs;
no personal camera images are included.

To regenerate the figures and analysis data from the repository root:

```powershell
python -m pip install -r report/requirements.txt
python report/analysis/generate_figures.py
```

The script writes the figures to `report/figs/`, the sample scores to
`report/analysis/pad_scores.csv`, and summary statistics to
`report/analysis/results.json`. It uses 30 generated examples in each of three
classes and a fixed random seed of 42. These are software-generated examples,
not physical presentation-attack tests.

Compile `report/main.tex` with IEEEtran in Overleaf or with a local LaTeX
distribution. Neev Mendapara's and Meet Kapadia's enrollment numbers remain
`ID pending` until supplied.
