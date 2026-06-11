# Optional static HTML report

A Streamlit app is better for exploration, but a static HTML report can be useful for sharing without setup.

Possible next step:

```text
scripts/build_static_report.py
```

The script would load the cleaned CSVs and write:

```text
docs/index.html
```

with embedded Plotly charts. GitHub Pages could then host the report.

Recommended order:

1. Keep private Streamlit dashboard for the group.
2. Finalize verified dataset.
3. Generate static HTML report after the methodology is stable.
4. Optionally publish the cleaned repo and GitHub Pages after grading.
