## Interactive dashboard

The repository also includes a small Streamlit dashboard for exploring the extracted data.

Run:

```powershell
pip install -r requirements_dashboard.txt
streamlit run streamlit_app.py
```

The dashboard shows:

- extracted recommendation candidates
- timestamped source table
- performance metrics
- event portfolio vs SPY/QQQ
- drawdowns
- ticker-level price charts with buy/sell/holding markers

The marker charts are useful for manual verification because each event keeps the timestamp URL and quote context.
