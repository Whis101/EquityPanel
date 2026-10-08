"""Turn audit results into a self-contained HTML report, with an optional checked LLM summary.

facts.py needs numpy + pandas only. charts.py and html.py need the [report] extra
(matplotlib, jinja2); summary.py needs the [llm] extra (anthropic).
"""
