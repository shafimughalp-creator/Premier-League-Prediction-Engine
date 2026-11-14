"""EPL Intelligence — Streamlit entry point.

Routing is real navigation through Streamlit query parameters:

    ?page=overview
    ?page=week
    ?page=match&id=<fixture id>
    ?page=table
    ?page=results[&mw=<matchweek>]

The prototype's markup, CSS and JS are served inside one ``st.html``
call so the document keeps its own layout, scroll container and fixed
sidebar/drawer. Streamlit supplies routing, the data module and the
server; everything visual is the approved prototype.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import data
import render

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="EPL Intelligence · Matchweek 11",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _read(name: str) -> str:
    return (ROOT / "static" / name).read_text(encoding="utf-8")


def _one(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


page = _one(st.query_params.get("page", "overview")) or "overview"
fixture_id = _one(st.query_params.get("id"))
mw = _one(st.query_params.get("mw"))

# Rebuild the live DATA object in place when an automation store changed
# (mtime fingerprint) or the TTL elapsed, so reruns never serve stale data
# and never rebuild unnecessarily.
data.refresh()
DATA = data.DATA

html, title = render.shell(page, fixture_id, mw=mw)

css = _read("style.css")
shell_css = _read("shell.css")

payload = (
    "<style>" + css + "</style>"
    "<style>" + shell_css + "</style>"
    + html
    + "<script>window.EPL_DATA="
    + json.dumps(DATA, ensure_ascii=False, separators=(",", ":"))
    + ";window.EPL_TITLE="
    + json.dumps(title, ensure_ascii=False)
    + ";</script>"
    # Served by Streamlit's static file server (see .streamlit/config.toml).
    # An external file keeps the JS out of st.html's sanitiser, which drops
    # inline scripts that contain literal HTML tags.
    + '<script src="app/static/app.js"></script>'
)

st.html(payload, unsafe_allow_javascript=True)
