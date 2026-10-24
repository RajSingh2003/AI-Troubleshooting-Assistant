"""Builds the assistant from environment config (GEMINI_API_KEY in .env).

Gemini is called through its OpenAI-compatible endpoint, so the `openai`
SDK is used as the HTTP client with Google's base URL.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from .assistant import TroubleshootingAssistant
from .kb import ROOT, load_kb
from .retriever import Retriever

load_dotenv(ROOT / ".env")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_client(api_key: str | None = None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    # Extra retries: Gemini returns transient 503s ("high demand") under load.
    return OpenAI(api_key=key, base_url=GEMINI_BASE_URL, max_retries=5)


def build_assistant(api_key: str | None = None, offline: bool = False,
                    top_k: int = 3, threshold: float | None = None) -> TroubleshootingAssistant:
    entries = load_kb()
    client = None if offline else make_client(api_key)
    retriever = Retriever(entries, client=client)
    return TroubleshootingAssistant(retriever, client=client, top_k=top_k, threshold=threshold)
