# Milestone 1 Known Limitations

- The browser is text-only. Speech-to-text, text-to-speech, microphones,
  WebRTC, and telephony belong to Milestone 2 or later.
- SQLite does not implement production row-level `FOR UPDATE` behavior. The
  unique slot constraint still prevents two committed appointments from using
  one slot in the personal demo.
- The durable SQLAlchemy conversation state is appropriate for this personal
  single-service demo. Multi-worker production checkpointing is not claimed.
- Organization facts such as opening hours, address, insurance, payments, and
  cancellation policy return controlled front-desk guidance until verified
  business content is configured.
- Semantic fallback quality depends on the configured provider, but invalid,
  low-confidence, timed-out, or unavailable output falls back safely.
- No authentication, multi-tenancy, RBAC, production monitoring, CI/CD,
  deployment, or admin dashboard is included.
- The test suite retains one upstream Starlette/httpx TestClient deprecation
  warning.
