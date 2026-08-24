# Draft analysis notes

Provisional write-ups of analyses, benchmarks and investigations: the numbers,
how they were produced, and a **proposed** conclusion marked as proposed.

**This directory is deliberately outside `docs/`.** Anything under `docs/` is
built by MkDocs and becomes reachable on the published site, including via its
search index and sitemap. These notes record conclusions that have not been
reviewed and agreed, so they must not be published. `mkdocs.yml` also carries
an `exclude_docs: drafts/` guard in case a `docs/drafts/` is ever recreated by
mistake.

Conventions:

- One file per topic, named `YYYY-MM-DD_<topic>.md`.
- Open with provenance: date, the script or command that produced the numbers,
  and a one-line reliability note saying what they can bear.
- Mark status explicitly (`[PROPOSED]`, `[UNRESOLVED]`, superseded).
- Corrections are first-class entries: strike the old claim, give the date and
  the reason, and keep it visible rather than editing it away.

Once a conclusion has been reviewed and agreed, it graduates into the real
documentation under `docs/` — not by moving the draft, but by writing the
finding up there.
