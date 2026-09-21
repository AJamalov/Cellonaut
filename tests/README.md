# Test Suite

Run the complete suite from the repository root with the project environment:

```powershell
.venv\Scripts\python.exe -m pytest
```

Most tests are isolated unit or component tests and use temporary files, stubs,
or monkeypatching. Tests marked `gui` require a real offscreen Qt application;
the shared fixture also closes top-level windows and checks that a main window
did not leave background threads running. Keep the marker on any module that
creates a `QApplication`, `QWidget`, `CellonautMainWindow`, or Qt event loop.

Every collected test also receives one primary category automatically:

- `unit` for isolated behavior with controlled dependencies;
- `integration` for GUI, multi-component, and real-file-format behavior; or
- `release` for packaging, provenance, legal, and release-tool validation.

The additional `slow` trait marks comparatively expensive modules. For example,
`pytest -m unit` runs isolated tests, while `pytest -m "integration and not slow"`
runs the faster integration subset.

The main higher-level boundaries are:

- `test_pipeline_runtime.py` and the scientific regression modules exercise
  multi-module analysis behavior with small generated image data.
- `test_release_*.py`, `test_legal_documents.py`, and release-check-specific
  modules validate source packaging, build metadata, scripts, and public files.
- GUI modules validate real Qt widgets offscreen; they do not replace an
  interactive desktop run.
- Packaging smoke tests validate source-side assembly rules; they do not prove
  a built installer or a clean-machine Windows installation.

Use focused test commands while developing, then run the full suite before a
release. Mutation analysis is a diagnostic workflow: surviving or timed-out
mutants are reported for review and do not currently fail that workflow.
