# Source Availability

Cellonaut is distributed under `GPL-3.0-or-later`. Each binary release at
https://github.com/AJamalov/Cellonaut/releases must include the matching
`Cellonaut-1.0.0-corresponding-sources.zip` archive.

The archive contains the exact Cellonaut source and pinned sources for the
redistributed Qt/PySide, Fiji, Bio-Formats, Trainable Weka Segmentation,
`fastremap`, and `fill-voids` components. Its `MANIFEST.json` records every
origin, byte size, and SHA-256 digest.

Build it from the clean `v1.0.0` tag with:

```text
python -m cellonaut.release_checks.corresponding_sources prepare --output installer_dist
```

The installer build generates `THIRD_PARTY_LICENSES/DEPENDENCIES.md` from the
exact packaged environment and includes it with the installed application; it
is not a generated file stored in this source checkout. Large bundled versions
are listed in `BUNDLED_COMPONENTS.md`. Fiji retains its own notices inside
`Fiji.app`.

Qt/PySide uses its GPL-3.0-only open-source option. ImageScience is not
redistributed and may be installed separately by users who need it for Weka.

Keep the source archive available for as long as the matching binaries are
distributed. Report missing source through
https://github.com/AJamalov/Cellonaut/issues.
