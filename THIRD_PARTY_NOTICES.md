# Third-Party Notices

Cellonaut packages open-source scientific software. Exact Python dependency
versions and copied licenses are provided in
`THIRD_PARTY_LICENSES/DEPENDENCIES.md`; bundled components are recorded
in `BUNDLED_COMPONENTS.md`.

| Component group | Purpose | Terms |
| --- | --- | --- |
| Python scientific stack | Image, array, table, and file processing | Primarily BSD, MIT, PSF, Apache, and MPL terms; see the generated inventory. |
| PySide6 / Qt | User interface | GPL-3.0-only open-source option selected by Cellonaut. No commercial license is claimed. |
| Lucide Icons | User-interface icons | ISC License, with MIT terms retained for Feather-derived icons. See `cellonaut/data/licenses/LUCIDE_LICENSE.txt`. |
| Cellpose, PyTorch, and `cpsam` / `cpsam_v2` | Cell segmentation | Cellpose code and model repository are marked BSD-3-Clause; PyTorch and transitive components carry their own terms. Cellpose training and annotated data are identified upstream as CC-BY-NC. |
| `fastremap` / `fill-voids` | Cellpose dependencies | LGPL-3.0 / LGPL-3.0-or-later. |
| PyImageJ and Java bridge libraries | Python–ImageJ integration | Apache, MIT, BSD, and Unlicense terms. |
| Fiji, ImageJ, Bio-Formats, and Trainable Weka Segmentation | Bundled image-analysis runtime | GPL-family and component-specific terms. Preserve their notices and corresponding source. |
| ImageScience | Optional Weka support | Not redistributed. Its terms prohibit redistribution without written permission. |

Cellonaut itself is `GPL-3.0-or-later`. Redistributions must retain `LICENSE`,
`COPYRIGHT`, this notice, `CITATIONS.md`, `BUNDLED_COMPONENTS.md`,
`SOURCE_AVAILABILITY.md`, and `THIRD_PARTY_LICENSES`, and must provide the
matching source archive. Do not add `imagescience.jar` without written
permission.

User-provided data, classifiers, and custom models are not part of Cellonaut’s
license. Load custom models only from trusted sources. DINOv3 and DINO-based
Cellpose models are not distributed with Cellonaut.

For publications, cite Cellonaut and the tools actually used. See
`CITATIONS.md` for exact references.

This summary is informational and not legal advice.
