# Bundled and Optional Components

Bundled components in the Cellonaut 1.0.0 Windows release:

| Component | Included version | Source and terms |
| --- | --- | --- |
| Fiji | 2.18.1-SNAPSHOT; build `c691dc761719086b49a9f927e77d744ae4e5a816`; archive snapshot 20260718-0417 | [Exact source](https://github.com/fiji/fiji/tree/c691dc761719086b49a9f927e77d744ae4e5a816); component-specific licenses. |
| ImgLib2 Python bridge | `imglib2-imglyb` 1.1.0; `imglib2-unsafe` 1.0.0 | [ImgLyb](https://github.com/imglib/imglib2-imglyb/tree/imglib2-imglyb-1.1.0) and [Unsafe](https://github.com/imglib/imglib2-unsafe/tree/imglib2-unsafe-1.0.0); BSD-2-Clause. |
| Azul Zulu OpenJDK | Java 21.0.7 | [OpenJDK source](https://www.azul.com/products/core/open-source-projects/); bundled notices apply. |
| Bio-Formats | 8.5.0; build `877c317e4e396381dc76e56c1539b24947f71dce` | [Exact source](https://github.com/ome/bioformats/tree/877c317e4e396381dc76e56c1539b24947f71dce); GPL-family terms. |
| Trainable Weka Segmentation | 4.0.0; build `a55f593c08ea3fd94e860a7cd1878f58ed1bff1b` | [Exact source](https://github.com/fiji/Trainable_Segmentation/tree/a55f593c08ea3fd94e860a7cd1878f58ed1bff1b); GPL-3.0. |
| ImageScience | Optional; not bundled | [Official terms](https://imagescience.org/meijering/software/); release assets must not contain `imagescience.jar`. |
| Cellpose model weights | `cpsam`, `cpsam_v2`; revision `7c61431b5fbb078f3296754bd15d9f51b320f837` | [Pinned repository](https://huggingface.co/mouseland/cellpose-sam/tree/7c61431b5fbb078f3296754bd15d9f51b320f837); repository marked BSD-3-Clause. `cpsam`: 1,233,587,898 bytes, SHA-256 `e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2`; `cpsam_v2`: 1,233,586,851 bytes, SHA-256 `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`. DINO-based models are not distributed. |

Preserve bundled notices and publish the matching source archive described in
`SOURCE_AVAILABILITY.md`. For exact Python package versions and licenses, see
`THIRD_PARTY_LICENSES/DEPENDENCIES.md`. This document is not legal advice.
