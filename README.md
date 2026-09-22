# Cellonaut

Cellonaut measures cells and biological structures in microscopy images without code.
It uses Fiji/Weka to create masks and Cellpose to identify individual cells, then
exports measurements, tables, overlays, and processed images.

Current release: **Cellonaut 1.0.1**. The packaged app includes Python, Fiji,
Java, and the supported Cellpose models (`cpsam` and `cpsam_v2`).
This release is available for 64-bit Windows only.

## Download and Install

Download the Windows package from
[GitHub Releases](https://github.com/AJamalov/Cellonaut/releases). Keep all
installer files together. GitHub's automatically generated **Source code**
archives are not runnable Cellonaut packages.

### Windows 10 or 11, 64-bit

Download `Cellonaut-1.0.1-windows.exe`, every matching `.bin` file, and
`Cellonaut-1.0.1-windows.sha256`.

Choose **Automatic** for compatible NVIDIA GPU acceleration with CPU fallback,
or **CPU only** to always use the CPU. AMD and Intel graphics use the CPU.

The installer is unsigned, so Windows may show an **Unknown publisher** or
SmartScreen warning.

## Getting Started

Open **Tutorial** in the Cellonaut sidebar for a guided introduction using the
included example files. It covers setup, channels, masks, processing,
measurements, previews, runs, and results.

Use the **Help** tab or a section's **Help** link for complete instructions and
explanations of individual settings.

## ImageScience Support

Some Trainable Weka Segmentation features require ImageScience, which is not
included. On Windows, close Cellonaut, open **Fiji (optional plugins)** from the
Start menu, and enable the **ImageScience** update site through
**Help > Update... > Manage update sites**. Restart Fiji, then Cellonaut.

## Citation and Licensing

Cellonaut is licensed under `GPL-3.0-or-later`. Cite Cellonaut using
[CITATION.cff](CITATION.cff). References for the analysis tools are in
[CITATIONS.md](CITATIONS.md).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[BUNDLED_COMPONENTS.md](BUNDLED_COMPONENTS.md), and
[SOURCE_AVAILABILITY.md](SOURCE_AVAILABILITY.md) for licensing and source details.
