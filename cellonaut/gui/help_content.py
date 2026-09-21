"""Task-focused in-app user manual for Cellonaut."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable

from cellonaut.config.defaults import (
    IMAGE_PROCESSING_STEP_DEFINITIONS,
    MASK_PROCESSING_STEP_DEFINITIONS,
    MEASUREMENT_METADATA,
)
from cellonaut.version import __version__


CELLPOSE_DIAMETER_HELP = (
    "Optional cell diameter in pixels, used to rescale the image. Blank keeps the original scale; "
    "Cellpose-SAM does not estimate diameter automatically."
)
IMAGESCIENCE_HELP = (
    "ImageScience must be installed separately if your 2D Weka classifier uses any of the "
    "Derivatives, Laplacian, and Structure features."
)


IMAGE_PROCESSING_GUIDANCE = {
    "Subtract Background": "Runs Fiji's Subtract Background command, including optional Light background, Sliding paraboloid, and Disable smoothing settings. Before mask creation, radii are applied in sequence to the Weka input. Before measurement, each radius creates a separate corrected measurement result after the other image steps.",
    "Gaussian Blur": "Reduces fine noise with a sigma in pixels, but also softens edges.",
    "Median": "Removes local noise while preserving edges. Set a radius in pixels.",
    "Despeckle": "Removes isolated noise with a 3-by-3 median filter. No value needed.",
    "Remove Outliers": "Fiji Process > Noise > Remove Outliers. Set neighborhood radius (px), threshold (raw intensity difference), and Bright or Dark. Only pixels that differ enough from the local median are replaced.",
    "Enhance contrast": "Runs Fiji's Enhance Contrast command with optional Normalize and Equalize histogram settings. Without either option, this step changes the display range but not pixel values; Equalize ignores the saturation and Normalize settings.",
    "Apply LUT": "Applies the current ImageJ lookup-table transformation to pixel values.",
    "Smooth": "Averages each pixel's 3-by-3 neighbourhood, repeated for the chosen number of passes.",
    "Convert Bit Depth": "Runs Fiji Image > Type on the working copy before later steps. The Scale setting matches Fiji Edit > Options > Conversions; scaling uses the current display range when converting to 8-bit or 16-bit.",
}

MASK_PROCESSING_GUIDANCE = {
    "Analyze skeleton": "Skeletonizes a copy of the mask at this point in the recipe and exports skeleton, branch, endpoint, and merged-junction counts.",
    "Erode": "Runs Fiji Process > Binary > Erode with the saved Iterations, Count, and Pad edges settings.",
    "Dilate": "Runs Fiji Process > Binary > Dilate with the saved Iterations and Count settings.",
    "Open": "Runs Fiji Process > Binary > Open with the saved Binary Options settings.",
    "Close": "Runs Fiji Process > Binary > Close with the saved Binary Options settings.",
    "Fill Holes": "Runs Fiji Process > Binary > Fill Holes on the entire binary mask.",
    "Watershed": "Runs Fiji Process > Binary > Watershed to separate touching mask regions.",
    "Outline": "Runs Fiji Process > Binary > Outline and replaces the mask with object outlines.",
    "Skeletonize": "Runs Fiji Process > Binary > Skeletonize and replaces the mask with the skeleton. This differs from Analyze skeleton, which measures a copy.",
    "Analyze Particles": "Runs Fiji Analyze > Analyze Particles and keeps the accepted particle mask. Set Size and Circularity ranges; Options contains Exclude on Edges and Include Holes. Size is in pixels because the working mask has no spatial calibration.",
    "Translate": "Runs Fiji Image > Transform > Translate on the binary mask using X and Y pixel offsets (positive X moves right, positive Y moves down). Interpolation is None, newly exposed edges become black, and the source image is unchanged.",
}


def _definition_items(definitions: Iterable[dict[str, Any]], guidance: dict[str, str]) -> str:
    items = []
    for definition in definitions:
        label = str(definition["label"])
        items.append(f"<li><b>{escape(label)}</b>: {escape(guidance[label])}</li>")
    return "\n".join(items)


def _measurement_items() -> str:
    items = []
    for metadata in MEASUREMENT_METADATA.values():
        label = str(metadata["label"])
        unit = str(metadata.get("unit", ""))
        unit_text = f" ({escape(unit)})" if unit else ""
        items.append(f"<li><b>{escape(label)}{unit_text}</b>: {escape(str(metadata['description']))}</li>")
    return "\n".join(items)


IMAGE_PROCESSING_ITEMS = _definition_items(IMAGE_PROCESSING_STEP_DEFINITIONS, IMAGE_PROCESSING_GUIDANCE)
MASK_PROCESSING_ITEMS = _definition_items(MASK_PROCESSING_STEP_DEFINITIONS, MASK_PROCESSING_GUIDANCE)
MEASUREMENT_ITEMS = _measurement_items()


HELP_REFERENCES = {
    "Input & ND2": [
        ("ND2 reader", "https://tlambert03.github.io/nd2/"),
        ("OME-TIFF", "https://ome-model.readthedocs.io/en/stable/ome-tiff/"),
    ],
    "Channels & Masks": [("Trainable Weka Segmentation", "https://imagej.net/plugins/tws/")],
    "Processing": [
        ("ImageJ processing", "https://imagej.net/ij/docs/guide/146-29.html"),
        ("ImageJ display and LUTs", "https://imagej.net/ij/docs/guide/146-28.html"),
    ],
    "Mask Processing": [
        ("ImageJ binary operations", "https://imagej.net/ij/docs/guide/146-29.html"),
        ("ImageJ Translate", "https://imagej.net/ij/docs/menus/image.html"),
        ("AnalyzeSkeleton", "https://imagej.net/plugins/analyze-skeleton/"),
    ],
    "Trainable Weka Segmentation": [
        ("Training and features", "https://imagej.net/plugins/tws/"),
        ("ImageScience installation", "https://imagej.net/libs/imagescience"),
    ],
    "Cellpose": [
        ("Cellpose inference API", "https://cellpose.readthedocs.io/en/latest/api.html"),
        ("Cellpose-SAM settings", "https://cellpose.readthedocs.io/en/latest/settings.html"),
        ("Models", "https://cellpose.readthedocs.io/en/latest/models.html"),
    ],
    "Measurements": [
        ("Fiji ROI creation", "https://imagej.net/ij/docs/menus/edit.html"),
        ("ImageJ measurements", "https://imagej.net/ij/docs/guide/146-30.html"),
        ("Cell geometry", "https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops"),
    ],
    "Image Preview Tools": [("ImageJ display", "https://imagej.net/ij/docs/guide/146-28.html")],
}


HELP_PAGES = [
    (
        "Introduction",
        """
        <p>Cellonaut measures structures and individual cells in microscopy images. Use a classifier trained with
        Fiji's Trainable Weka Segmentation to create masks, and use Cellpose to identify individual cells.</p>
        <p>To get started, choose input and output folders, check the detected channels, configure masks and
        measurements, then run <b>Check Setup</b> and <b>Preview One Sample</b>. When the preview looks right,
        select <b>Run Pipeline</b> for the full dataset.</p>
        <p>For a walkthrough using bundled example files without processing your dataset, open
        <b>Tutorial</b> in the sidebar, below <b>Settings</b>. It restores your setup when you leave.</p>
        """,
    ),
    (
        "Input & ND2",
        """
        <p>Choose a TIFF input folder and an output folder <b>outside it</b>, so results are not mistaken for new samples.
        Cellonaut detects channels in TIFF stacks and supported folder layouts. Check the detected channel names before running.</p>
        <table width="100%" cellspacing="0" cellpadding="12"><tr><td class="help-example">
        <h3>Supported folder layouts</h3>
        <p>Matching filenames identify the same sample across channel folders.</p>
        <ul>
            <li><b>TIFF stack:</b> <code>Input/sample.tif</code></li>
            <li><b>Channel folders:</b> <code>Input/GFP/sample.tif</code> and <code>Input/DAPI/sample.tif</code></li>
            <li><b>Sample folders:</b> <code>Input/sample/GFP/image.tif</code> and <code>Input/sample/DAPI/image.tif</code></li>
            <li><b>Group folders:</b> <code>Input/group/sample/GFP/image.tif</code></li>
        </ul>
        </td></tr></table>
        <h3>ND2 files</h3>
        <p>Use <b>Import ND2 files...</b> to convert ND2 files to TIFF, then select the converted folder as input.
        Check the channel mapping; leave a TIFF channel name blank to skip that channel. For Z stacks, choose
        <b>Max projection</b> or a <b>Single Z slice</b> numbered from 1. The chosen slice must exist in every file.</p>
        <p><b>Check before converting:</b> Multiple timepoints and XY positions use the first one. RGB data uses
        only its first component. Conversion writes lossless OME-TIFF files and does not overwrite existing output files. Existing names receive
        a numbered suffix: sample.ome.tif, sample_2.ome.tif, sample_3.ome.tif. Repeating a conversion creates
        additional copies; it does not resume by skipping completed files.</p>
        """,
    ),
    (
        "Channels & Masks",
        """
        <p>A <b>channel</b> is an input image. A <b>mask</b> marks the region to measure. You can create a mask
        from one channel and use it to measure another. For example, create a nucleus mask from DAPI and measure
        GFP inside it.</p>
        <p>Check the channels under Setup. In <b>Masks</b>, select the source image and choose <b>Add mask</b>.
        Use a trained Weka <code>.model</code>, or select <b>Combined masks</b> to build a mask from existing masks.
        Then turn ON the channel/mask pairs you want in <b>Measurements</b>.</p>
        <p>Combined masks support <b>OR</b> (all selected areas), <b>AND</b> (shared areas), and <b>XOR</b>
        (areas covered by an odd number of masks). A combined mask can also be used to build another combined mask.</p>
        <p>A Weka mask measures its selected regions together, even if they are disconnected. To identify and measure
        individual cells separately, enable <b>Cellpose</b> for the measured channel; it does not require a Weka mask.</p>
        """,
    ),
    (
        "Processing",
        f"""
        <p>Processing steps run in row order for the images or masks where you set a value:</p>
        <ul>
            <li><b>Before mask creation</b> changes the image sent to Weka.</li>
            <li><b>After mask creation</b> changes the resulting masks.</li>
            <li><b>Before measurement</b> changes image values used for measurements without changing the masks.</li>
        </ul>
        <p>Leave a value blank, or turn a value-free step OFF, to skip it for that source. If you preprocess an image
        before Weka, train its classifier on images processed the same way.</p>
        <p><b>Subtract Background</b> is always last in Before measurement. Each radius produces a separate corrected
        measurement result; the radii are not applied to one another. Before mask creation, multiple radii instead
        run in sequence.</p>
        <p><b>Enhance contrast</b> without Normalize or Equalize changes only the display range. Use <b>Apply LUT</b>
        if you need that display change applied to pixel values.</p>
        <p>Click the small settings icon inside a numeric field to open <b>Fiji command options</b>,
        including Light background, Normalize pixels, and Equalize histogram where available.</p>
        <h3>Step reference</h3>
        <ul>{IMAGE_PROCESSING_ITEMS}</ul>
        <p>For operations in <b>After mask creation</b>, choose <b>Mask Processing</b> from the
        <b>Help topic</b> menu for the full mask-step reference.</p>
        <p>Preview a sample before running the dataset; processing montages show intermediate results.</p>
        """,
    ),
    (
        "Mask Processing",
        f"""
        <p>Mask Processing runs Fiji operations on binary masks after creation and before measurement. Steps run in
        row order; each uses the previous mask result.</p>
        <p>The switch on a row enables or disables that step for every mask. Settings in each mask column apply only
        to that mask. Use <b>Preview One Sample</b> to check the result.</p>
        <p><b>Analyze skeleton</b> measures a copy without changing the mask. <b>Skeletonize</b> changes the mask
        used by later steps and measurements.</p>
        <h3>Step reference</h3>
        <ul>{MASK_PROCESSING_ITEMS}</ul>
        """,
    ),
    (
        "Trainable Weka Segmentation",
        f"""
        <p>Trainable Weka Segmentation learns from examples painted in Fiji. Open a representative image in the
        bundled Fiji, choose <b>Plugins &gt; Segmentation &gt; Trainable Weka Segmentation</b>, paint examples for
        each class, train, and check the classifier on other images. Save it as a <code>.model</code> file.</p>
        <p>In Cellonaut's <b>Masks</b> panel, select the source image, saved model, class number(s), and
        probability-map threshold method. Class numbers follow the order saved in Fiji, starting at <code>1</code>;
        they are not names. Enter <code>1</code>, <code>1,3</code>, or <code>1-3</code>. Each selected class
        produces a separate mask.</p>
        <p>Thresholding turns each class's probability map into a binary mask. Higher-probability pixels become
        foreground; the <b>Default</b> method is Fiji's modified IsoData. Train on images processed the same way
        as Cellonaut's <b>Before mask creation</b> stage, then preview a sample.</p>
        <h3>ImageScience</h3>
        <p>{IMAGESCIENCE_HELP}</p>
        <p>In Settings, use <b>Open Fiji Folder</b>, launch Fiji, then choose
        <b>Help &gt; Update... &gt; Manage update sites</b>. Enable <b>ImageScience</b>, apply changes,
        restart Fiji to finish the update, then restart Cellonaut.</p>
        """,
    ),
    (
        "Cellpose",
        f"""
        <p>Cellpose identifies individual cells and assigns each an ID. Enable it on the channel you want to measure,
        then choose the image Cellpose should use to find cell boundaries. These can be different channels. Preview
        one sample before adjusting settings.</p>
        <h3>Settings reference</h3>
        <ul>
            <li><b>Model</b>: <code>cpsam</code> is the default; <code>cpsam_v2</code> is also bundled.</li>
            <li><b>Custom model</b>: use the folder button to select a compatible Cellpose 4 model file. It replaces
            the selected built-in model until you clear it with the delete button.</li>
            <li><b>Diameter</b>: {CELLPOSE_DIAMETER_HELP}</li>
            <li><b>Minimum cell area</b>: remove detections smaller than this area in pixels squared.</li>
            <li><b>Cell probability threshold</b>: lower to include more pixels and find more or larger cells; higher is more selective.</li>
            <li><b>Flow threshold</b>: lower positive values reject more inconsistent shapes; higher keeps more. Zero disables this check.</li>
            <li><b>Remove border cells</b>: exclude partial cells touching the image edge.</li>
        </ul>
        <p>The cell probability threshold is a score, not a percentage. Cellonaut sends the selected source as
        a 2D grayscale image.</p>
        <h3>Acceleration</h3>
        <p>On Windows, the installer's <b>Automatic</b> choice uses CUDA when a compatible NVIDIA GPU and driver
        are available, otherwise CPU; <b>CPU only</b> disables GPU use. <b>Check Setup</b> shows the selected backend.</p>
        <h3>How the final Cellpose masks are produced</h3>
        <p>Cellonaut runs Cellpose in Python. Use the run's <code>Results/Logs/RunSummary.json</code>
        and log to identify the source channel, model, backend, and settings.</p>
        <ol>
            <li>Read the selected Cellpose source as the same 2D grayscale plane or Z projection used by
            the run. The Weka Before mask creation recipe is not a Cellpose preprocessing recipe.</li>
            <li>Convert to float32. Integer images are divided by the maximum representable value of their
            dtype; floating images are divided by their maximum when positive. Replace NaN with 0,
            positive infinity with 1, and negative infinity with 0.</li>
            <li>Run the recorded Cellpose model with the saved diameter, minimum area, cell probability
            threshold, and flow threshold. Cellpose's own default normalization remains enabled.
            Reproduction requires the same Cellpose version, model weights, input, and settings;
            different compute backends can produce different boundaries.</li>
            <li>Remove labels with fewer pixels than Minimum cell area and renumber the retained labels
            consecutively. If Remove border cells is ON, remove every label touching any image edge and
            renumber again. Do not apply Fiji Analyze Particles to a binary union of cells: touching
            cells would lose their separate IDs.</li>
            <li>If the saved run has nonzero cell-mask adjustments, Cellonaut applies them after cleanup:
            translate, reject undersized labels, fill small holes, then grow or shrink each label.
            Growth uses a circular pixel-radius footprint; higher cell IDs take shared pixels where labels overlap.
            Temporary Image Preview Tools
            adjustments and cell-group colors are display edits, not segmentation steps.</li>
        </ol>
        <p>When existing Cellpose labels are reused, inference, minimum-area cleanup, and border removal
        are skipped; the log identifies reuse. Reused labels retain their saved adjustments; current cell-mask adjustments are not applied again. Regenerate the cell masks to apply different adjustments.</p>
        <h3>Check saved Cellpose labels in Fiji</h3>
        <ol>
            <li>Open the sample's label TIFF under <code>Results/Cells/TIFF Labels</code> and its source
            image. Keep label values intact: zero is background and each positive value is a cell ID.</li>
            <li>For a cell ID, use <b>Image &gt; Adjust &gt; Threshold</b> and set both limits to that ID.
            Choose <b>Edit &gt; Selection &gt; Create Selection</b>, then add the selection to
            <b>Analyze &gt; Tools &gt; ROI Manager</b>. Repeat for the IDs to inspect.</li>
            <li>In <code>_cell_measurements.csv</code>, <code>Mean</code>, <code>Min</code>, <code>Max</code>,
            <code>Median</code>, <code>RawIntDen</code>, and <code>IntDen</code> describe the Cellpose source image
            without Before measurement processing. Selected columns prefixed with <code>Cell</code>, such as
            <code>CellMean</code> and <code>CellIntDen</code>, describe the measured channel after its
            Before measurement transforms, before background subtraction. These sources can differ.
            For cell-and-mask signal tables, use the measured channel and the matching raw or
            background-corrected image. Follow the Fiji instructions at the end of
            <b>Measurements</b> for the matching intensity source. Compare cell IDs with the tables under
            <code>Results/CSV Data/Cell Measurements</code>. Area and pixel-intensity sums can be checked
            this way; Fiji and Cellonaut's scikit-image cell perimeter/shape estimators may differ.</li>
        </ol>

        """,
    ),
    (
        "Measurements",
        f"""
        <p>In the measurement matrix, <b>rows are image channels to measure</b> and <b>columns are masks</b>.
        Turn ON an intersection to measure that channel inside that mask. Enable Cellpose for a channel to measure
        separately identified cells; you can use both routes together.</p>
        <p><b>Measurement settings</b> controls which metrics are included. Necessary result tables and review overlays are saved automatically.
        Cell-and-mask measurements require Cellpose
        on the measured channel and an assigned measurement mask because they describe a mask region inside each cell.</p>
        <p>Lengths and areas remain in pixels and pixels squared, even when the image contains physical calibration.
        <b>Integrated density</b> is the sum of pixel intensities in the region (ImageJ's RawIntDen).</p>
        <p>Saved results include a table per measured channel, a combined dataset table, detailed cell tables,
        and a review overlay. Derived fractions and ratios are left to analysis after export.</p>
        <p>Mask-inside-cell standard deviation uses the sample formula (dividing by pixel count minus one)
        and is blank for fewer than two pixels. Whole-mask statistics use Fiji's measurement implementation.
        Skewness describes intensity-distribution asymmetry; kurtosis describes its tail weight.
        Use the recorded Fiji version when reproducing these statistics.</p>
        <h3>Measurement reference</h3>
        <ul>{MEASUREMENT_ITEMS}</ul>
        <h3>Verify measurements manually in Fiji</h3>
        <p>Use the same sample, measured channel, mask, and settings recorded in
        <code>Results/Logs/RunSummary.json</code>. Work on duplicates. Preview-export PNGs and color TIFFs
        are rendered display images; use the original grayscale channel and binary masks for verification.</p>
        <ol>
            <li>Open the measured channel in Fiji. For a stack, select the same channel and single Z slice,
            or use <b>Image &gt; Stacks &gt; Z Project...</b> with Max Intensity and the same slices as the
            run. Match the run's timepoint and position.</li>
            <li>Use <b>Analyze &gt; Set Scale...</b>: set Distance in pixels and Known distance to 1,
            Pixel aspect ratio to 1, and Unit of length to pixel; leave Global unchecked. Remove any
            intensity calibration with <b>Analyze &gt; Calibrate...</b>, Function None. Cellonaut measures
            uncalibrated pixel values and pixel areas.</li>
            <li>Apply the channel's <b>Before measurement</b> steps in recorded order using the Fiji
            commands in the Processing reference. Match all options, including conversion scaling and
            contrast normalization. Keep a copy before Subtract Background.</li>
            <li>Open the matching mask under <code>Results/Masks/BinaryMasks</code>. This is the final
            measurement mask. If starting from <code>Results/Masks/MaskImages</code> instead, apply the
            saved After mask creation recipe first; for a combined mask, reproduce its source-mask
            operations and combination before its own recipe. Do not process a final mask twice.</li>
            <li>On the final binary mask, use <b>Image &gt; Adjust &gt; Threshold</b> to select foreground
            values greater than zero. Choose <b>Edit &gt; Selection &gt; Create Selection</b>, then add
            it to <b>Analyze &gt; Tools &gt; ROI Manager</b>. Keep disconnected regions in this one
            composite ROI: Cellonaut measures their union, not a separate row for each particle.</li>
            <li>Activate the prepared measured-channel image and select that ROI in ROI Manager.
            In <b>Analyze &gt; Set Measurements...</b>, select the same metrics as Cellonaut, set
            Redirect to None, and leave Limit to threshold unchecked. Use sufficient decimal places
            for comparison, then choose <b>Analyze &gt; Measure</b>. Compare Area, Mean, and RawIntDen
            with the matching channel/mask columns in <code>Results/CSV Data/Measurements.csv</code>.</li>
            <li>For each background radius, duplicate the image saved before subtraction and run
            <b>Process &gt; Subtract Background...</b> with that radius and the recorded Light background,
            Sliding paraboloid, and Disable smoothing options. Apply the same ROI and measure again.
            Compare RawIntDen with the corresponding background-corrected result. Start each radius
            from the same pre-subtraction image; do not subtract successive radii from one another.</li>
        </ol>
        <p>For mask-inside-cell measurements, make a ROI for one Cellpose label as described in <b>Cellpose</b>,
        then intersect it with the mask ROI using ROI Manager's <b>More &gt; AND</b>. Measure that intersection
        on the measured channel. An empty mask-cell intersection has area and integrated density 0; intensity
        statistics are blank because there are no pixels to describe.</p>

        """,
    ),
    (
        "Image Preview Tools",
        """
        <h3>Inspect images</h3>
        <p>Click a file once to open it in Image Preview, or run <b>Preview One Sample</b>. <b>Recent images</b>
        reopens a previously viewed file. Image Preview opens TIFF, ND2, PNG, and JPG files. Use the result menu
        to switch between overlays, Weka probability maps, Weka threshold masks, final binary masks, processing
        montages, and snapshots. <b>Layers</b> controls visibility, order, color, opacity, and <b>Composite</b>
        blending; these display choices do not change measurements.</p>
        <h3>Mask Adjustments</h3>
        <p>Open a generated overlay with linked Weka mask metadata. A standalone binary TIFF may not provide
        a selectable mask target.</p>
        <p>Select a Weka-mask layer (Cellpose labels are not available here), set X/Y shifts or cleanup values, and choose <b>Preview adjustment</b> to compare the
        result. These changes are temporary and do not change the pipeline. <b>Reset adjustments</b> clears only the selected mask; other masks retain their adjustments.</p>
        <p>Positive X moves right and positive Y moves down. Adjustments run in this order: shift, fill enclosed
        holes up to the chosen area, expand or shrink using a circular pixel-radius footprint, then remove
        regions below Minimum area. Minimum area treats diagonally touching foreground pixels as connected.
        These preview operations use SciPy and differ from Fiji's Binary Erode/Dilate commands.</p>
        <h3>Cell Groups</h3>
        <p>A compatible Cellpose result is required. Set cell or target-mask conditions with <b>At least</b> and
        <b>At most</b>; blank means no limit. A group with no conditions is inactive: it highlights no cells and excludes
        none from CSV exports. Groups can overlap. <b>Update groups</b> recalculates membership
        without rerunning segmentation. <b>Match either category</b> means all cell conditions OR all target-mask
        conditions, not any single condition. Categories with no conditions are ignored. A missing or nonnumeric value fails an active
        condition in its category; Match either category can still match through the other category.</p>
        <p>For fraction conditions, <code>25%</code> and <code>0.25</code> mean the same thing.
        Include the percent sign: <code>25</code> means twenty-five, not 25 percent.</p>
        <p>Choose <b>Target mask for cell groups</b> when target-mask conditions use a mask. Whole-cell intensity
        uses the Cellpose source channel; <b>Measure target intensity from</b> selects the target-mask intensity
        source. <b>Exclude this group from CSV files</b> affects filtered exports, not original measurements.</p>
        <p>Use <b>Export filtered CSV for this sample</b> or <b>Export filtered CSV for all samples in the pipeline</b>.
        Derived files are kept under <code>Results/Image Preview Tools/Cell Groups</code>, separate from original
        measurements. The single-sample export writes a filtered table and a filter report. The all-samples
        export also writes combined tables and, when cells match groups, <code>Cell_Group_Membership.csv</code>
        with one cell/group assignment per row. It uses the open image's Results folder when linked,
        otherwise the configured output folder. Only compatible saved results are processed.</p>
        <h3>Save the edited image</h3>
        <p>Use <b>Save preview image</b> in the vertical toolbar to choose a TIFF or PNG destination.
        It saves the full current page at image resolution, including visible layers, colors, opacity,
        mask adjustments, and cell-group layers. Zoom and the snapshot square do not crop this export.
        Hidden layers stay hidden. Both formats save a flattened 8-bit RGB display image, not separate
        editable layers or the original measurement intensities. For stacks, only the current page is saved.</p>
        <h3>Snapshots</h3>
        <p><b>Snapshot</b> captures a crop of the current view. Choose <b>Layer montage PNG</b> for the same crop
        across source layers, including hidden layers, with each individual tile at full opacity.
        Cell-group overlays appear only in the final composite tile, not as separate group tiles. Files go to <code>Results/Image Preview Tools/Snapshots</code>.</p>
        """,
    ),
    (
        "Files & Results",
        """
        <p>Each run or preview has a numbered <code>run_N</code> or <code>preview_N</code> folder. Open its
        <code>Results</code> folder.</p>
        <h3>Start here</h3>
        <ul>
            <li><code>CSV Data/Measurements.csv</code>: the main sample table; measurement columns include units.</li>
            <li><code>Overlays/TIFF Overlays/*_combined_overlay.tif</code>: check which regions were measured.</li>
            <li><code>Logs/RunSummary.txt</code>: check the run's settings and outcome.</li>
        </ul>
        <h3>Other results</h3>
        <p><code>Measurements_By_Metric.csv</code> puts measurements in rows;
        <code>Measurement_Summary.csv</code> summarizes measurements across samples. Both are in <b>CSV Data</b>.
        For each measurement and group, <b>N</b> counts samples with a numeric value; missing values are omitted.
        <b>SD</b> is the sample standard deviation and is blank when fewer than two values are available.</p>
        <p>Per-cell summary means average the individual cell means, giving each cell equal weight regardless
        of its area. They are not the average of all pooled cell pixels.</p>
        <p>Cell tables include summary rows such as <code>ALL_CELLS_SUM</code> and
        <code>ALL_CELLS_MEAN</code>; some filtered tables also contain <code>FILTERED_CELLS_SUM</code>
        and <code>FILTERED_CELLS_MEAN</code>. These are summaries, not individual cells.
        Use only positive numeric cell IDs when counting cells or analysing individual-cell records.</p>
        <p>When Analyze skeleton is enabled, <code>CSV Data/Skeleton_Measurements.csv</code> contains
        <code>SkeletonCount</code> (separate skeletons), <code>BranchCount</code> (branches),
        <code>EndpointCount</code> (terminal points), and <code>JunctionCount</code> (merged junctions).
        Counts describe the mask at that recipe step, before any later mask processing.</p>
        <p><b>Cells</b> contains cell labels, outlines, and review files; cell measurement tables are in
        <code>CSV Data/Cell Measurements</code>. <b>Masks</b> contains probability maps and masks, and <b>Processing Montages</b> shows intermediate
        stages. <b>Image Preview Tools</b> contains derived exports separately from the original results.</p>
        <p>In <b>Files</b>, use <b>Open location</b> to show a folder in your system file explorer.
        <b>Compare Runs</b> compares settings and summaries from two runs or previews; it does not match
        individual cells, whose IDs may change.</p>
        <p>In Compare Runs, <b>Difference</b> is current minus baseline. Percentage <b>Change</b> is
        100 times that difference divided by the absolute baseline. <b>N/A</b> means the baseline is zero
        or a value is missing (Added/Removed). <b>Minimum change</b> filters by absolute percentage
        and keeps N/A rows visible.</p>
        <p><b>Export CSV...</b> on Measurements exports only rows remaining after search, Changed only,
        and Minimum change filtering. Summary counts describe the complete comparison. The Settings CSV
        exports all differing settings. Neither export contains images or individual-cell matches.</p>
        """,
    ),
    (
        "Presets & Settings",
        """
        <h3>Presets</h3>
        <p>Presets save pipeline choices, but not input/output folders or ND2 conversion choices. An asterisk
        marks unsaved changes; <b>Revert</b> discards them. <b>Save</b> updates a custom preset. <b>Default</b>
        has a lock icon and cannot be overwritten: saving changes made from it opens <b>Save As</b>.
        <b>Delete</b> is available only for custom presets.</p>
        <p>Each opened pipeline step and the <b>Image Preview Tools</b> panel has its own <b>Notes</b> area beneath
        the panel description. Notes are plain text, limited to 20 lines, and saved with the preset. The editor
        shows three lines normally; use the small down arrow to expand it to 20 lines and the up arrow to reduce it.
        Notes document the workflow only and do not change analysis behavior.</p>
        <h3>Sharing</h3>
        <p>Use <b>Export Selected Preset</b> and <b>Import Preset</b>. Shared paths retain only the
        final file or folder name, so the recipient must select local folders and model files again.
        This removes parent directories from known path fields; filenames, panel notes, and other text remain.
        Review those details before sharing. Export includes the current editor settings, including
        unsaved edits, rather than only the last saved version of the selected preset.</p>
        <h3>Reusing masks</h3>
        <p>Turn on <b>Reuse existing masks</b> to use masks from the current output or a chosen previous output.
        Run <b>Check existing masks</b> to check that files and image sizes match. Regenerate masks after changing
        detection settings. Matching filenames and dimensions do not certify that the saved masks were
        generated with the current classifier, thresholds, or preprocessing. Missing Weka classes or Cellpose labels can be regenerated with the current
        settings, so one run can contain both reused and newly generated masks. Check the Log to see which
        masks were reused and which were generated.</p>
        <h3>Settings</h3>
        <p>Change the theme and check installed tool status. <b>Open Cellonaut Data Folder</b> shows local settings,
        presets, and crash logs; <b>Open Fiji Folder</b> opens the selected Fiji location. Cellonaut checks
        destination free space before each task. Critically low destinations are blocked; below-recommended space
        requires confirmation.</p>
        """,
    ),
    (
        "Troubleshooting",
        """
        <p>Start with <b>Check Setup</b>, then <b>Preview One Sample</b>. Read the <b>Log</b> tab for the current task.</p>
        <ul>
            <li><b>No samples found</b>: check the TIFF folder layout, or use <b>Import ND2 files...</b> for ND2 data.</li>
            <li><b>Fiji cannot start</b>: check that the complete offline package was installed. On Windows, keep the
            installer EXE and every BIN file together during installation.</li>
            <li><b>Weka fails</b>: check the <code>.model</code> path, source channel, and class number. Install
            ImageScience only if the classifier requires it.</li>
            <li><b>Cellpose fails or boundaries look wrong</b>: check model and backend status in <b>Check Setup</b>;
            then preview the source channel, diameter, thresholds, minimum area, and border-cell option.</li>
            <li><b>Reused mask rejected</b>: check the sample name, mask filename, and image dimensions. Regenerate
            after changing detection settings.</li>
            <li><b>Overlay or cell groups look wrong</b>: check channel order, mask source, Weka class, and
            adjustments. Keep the overlay, cell table, and cell labels in the same result folder.</li>
        </ul>
        <h3>Incomplete runs</h3>
        <p><b>Completed with errors</b> means processing finished with failures; some results may be available.
        Skipped targets are counted separately and can also occur in a completed run.
        A target is a configured measurement task within a sample; a sample can have several.
        Review the successful, failed, and skipped target counts and the Log before using the results.</p>
        <p>When cancelling, wait for the worker to finish stopping. Saved results may be incomplete;
        cancellation does not make the output a completed run.</p>
        <h3>Saved diagnostics</h3>
        <p><code>RunSummary.txt</code> and <code>RunState.json</code> are under each run or preview
        folder in <code>Results/Logs</code>. <code>setup_check_report.txt</code> is saved directly in
        the configured output folder when Check Setup completes and that folder is writable.</p>
        """,
    ),
    (
        "About",
        f"""
        <h3>Cellonaut {__version__}</h3>
        <p>Copyright (C) 2026 Adam Jamalov. Cellonaut is free software under
        <b>GPL-3.0-or-later</b> and is provided without warranty.</p>
        <p>Source: <a href="https://github.com/AJamalov/Cellonaut">github.com/AJamalov/Cellonaut</a>.</p>
        <h3>Citation</h3>
        <p>Use <code>CITATION.cff</code> for Cellonaut. <code>CITATIONS.md</code> contains references for Cellpose,
        Fiji, ImageJ, PyImageJ, and Trainable Weka Segmentation.
        Cite the tools used in the reported workflow and record the Cellonaut version from <code>RunSummary.txt</code>.</p>
        <h3>Licenses and source availability</h3>
        <p>DINO-based models and the DINOv3 runtime are not distributed with Cellonaut.</p>
        <p>See <code>LICENSE</code>, <code>THIRD_PARTY_NOTICES.md</code>, <code>THIRD_PARTY_LICENSES</code>, and
        <code>SOURCE_AVAILABILITY.md</code>.</p>
        <h3>Official documentation</h3>
        <ul>
            <li><a href="https://imagej.net/">ImageJ and Fiji</a></li>
            <li><a href="https://imagej.net/plugins/tws/">Trainable Weka Segmentation</a></li>
            <li><a href="https://cellpose.readthedocs.io/">Cellpose</a></li>
        </ul>
        """,
    ),
]

HELP_PAGES = [
    (
        title,
        body
        + (
            "<p>References: "
            + " · ".join(
                f'<a href="{escape(url, quote=True)}">{escape(label)}</a>' for label, url in HELP_REFERENCES[title]
            )
            + "</p>"
            if title in HELP_REFERENCES
            else ""
        ),
    )
    for title, body in HELP_PAGES
]
