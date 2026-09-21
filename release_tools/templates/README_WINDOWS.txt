Cellonaut {{APP_VERSION}} Windows {{PROFILE}} build

How to run
----------
If you installed Cellonaut with the installer, launch it from the Start Menu or desktop shortcut.

If you are testing the raw dist folder, run:

   Cellonaut.exe

Offline runtime
---------------
This installer includes Fiji, Java, Bio-Formats/ND2, Trainable Weka Segmentation,
and both supported built-in Cellpose models. No internet connection is required
for installation or analysis. Settings > Runtime status reports the bundled
component versions. Cellonaut selects the packaged runtime automatically; if it
is reported missing or incomplete, reinstall using the EXE and all BIN slices.

ImageScience for Weka
------------------------------
ImageScience is not included. Some Weka classifiers require it for derivative,
edge, Hessian, Laplacian, or structure-tensor features. Close Cellonaut, open
Fiji (optional plugins) from the Cellonaut Start-menu folder, then use Help >
Update... > Manage update sites to enable ImageScience. Apply the changes,
close Fiji, and restart Cellonaut.

Cellpose models
---------------
The included models are cpsam and cpsam_v2. cpsam is the default for new
configurations. Selecting either built-in model does not download weights during
analysis. DINO-based models and the DINOv3 runtime are not distributed.

Cellpose backend
----------------
{{CELLPOSE_BACKEND}}

Troubleshooting
---------------
Run Check Setup inside Cellonaut to verify Fiji, classifiers, samples, and the Cellpose backend.
Keep the Log tab output and Results\Logs files if a run fails.
