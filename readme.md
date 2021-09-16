## xradia_scripts
Python scripts to work with Zeiss xradia microCT data. **xradia_data.py** is intended for more general usage, while other scripts are more specific to our applications.

**xradia_data.py**: API to interact with Zeiss xradia microCT data (.txm, .txrm). It can be used to retrieve both image data and most relevant metadata from the microCT files.
**export_image.py**: export volumetric data from Zeiss txm files.
**run_elastix.py**: use elastix to register volumes. See *miscs/Parameters* for the parameter file used.
**segment_lvlset.py**: use thresholding and levelset mothods to segment microCT data for volume measurement.
**visualize_segmentation.py**: use Python vtk library to render the 3d volumetric microCT data.