#!/bin/bash

# mkdir beam_2_5_scan
# mkdir beam_5_scan

echo "Building Beam 2.5 scan"

cp ../../174658/scan_beam_and_RF/beam_2_5_scan/retrieve_files_for_scan.py beam_2_5_scan/
cp ../../174658/scan_beam_and_RF/beam_2_5_scan/rf_zoom_pwrscan_notebook.ipynb beam_2_5_scan/
cp ../../174658/scan_beam_and_RF/beam_2_5_scan/getInputFileDictionary.py beam_2_5_scan/

echo "Building Beam 5 scan"

cp ../../174658/scan_beam_and_RF/beam_5_scan/retrieve_files_for_scan.py beam_5_scan/
cp ../../174658/scan_beam_and_RF/beam_5_scan/rf_zoom_pwrscan_notebook.ipynb beam_5_scan/
cp ../../174658/scan_beam_and_RF/beam_5_scan/getInputFileDictionary.py beam_5_scan/
