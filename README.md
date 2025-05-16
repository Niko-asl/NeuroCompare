# 3D-MRI-ML-Framework-DEMON-

</br>
<h1> End to end MRI analysis pipeline with 3 networks (currently) </h1> </br>
1. 3D CNN </br>
2. 3D Vision Transformer </br>
3. Hybrid CNN-ViT </br>

<h1> Preprocessing <h1> </br>
This framework uses an integrated 4 step preprocessing pipeline (code in preprocessing directory, preprocess_data.py script in scripts dir) </br>
1. DICOM to NIFTI conversion with dcm2niix </br>
2. Skull Stripping using HD-BET </br>
3. Spatial normalization with FSL </br>
4. Intensity normalization using percentile clipping </br>
