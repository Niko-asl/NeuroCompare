# 3D-MRI-ML-Framework-DEMON-

</br>
<h2> End to end MRI analysis pipeline with 3 networks (currently) </h2> </br>
1. 3D CNN </br>
2. 3D Vision Transformer </br>
3. Hybrid CNN-ViT </br>

<h2> Preprocessing </h2> </br>
This framework uses an integrated 4 step preprocessing pipeline </br>(code in preprocessing directory, preprocess_data.py script in scripts dir) </br>
1. DICOM to NIFTI conversion with dcm2niix </br>
2. Skull Stripping using HD-BET </br>
3. Spatial normalization with FSL </br>
4. Intensity normalization using percentile clipping </br>

<h2> Augmentation </h2> </br>
This framework provides for the option to use a wide array of augmentation options, </br>
See below for a comprehensive list

<h2> Framework </h2> </br>
Once the data is preprocessed, training can be tailored from the command line. </br>
Model size, augmentation, loss function, and many or most hyperparameters can be configured on a per run basis <strong> without chainging any code!! </strong> </br>

<h2> Commands and Command line use </h2> </br>

<strong> Preprocessing: </strong> </br>
--ad_dir <em> Directory containing raw dicom scans of AD patients </em> </br>
-- control_dir <em> Directory containing raw dicom scans of controls </em> </br>
