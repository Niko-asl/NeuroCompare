# 3D-MRI-ML-Framework-DEMON-

</br>

This is an early beta implementation of the 3D end-to-end model comparison framework for the DEMON group. </br> </br>

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
This leads to a lot of arguments being appended, but is still more rapidly flexible than manually recoding stuff. </br>
N.B, The hybrid model currently has a hardcoded CNN frontend with only the ViT backend being malleable. </br>
A future refactor will add the ability to modify the CNN from command line but that utility is not currently implemented. </br>
If modification to the CNN portion is essential, it is always possible to recode models/hybrid_model.py


<h2> Commands and Command line use </h2> </br>

<strong> Preprocessing: </strong> </br>
</br>
--ad_dir : <em> Directory containing raw dicom scans of AD patients </em> </br>
--control_dir : <em> Directory containing raw dicom scans of controls </em> </br>
--output_dir : <em> Directory for output (should default to /home/usr) </em> </br>

<strong> Training </strong> </br>
</br>
DATA CONFIGURATION: <br/>
</br>
--data : <em> directory containing preprocessed data </em> </br>
--base_dir : <em> specifies root directory if using relative paths (optional, sometimes necessary), default none </em> </br>
--batch_size : <em> batch size, default 4 </em> </br>
--balanced_sampling : <em> Use balanced sampling for training batches (recommended), default False </em> </br>
</br>
MODEL ARCHITECTURE: </br>
</br>
--model : <em>Which model architecture to train (currently supports CNN, ViT, or Hybrid), default cnn </em> </br>
CNN ARCHITECTURE OPTIONS (N.B, THE DEFAULT CNN IS A RESNET):
--cnn_layers : <em> Depth of CNN network, 2-6 layers currently supported, default 3  </em> </br>
--cnn_base_filters : <em> number of filters in the first convolutional layer of the CNN (doubles every layer after), default 32 </em> </br>
--cnn_dropout : <em> dropout rate for CNN, default 0.5 (specification highly recommended) </em> </br>
</br>
ViT ARCHITECTURE OPTIONS: </br>
</br>
--vit_patch_size : <em> patch size for the vision transformer, default 16 </em> </br>
--vit_depth : <em> number of transformer blocks, default 12 </em> </br>
--vit_heads : <em> number of attention heads, default 12 </em> </br>
--vit_embed_dim : <em> embdedding dimension, default 768 </em> </br>
--vit_mlp_ratio : <em> multilayer perceptron ratio for ViT, default 4 </em> </br>
--vit_dropout : <em> dropout for ViT </em> </br>
</br>
HYBRID MODEL ARCHITECTURE OPTIONS: (N.B ONLY VIT PORTION IS MODIFIABLE) </br>
</br>
--hybrid_feature_dim : <em> feature dimension for hybrid model, default 256 </em> </br>
--hybrid_transformer_dim : <em> dimension of transformer, default 256 </em> </br>
--hybrid_num_heads : <em> number of attention heads, default 8 </em> </br>
--hybrid_depth : <em> number of transformer blocks </em> </br>
--hybrid_mlp_ratio : <em>MLP ratio for the hybrid model </em> </br>
--hybrid_dropout : <em>dropout rate for the ViT portion of the hybrid model, default 0.1 </em> </br>
</br>
TRAINING CONFIG
