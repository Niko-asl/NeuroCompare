# 3D-MRI-ML-Framework-DEMON-

</br>

This is an early beta implementation of the 3D end-to-end model comparison framework for the DEMON group. </br> </br>

<h2> End to end MRI analysis pipeline with 3 networks (currently) </h2> </br>
1. 3D CNN </br>
2. 3D Vision Transformer </br>
3. Hybrid CNN-ViT </br>

<h2> Preprocessing </h2> </br>
This framework uses an integrated 4 step preprocessing pipeline </br>(code in preprocessing directory, preprocess_data.py script in scripts dir) </br> </br>
1. DICOM to NIFTI conversion with dcm2niix </br>
2. Skull Stripping using HD-BET </br>
3. Spatial normalization with FSL </br>
4. Intensity normalization using percentile clipping </br>

<h2> Augmentation </h2> </br>
This framework provides for the option to use a wide array of augmentation options, </br>
See below for a comprehensive list

<h2> Framework </h2> </br>
Once the data is preprocessed, training can be tailored from the command line. </br>
Model size, augmentation, loss function, and many or most hyperparameters can be configured on a per run basis <strong> without changing any code!! </strong> </br>
This leads to a lot of arguments being appended, but is still more rapidly flexible than manually recoding everything. </br>
Additionally, this maintains a consistent implementation of every modification and counteracts methodological variablility </br>
N.B, The hybrid model currently has a hardcoded CNN frontend with only the ViT backend being malleable. </br>
A future refactor will add the ability to modify the CNN from command line but that utility is not currently implemented. </br>
If modification to the CNN portion is essential, it is always possible to recode models/hybrid_model.py


<h2> Commands and Command line use </h2> </br>

<strong> Preprocessing - preprocess_data.py: </strong> </br>
</br>
--ad_dir : <em> Directory containing raw dicom scans of AD patients </em> </br>
--control_dir : <em> Directory containing raw dicom scans of controls </em> </br>
--output_dir : <em> Directory for output (should default to /home/usr) </em> </br>

<strong> Training - train.py: </strong> </br>
</br>
DATA CONFIGURATION: </br>
</br>
--data : <em> directory containing preprocessed data </em> </br>
--base_dir : <em> specifies root directory if using relative paths (optional, sometimes necessary), default none </em> </br>
--batch_size : <em> batch size, default 4 </em> </br>
--balanced_sampling : <em> Use balanced sampling for training batches (recommended), default False </em> </br>
</br>
MODEL ARCHITECTURE: (N.B, THE DEFAULT CNN IS A RESNET)</br>
</br>
--model : <em>Which model architecture to train (currently supports CNN, ViT, or Hybrid), default cnn </em> </br>
</br>
CNN ARCHITECTURE OPTIONS:
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
TRAINING CONFIG: </br>
--epochs : <em>number of epochs to train, default 50 </em> </br>
--early_stopping : <em>number of epochs to wait for improvement in the monitor metric before early stopping, default 15 </em> </br>
--monitor : <em> metric to monitor for early stopping/best model (auc, f1, balanced_acc), default balanced_acc </em> </br>
--initial_freeze : <em> number of epochs for which to initially freeze the feature extractor, default 0 </em> </br>
--margin : <em>decision margin for predictions +/- to 0.5, default 0.0 </em> </br>
--grad_clip : <em> max norm for grad clipping, 0 disables gradient clipping, default 1 </em> </br>
--warmup_epochs : <em> number of epochs for learning rate warmup </em> </br>
--warmup_method : <em> if using lr warmup, which warmup scale method to use (linear or exponential), default linear </em> </br>
--monitor_gradients : <em> enable gradient monitoring during training, default false </em> </br>
--track_parameters : <em> track model params during training, default false </em> </br>
</br>
DATA AUGMENTATION: </br>

<strong> Augmentation options: </strong>
* Rotation
* Translation
* Scale
* Flip
* Deformation
* Noise
* Bias
* Ghosting
* Gamma
* Constrast
* Crop
* Cutout 
</br>
N.B: Increasing the number/probability of augments <strong> significantly </strong> increases runtime </br>
--no_augmentation : <em> disables ALL augmentation </em> </br>
--all_augmentation : <em> enables ALL augmentation (very highly not recommended, makes MRI soup) </em> </br>
--augment : <em> declare desired augmentation(s) from above list </em> </br>
--aug_prob : <em> probability of applying augmentation, default 0.5 </em> </br>
--rotation_angle : <em> max rotation angle for rotation augment in degrees, default 10 </em> </br>
--translation_voxels : <em> max translation in voxels for translation augment, default 5 </em> </br>
--scale_range : <em> range for random scaling, in format [min, max], default [0.90. 1.10] </em> </br>
--noise_percent : <em> standard deviation of noise as a percent of MRI max intensity, default 0.03 </em> </br>
--deformation_sigma : <em> elastic deformation sigma param, default: 3 </em> </br>
--deformation_magnitude : <em> elastic deformation magnitude parameter, default: 10 </em> </br>
--bias_coeff : <em> bias field coefficient, default 0.5 </em> </br>
--ghosting_max : <em> maximum number of ghosts allowed to haunt the MRI, default, 3 </em> </br>
--gamma_range : <em> range for random gamma adjustment, in format [min, max], default [0.70. 1.50] </em> </br>
--crop_size : <em> size of random crop (in voxels), default 80 </em> </br>
--cutout_size_range : <em> Size range for cutouts (in voxels, [min, max]), default [20, 40] </em> </br>
  </br>
  </br>
OPTIMIZATION PARAMETERS: </br>
--weight_decay : <em> weight decay for L2 regularization, default 0.0001 </em> </br>
--optimizer : <em> optimizer types, adam, adamw, or sgd, default adamw </em> </br>
--scheduler : <em> learning rate scheduler (plateau, cosine, step, none), default plateau </em> </br>
--plateau_factor : <em> factor by which to reduce learning rate on plateau, default 0.5 (half lr on plateau) </em> </br>
--plateau_patience : <em> patience for plateau scheduler (in epochs), default 7 </em> </br>
--cosine_min_lr : <em> minimum learning rate for cosine scheduler, default 1e-06 </em> </br>
--step_size : <em> step size for stepLR scheduler, default 0.1 </em> </br>
--step_gamma : <em> gamma for stepLR scheduler, default 0.1 </em> </br>
  </br> </br>
LOSS CONFIG: </br>
--loss : <em> loss function, currently supported are: cross entropy (ce), weighted cross entropy (weighted_ce) focal loss (focal), class balanced focal loss (cb_focal), asymmetric loss (asymmetric), margin loss (margin), binarcy crossentropy with logits (bce_logits), default weighted_ce </em> </br>
--class_weights : <em> use class weights with loss function to handle class imbalance, default False </em> </br>
--focal_gamma : <em> gamma param for focal loss, default 2.0 </em> </br>
--focal_alpha : <em> alpha param for focal loss, default 0.25 </em> </br>
--beta : <em> beta parameter for class-balanced weighing </em> </br>
--gamma_pos : <em> positive gamma parameter for asym loss, default 1 </em> </br>
--gamma_neg : <em> negative gamma parameter for asym loss, default 4 </em> </br>
--pos_margin : <em> margin for positive class predictions when using margin loss, default 0.5 </em> </br>
--neg_margin : <em> margin for negative class predictions, default 0.5 </em> </br>
--pos_weight : <em> weight for positive class predictions, default 2 </em> </br>
</br>
EVAL CONFIG:
</br>
--eval_batch_size : <em> batch size for evaluation process, default 4 </em> </br>
--decision_margin : <em> decision margin for evaluation predictions, default 0.0 </em> </br>
--save_predictions : <em> save individual decisions for downstream analysis, default False </em> </br>
--export_features : <em> exports extracted features for visualisation (experimental), default False </em> </br>
--test_time_augmentation : <em> use test time augmentation, not implemented </em> </br>
--tta_samples : <em> number of augmented samples to use for test time augmentation, not implemented </em> </br>
</br>
LOGGING AND OUTPUT: </br>
--output : <em> directory for results output (model file, metrics, etc), default scripts/output </em> </br>
--exp_name : <em> Experiment name, defaults to {model_type}_{timestamp} </em> </br>
--save_freq : <em> Save checkpoints every N epochs (0 saves best only) </em> </br>
--debug : <em> debug mode with extra logging (not recommended), default False </em> </br>
--verbose : <em> enable verbose output, default False </em> </br>
--tensorboard : <em> tensorboard logging, not implemented yet </em> </br>
--wandb : <em> enables weights and biases logging, not implemented </em> </br>
--wandb_project : <em> project name for wandb, not implemented </em> </br>
--wandb_entity : <em> wandb entity name, not implemented </em> </br> </em> </br>
</br>
HARDWARE: </br>
--gpu : <em> GPU hardware ID to use, default GPU 0 </em> </br>
--num_workers : <em> number of data loading workers, default 4 </em> </br>
--pin_memory : <em> use pin memory for GPU data transfer acceleration, default False </em> </br>
--mixed_precision : <em> use mixed precision training, not implemented </em> </br>
--distributed : <em> use distributed training, not implemented </em> </br>
--world_size : <em> number of processes for dist training, not implemented </em> </br>
--dist_url : <em> URL for distributed processing, not implemented </em> </br>
</br>
</br>
<h2> Example commands </h2> </br> 
A command can be as simple or complicated as your training setup. Below is an example command for training a 2 layer ViT with bce_logits loss and some augmentations - all other values would be set to defaults. </br> </br>
python train.py --data /home/usr/preprocessed_data/dataset_splits.csv --base_dir /home/usr --loss bce_logits --model vit --vit_depth 2 --augment rotation scale noise cutout --aug_prob 0.4 --weight_decay 0 --track_parameters --balanced_sampling
