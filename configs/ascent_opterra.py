# ASCENT's inference command currently loads Python configuration files.
# Large public artifacts are kept off the nearly-full workspace drive.

# ---------------- DATASET ----------------
dataset_file_image = "/mnt/d/CodexData/wha_neuron_data/InHouse_Opterra_CH_RG_Day1_OP50Stim_1100frames.h5"
dataset_file_coord = "work/data/InHouse_Opterra_coords_PRED.csv"
dataset_image_channel = 0
dataset_axis_order = "ZYX"
dataset_spacing = [1.5, 0.243, 0.243]
dataset_normalize = "percentile"
dataset_norm_p_low = 1.0
dataset_norm_p_high = 99.99

# ---------------- MODEL ------------------
dataset_note = "Device-restricted Opterra benchmark; mNeptune channel used for identity embeddings."
model_ckpt = "/mnt/d/CodexData/wha_neuron_data/NETr-Opterra.pth"
model_lf_patch_size_xy = 64
model_lf_patch_size_z = 3
model_lf_pretrained = "imagenet_channelvit_small_p16_DINO"
model_pe_num_mlp_layers = 3
model_pe_norm = "batch"
model_pe_scaling = "relative"
model_tr_d_model = 256
model_tr_nhead = 4
model_tr_num_encoder_layers = 4
model_tr_dim_feedforward = 512
model_tr_dropout = 0.1
model_tr_activation = "relu"
model_use_local_features = True
model_use_positional_encoding = True
model_use_transformer = True

# ---------------- TRACKING ---------------
tracking_momentum = 0.5
tracking_temperature = 0.05
tracking_max_gap_frames = 9999
tracking_w_within = 0

# ---------------- RUNTIME ----------------
runtime_output_dir = "work/results/ascent_end_to_end"
runtime_output_prefix = "opterra_pred_detections"
runtime_batch_size_frame = 16
runtime_device = "cuda"
runtime_loglevel = "INFO"
