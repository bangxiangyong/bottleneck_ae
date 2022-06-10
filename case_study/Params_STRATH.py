import pickle as pickle

import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

from baetorch.baetorch.models_v2.base_layer import flatten_np
from baetorch.baetorch.util.seed import bae_set_seed
from thesis_experiments.Params_GLOBAL import (
    twin_output_map,
    homoscedestic_mode_map,
    likelihood_map,
    n_bae_samples_map,
    bae_type_classes,
)
import torch
from uncertainty_ood_v2.util.sensor_preproc import (
    FFT_Sensor,
    MinMaxSensor,
    Resample_Sensor,
    StandardiseSensor,
    FlattenStandardiseScaler,
    extract_wt_feats,
    FlattenMinMaxScaler,
    MinMaxSeqSensor,
)
from util.evaluate_ood import flag_tukey_fence
import pywt
from scipy.stats import kurtosis
from scipy.stats import skew

# forging_sensors_all = np.array([2, 9, 11, 13, 54, 71, 82, 3])
# forging_sensors_all = np.array([9])
resample_factor = 50

# grid_STRATH = {
#     "random_seed": np.random.randint(
#         0,
#         1000,
#         size=3,
#     ),
#     "apply_fft": [False],
#     "ss_id": [forging_sensors_all],
#     "mode": ["forging"],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [False],
#     "layer_norm": ["none"],  # {"layer","none"}
#     # "layer_norm": ["layer"],
#     "latent_factor": [0.5],
#     "bae_type": ["ae"],
#     # "full_likelihood": ["homo-tgauss"],
#     # "full_likelihood": ["bernoulli"],
#     "full_likelihood": ["mse"],
#     # "full_likelihood": ["hetero-tgauss"],
#     # "full_likelihood": ["hetero-gauss"],
#     "weight_decay": [1e-11],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [50],
# }

# SENSOR RANKING BASED LL
# grid_STRATH = {
#     "random_seed": np.random.randint(0, 1000, size=10),
#     "apply_fft": [False],
#     "ss_id": [2, 9, 11, 13, 54, 71, 82, 3],
#     "mode": ["forging"],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [False],
#     "layer_norm": ["none"],  # {"layer","none"}
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [50],
# }

# EXTRA: SENSOR RANKING BASED LL
# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     "ss_id": [2, 9, 11, 13, 54, 71, 82, 3],
#     "mode": ["forging"],
#     "target_dim": [2],
#     "resample_factor": [resample_factor],
#     "skip": [False],
#     "layer_norm": ["none"],  # {"layer","none"}
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [50],
# }


# LL SWEEP TOPK
# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [50],
# }

# EXTRA: LL SWEEP TOPK
# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [50],
# }

# SINGLE LL
# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     "ss_id": [[9]],
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ens"],
#     "full_likelihood": [
#         "homo-gauss",
#     ],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [5],  # default
#     "num_epochs": [200],
# }

# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 94],
#     "apply_fft": [False],
#     "ss_id": [[13, 71]],
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["std-mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }

# REBOOT: 200 EPOCH LL RANKING
# grid_STRATH = {
#     "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
#     "apply_fft": [False],
#     "ss_id": [2, 9, 11, 13, 54, 71, 82, 3],
#     "mode": ["forging"],
#     "target_dim": [2],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli", "homo-gauss", "homo-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }

# REBOOT: FULL LL + BAE
# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse", "bernoulli", "cbernoulli", "homo-gauss", "homo-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }

# BOTTLENECK
# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False, True],
#     "layer_norm": ["none", "layer"],
#     "latent_factor": [0.1, 0.5, 1.0, 10],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }

# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae", "vae", "mcd", "vi", "ens"],
#     "full_likelihood": ["std-mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
# }

# grid_STRATH = {
#     "random_seed": [79, 835, 792],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[76,9]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["heating"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     # "full_likelihood": ["static-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     # "standardise": ["standard"],
# }

# grid_STRATH = {
#     "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[-1]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "skip": [False],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu", "gelu", "elu", "selu", "none"],
#     "num_dense_layers": [1, 2, 3, 4],
# }

# 9999 # 0.6 AUROC bad
# 5 # 0.8 AUROC decent?
# grid_STRATH = {
#     # "random_seed": [22, 33, 44],
#     "random_seed": [10],
#     "apply_fft": [False],
#     # fmt: off
#     # "ss_id": [[9,13,25,3]],
#     "ss_id": [[9,13]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     # "target_dim": [0],
#     "mode": ["forging"],
#     # "mode": ["heating"],
#     # "resample_factor": [50],
#     # "resample_factor": [10],
#     "resample_factor": [10],
#     "skip": [False],
#     # "skip": [True],
#     "layer_norm": ["none"],
#     "latent_factor": [0.1],
#     # "latent_factor": [2],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     # "full_likelihood": ["homo-tgauss"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [0],
#     "n_conv_layers": [5],
#     "n_enc_capacity": [1],
# }

# grid_STRATH = {
#     "random_seed": [930717, 10, 5477, 2, 125, 5, 7, 8, 98, 123],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[9,13]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [10],
#     "layer_norm": ["none"],
#     "skip": [False, True],
#     "latent_factor": [0.01, 0.05, 0.1, 0.2, 2.0, 10],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [0],
#     "n_conv_layers": [1, 2, 3, 4, 5],
#     "n_enc_capacity": [1, 5, 10, 20, 30],
# }

# grid_STRATH = {
#     "random_seed": [891, 267, 40, 894, 781, 54, 69, 517, 88, 46],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[9]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "layer_norm": ["none", "layer"],
#     "skip": [False, True],
#     "latent_factor": [0.1, 0.2, 1, 2, 10],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [1e-10],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [1, 3, 5],
#     "n_enc_capacity": [20],
# }


# # N LAYERS VS SKIP
# grid_STRATH = {
#     # "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
#     "random_seed": [333],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[9]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     # "mode": ["heating"],
#     "resample_factor": [50],
#     "layer_norm": ["none"],
#     "skip": [False],
#     "latent_factor": [2],
#     "bae_type": ["sae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [0],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [3],
#     "n_enc_capacity": [20],
# }

# grid_STRATH = {
#     "random_seed": [79],
#     "apply_fft": [False],
#     # fmt: off
#     "ss_id": [[9]],
#     # to be replaced via Best TOP-K Script
#     # fmt: on
#     "target_dim": [2],
#     "mode": ["forging"],
#     "resample_factor": [50],
#     "layer_norm": ["none"],
#     "skip": [False],
#     "latent_factor": [1],
#     "bae_type": ["ae"],
#     "full_likelihood": ["mse"],
#     "weight_decay": [0, 1e-10, 1e-6, 1e-4, 1e-3, 1e-2],
#     "n_bae_samples": [-1],  # default
#     "num_epochs": [200],
#     "activation": ["leakyrelu"],
#     "n_dense_layers": [1],
#     "n_conv_layers": [1],
#     "n_enc_capacity": [20],
# }

grid_STRATH = {
    "random_seed": [79, 835, 792, 906, 520, 944, 871, 855, 350, 948],
    "apply_fft": [False],
    # fmt: off
    "ss_id": [[9]],
    # to be replaced via Best TOP-K Script
    # fmt: on
    "target_dim": [2],
    "mode": ["forging"],
    "resample_factor": [50],
    "layer_norm": ["none"],
    "skip": [False],
    "latent_factor": [0.1, 1],
    "bae_type": ["ae"],
    "full_likelihood": ["mse"],
    "weight_decay": [1e-10],
    "n_bae_samples": [-1],  # default
    "num_epochs": [200],
    "activation": ["leakyrelu"],
    "n_dense_layers": [1],
    "n_conv_layers": [1],
    "n_enc_capacity": [20],
}


def prepare_data(pickle_path="pickles"):
    strath_data = pickle.load(
        open(pickle_path + "/" + "strath_inputs_outputs_bxy20.p", "rb")
    )
    # strath_data = pickle.load(open(pickle_path + "/" + "strath_inputs_outputs.p", "rb"))
    # strath_data = pickle.load(open(pickle_path + "/" + "strath_DOE0_UPSAMPLED.p", "rb"))

    # if n_doe == 0:
    #     strath_data = pickle.load(
    #         open(pickle_path + "/" + "strath_inputs_outputs.p", "rb")
    #     )
    # else:
    #     strath_data = pickle.load(
    #         open(pickle_path + "/" + "strath_DOE" + str(n_doe) + ".p", "rb")
    #     )
    return strath_data


def get_x_splits(
    strath_data,
    exp_params,
    min_max_clip=True,
    train_size=0.70,
    tukey_threshold=1.5,
    tukey_adjusted=False,
    apply_dwt=False,
):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test

    # unpack exp_params variables
    apply_fft = exp_params["apply_fft"]
    random_seed = exp_params["random_seed"]
    target_dim = exp_params["target_dim"]
    resample_factor_ = exp_params["resample_factor"]
    sensor_i = exp_params["ss_id"]
    mode = exp_params["mode"]
    standardise = (
        "standard"
        if ("full_likelihood" in exp_params)
        and (exp_params["full_likelihood"] == "std-mse")
        else "minmax"
    )
    # unpack strath data
    heating_traces = strath_data["heating"].copy()
    forging_traces = strath_data["forging"].copy()
    column_names = strath_data["sensor_names"]
    cmm_data_abs_err = strath_data["cmm_data_abs_err"]
    # cmm_data_abs_err = strath_data["cmm_data"]
    cmm_data = strath_data["cmm_data"]
    cmm_data_header = strath_data["cmm_header"]

    # get residuals by subtracting actual from nominal
    # forging_traces[:, :, 9] = forging_traces[:, :, 9] - forging_traces[:, :, 33]
    # forging_traces[:, :, 25] = forging_traces[:, :, 25] - forging_traces[:, :, 49]
    # forging_traces[:, :, 13] = forging_traces[:, :, 13] - forging_traces[:, :, 29]

    # EXPERIMENTAL: apply dwt?
    if apply_dwt:
        RNompos_id = 30
        ANompos_id = 33
        RNompos_id_absdiff = np.abs(np.diff(forging_traces[:, :, RNompos_id], axis=-1))
        ANompos_id_absdiff = np.abs(np.diff(forging_traces[:, :, ANompos_id], axis=-1))

        def get_peaks(arr_, threshold=1):
            peaks, _ = find_peaks(arr_, height=threshold)
            return peaks

        peaksR = np.apply_along_axis(
            func1d=get_peaks, axis=1, arr=RNompos_id_absdiff, threshold=1
        )
        peaksA = np.apply_along_axis(
            func1d=get_peaks, axis=1, arr=ANompos_id_absdiff, threshold=10
        )
        segmented_forging_traces = []
        all_lens = []
        for peak_r, peak_a in zip(peaksR, peaksA):
            all_lens.append(peak_a[-1] - peak_r[1])
        min_len = np.min(all_lens)
        for trace, peak_r, peak_a in zip(forging_traces, peaksR, peaksA):
            segmented_forging_traces.append(trace[peak_r[1] : peak_a[-1]][:min_len])
        forging_traces = np.array(segmented_forging_traces)

    # select sensors
    target_dim = [target_dim] if not isinstance(target_dim, list) else target_dim
    heating_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)
    forging_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)

    # get heating/forging traces
    x_heating = heating_traces[:, :, heating_sensors]
    x_forging = forging_traces[:, :, forging_sensors]

    # ood and id
    tukey_flags = np.apply_along_axis(
        flag_tukey_fence,
        arr=cmm_data_abs_err,
        axis=0,
        level=tukey_threshold,
        adjusted=tukey_adjusted,
    )
    tukey_flags = tukey_flags[:, target_dim] if len(target_dim) > 0 else tukey_flags
    ood_args = np.unique(np.argwhere(tukey_flags.sum(1) >= 1)[:, 0])
    id_args = np.unique(np.argwhere(tukey_flags.sum(1) == 0)[:, 0])

    # get id and ood args
    x_heating_train = x_heating[id_args]
    x_forging_train = x_forging[id_args]
    x_heating_ood = x_heating[ood_args]
    x_forging_ood = x_forging[ood_args]

    # move axis
    x_heating_train = np.moveaxis(x_heating_train, 1, 2)
    x_forging_train = np.moveaxis(x_forging_train, 1, 2)
    x_heating_ood = np.moveaxis(x_heating_ood, 1, 2)
    x_forging_ood = np.moveaxis(x_forging_ood, 1, 2)

    if mode == "forging":
        x_id_train = x_forging_train
        x_ood_test = x_forging_ood

    elif mode == "heating":
        x_id_train = x_heating_train
        x_ood_test = x_heating_ood

    # data splitting
    x_id_train, x_id_test = train_test_split(
        x_id_train, train_size=train_size, shuffle=True, random_state=random_seed
    )

    # option to apply fft
    if apply_fft:
        x_id_train = FFT_Sensor().transform(x_id_train)
        x_id_test = FFT_Sensor().transform(x_id_test)
        x_ood_test = FFT_Sensor().transform(x_ood_test)

    # resample
    if resample_factor_ > 1:
        downsample_type = "decimate"
        x_id_train = Resample_Sensor().transform(
            x_id_train, n=resample_factor_, downsample_type=downsample_type
        )
        x_id_test = Resample_Sensor().transform(
            x_id_test, n=resample_factor_, downsample_type=downsample_type
        )
        x_ood_test = Resample_Sensor().transform(
            x_ood_test, n=resample_factor_, downsample_type=downsample_type
        )

    if apply_dwt:
        # apply wavelet?
        # wt_type = "db2"
        wt_type = "haar"
        # wt_level = None
        wt_level = 3
        wt_summarise = True
        # wt_summarise = False
        wt_partial = False

        x_id_train = extract_wt_feats(
            x_id_train,
            wt_type=wt_type,
            wt_level=wt_level,
            wt_summarise=wt_summarise,
            wt_partial=wt_partial,
        )
        x_id_test = extract_wt_feats(
            x_id_test,
            wt_type=wt_type,
            wt_level=wt_level,
            wt_summarise=wt_summarise,
            wt_partial=wt_partial,
        )
        x_ood_test = extract_wt_feats(
            x_ood_test,
            wt_type=wt_type,
            wt_level=wt_level,
            wt_summarise=wt_summarise,
            wt_partial=wt_partial,
        )

    # Min max
    if standardise == "minmax":
        sensor_scaler = MinMaxSensor(
            num_sensors=x_id_train.shape[1], axis=1, clip=min_max_clip
        )
    elif standardise == "standard":
        sensor_scaler = StandardiseSensor(num_sensors=x_id_train.shape[1], axis=1)

    x_id_train = sensor_scaler.fit_transform(x_id_train)
    x_id_test = sensor_scaler.transform(x_id_test)
    x_ood_test = sensor_scaler.transform(x_ood_test)

    return x_id_train, x_id_test, x_ood_test


def get_x_splits_combine(
    strath_data,
    exp_params,
    min_max_clip=True,
    train_size=0.70,
    tukey_threshold=1.5,
    tukey_adjusted=False,
    apply_dwt=False,
):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test

    # unpack exp_params variables
    apply_fft = exp_params["apply_fft"]
    random_seed = exp_params["random_seed"]
    target_dim = exp_params["target_dim"]
    resample_factor_ = exp_params["resample_factor"]
    sensor_i = exp_params["ss_id"]
    mode = exp_params["mode"]
    standardise = "standard" if exp_params["full_likelihood"] == "std-mse" else "minmax"

    # unpack strath data
    # heating_traces = strath_data["heating"].copy()
    # forging_traces = strath_data["forging"].copy()
    # heating_traces = strath_data["heating"]
    heating_traces = strath_data["transfer"]
    forging_traces = strath_data["forging"]
    column_names = strath_data["sensor_names"]
    cmm_data_abs_err = strath_data["cmm_data_abs_err"]
    # cmm_data_abs_err = strath_data["cmm_data"]
    cmm_data = strath_data["cmm_data"]
    cmm_data_header = strath_data["cmm_header"]

    # get residuals by subtracting actual from nominal

    # select sensors
    target_dim = [target_dim] if not isinstance(target_dim, list) else target_dim
    heating_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)
    forging_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)

    # get heating/forging traces
    x_heating = heating_traces[:, :, [76]]
    x_forging = forging_traces[:, :, [9]]

    # ood and id
    tukey_flags = np.apply_along_axis(
        flag_tukey_fence,
        arr=cmm_data_abs_err,
        axis=0,
        level=tukey_threshold,
        adjusted=tukey_adjusted,
    )
    tukey_flags = tukey_flags[:, target_dim] if len(target_dim) > 0 else tukey_flags
    ood_args = np.unique(np.argwhere(tukey_flags.sum(1) >= 1)[:, 0])
    id_args = np.unique(np.argwhere(tukey_flags.sum(1) == 0)[:, 0])

    # get id and ood args
    x_heating_train = x_heating[id_args]
    x_forging_train = x_forging[id_args]
    x_heating_ood = x_heating[ood_args]
    x_forging_ood = x_forging[ood_args]

    # move axis
    x_heating_train = np.moveaxis(x_heating_train, 1, 2)
    x_forging_train = np.moveaxis(x_forging_train, 1, 2)
    x_heating_ood = np.moveaxis(x_heating_ood, 1, 2)
    x_forging_ood = np.moveaxis(x_forging_ood, 1, 2)

    if mode == "forging":
        x_id_train = x_forging_train
        x_ood_test = x_forging_ood

    elif mode == "heating":
        x_id_train = x_heating_train
        x_ood_test = x_heating_ood

    # resample
    if resample_factor_ > 1:
        downsample_type = "decimate"
        x_heating_train = Resample_Sensor().transform(
            x_heating_train, n=resample_factor_, downsample_type=downsample_type
        )
        x_forging_train = Resample_Sensor().transform(
            x_forging_train, n=resample_factor_, downsample_type=downsample_type
        )
        x_heating_ood = Resample_Sensor().transform(
            x_heating_ood, n=resample_factor_, downsample_type=downsample_type
        )
        x_forging_ood = Resample_Sensor().transform(
            x_forging_ood, n=resample_factor_, downsample_type=downsample_type
        )
    # data splitting
    x_heating_train, x_heating_test = train_test_split(
        x_heating_train, train_size=train_size, shuffle=True, random_state=random_seed
    )
    x_forging_train, x_forging_test = train_test_split(
        x_forging_train, train_size=train_size, shuffle=True, random_state=random_seed
    )

    # Min max
    if standardise == "minmax":
        sensor_scaler = MinMaxSensor(
            num_sensors=x_forging_train.shape[1], axis=1, clip=min_max_clip
        )
    elif standardise == "standard":
        sensor_scaler = StandardiseSensor(num_sensors=x_forging_train.shape[1], axis=1)

    x_heating_train = sensor_scaler.fit_transform(x_heating_train)
    x_heating_test = sensor_scaler.transform(x_heating_test)
    x_heating_ood = sensor_scaler.transform(x_heating_ood)

    x_forging_train = sensor_scaler.fit_transform(x_forging_train)
    x_forging_test = sensor_scaler.transform(x_forging_test)
    x_forging_ood = sensor_scaler.transform(x_forging_ood)

    x_id_train = np.concatenate((x_heating_train, x_forging_train), axis=2)
    x_id_test = np.concatenate((x_heating_test, x_forging_test), axis=2)
    x_ood_test = np.concatenate((x_heating_ood, x_forging_ood), axis=2)

    # x_id_train = x_forging_train
    # x_id_test = x_forging_test
    # x_ood_test = x_forging_ood

    return x_id_train, x_id_test, x_ood_test


def get_x_splits_combine_v2(
    strath_data,
    exp_params,
    min_max_clip=True,
    train_size=0.70,
    tukey_threshold=1.5,
    tukey_adjusted=False,
    apply_dwt=False,
):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test

    # unpack exp_params variables
    apply_fft = exp_params["apply_fft"]
    random_seed = exp_params["random_seed"]
    target_dim = exp_params["target_dim"]
    resample_factor_ = exp_params["resample_factor"]
    sensor_i = exp_params["ss_id"]
    mode = exp_params["mode"]
    standardise = "standard" if exp_params["full_likelihood"] == "std-mse" else "minmax"

    # unpack strath data
    # heating_traces = strath_data["heating"].copy()
    # forging_traces = strath_data["forging"].copy()
    # heating_traces = strath_data["heating"]
    heating_traces = strath_data["transfer"]
    forging_traces = strath_data["forging"]
    column_names = strath_data["sensor_names"]
    cmm_data_abs_err = strath_data["cmm_data_abs_err"]
    # cmm_data_abs_err = strath_data["cmm_data"]
    cmm_data = strath_data["cmm_data"]
    cmm_data_header = strath_data["cmm_header"]

    # get residuals by subtracting actual from nominal

    # select sensors
    target_dim = [target_dim] if not isinstance(target_dim, list) else target_dim
    heating_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)
    forging_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)

    # get heating/forging traces
    x_heating = heating_traces[:, :, [76]]
    x_forging = forging_traces[:, :, [9]]

    # ood and id
    tukey_flags = np.apply_along_axis(
        flag_tukey_fence,
        arr=cmm_data_abs_err,
        axis=0,
        level=tukey_threshold,
        adjusted=tukey_adjusted,
    )
    tukey_flags = tukey_flags[:, target_dim] if len(target_dim) > 0 else tukey_flags
    ood_args = np.unique(np.argwhere(tukey_flags.sum(1) >= 1)[:, 0])
    id_args = np.unique(np.argwhere(tukey_flags.sum(1) == 0)[:, 0])

    # get id and ood args
    x_heating_train = x_heating[id_args]
    x_forging_train = x_forging[id_args]
    x_heating_ood = x_heating[ood_args]
    x_forging_ood = x_forging[ood_args]

    # move axis
    x_heating_train = np.moveaxis(x_heating_train, 1, 2)
    x_forging_train = np.moveaxis(x_forging_train, 1, 2)
    x_heating_ood = np.moveaxis(x_heating_ood, 1, 2)
    x_forging_ood = np.moveaxis(x_forging_ood, 1, 2)

    # resample
    if resample_factor_ > 1:
        downsample_type = "decimate"
        # downsample_type = "agg"
        target_len = resample_factor_
        # forging
        forge_factor_ = x_forging_train.shape[-1] / target_len
        x_forging_train = Resample_Sensor().transform(
            x_forging_train, n=forge_factor_, downsample_type=downsample_type
        )
        x_forging_ood = Resample_Sensor().transform(
            x_forging_ood, n=forge_factor_, downsample_type=downsample_type
        )

        # heating
        heat_factor_ = x_heating_train.shape[-1] / target_len
        x_heating_train = Resample_Sensor().transform(
            x_heating_train, n=heat_factor_, downsample_type=downsample_type
        )
        x_heating_ood = Resample_Sensor().transform(
            x_heating_ood, n=heat_factor_, downsample_type=downsample_type
        )
    print(x_forging_train.shape)
    print(x_heating_ood.shape)

    x_id_train = np.concatenate((x_forging_train, x_heating_train), axis=1)
    x_ood_test = np.concatenate((x_forging_ood, x_heating_ood), axis=1)

    # data splitting
    x_id_train, x_id_test = train_test_split(
        x_id_train, train_size=train_size, shuffle=True, random_state=random_seed
    )

    # Min max
    if standardise == "minmax":
        sensor_scaler = MinMaxSensor(
            num_sensors=x_id_train.shape[1], axis=1, clip=min_max_clip
        )
    elif standardise == "standard":
        sensor_scaler = StandardiseSensor(num_sensors=x_id_train.shape[1], axis=1)

    x_id_train = sensor_scaler.fit_transform(x_id_train)
    x_id_test = sensor_scaler.transform(x_id_test)
    x_ood_test = sensor_scaler.transform(x_ood_test)

    return x_id_train, x_id_test, x_ood_test


def get_x_splits_ADHOC(
    strath_data,
    exp_params,
    min_max_clip=True,
    train_size=0.70,
    tukey_threshold=1.5,
    tukey_adjusted=False,
):
    # accepts raw data and exp_params
    # returns x_id_train, x_id_test, x_ood_test

    # unpack exp_params variables
    apply_fft = exp_params["apply_fft"]
    random_seed = exp_params["random_seed"]
    target_dim = exp_params["target_dim"]
    resample_factor_ = exp_params["resample_factor"]
    sensor_i = exp_params["ss_id"]
    mode = exp_params["mode"]
    standardise = "standard" if exp_params["full_likelihood"] == "std-mse" else "minmax"

    # unpack strath data
    heating_traces = strath_data["heating"].copy()
    forging_traces = strath_data["forging"].copy()
    column_names = strath_data["sensor_names"]
    cmm_data_abs_err = strath_data["cmm_data_abs_err"]
    # cmm_data_abs_err = strath_data["cmm_data"]
    # cmm_data_abs_err = strath_data["cmm_data"] - 38
    cmm_data = strath_data["cmm_data"]
    cmm_data_header = strath_data["cmm_header"]

    # get residuals by subtracting actual from nominal
    forging_traces[:, :, 9] = forging_traces[:, :, 9] - forging_traces[:, :, 33]
    forging_traces[:, :, 25] = forging_traces[:, :, 25] - forging_traces[:, :, 49]
    forging_traces[:, :, 13] = forging_traces[:, :, 13] - forging_traces[:, :, 29]
    forging_traces[:, :, 21] = forging_traces[:, :, 21] - forging_traces[:, :, 45]
    forging_traces[:, :, 14] = forging_traces[:, :, 14] - forging_traces[:, :, 30]
    forging_traces[:, :, 22] = forging_traces[:, :, 22] - forging_traces[:, :, 46]
    forging_traces[:, :, 11] = forging_traces[:, :, 11] - forging_traces[:, :, 35]
    forging_traces[:, :, 17] = forging_traces[:, :, 17] - forging_traces[:, :, 57]

    ### segment forging only
    # RNompos_id = 30
    # ANompos_id = 33
    # RNompos_id_absdiff = np.abs(np.diff(forging_traces[:, :, RNompos_id], axis=-1))
    # ANompos_id_absdiff = np.abs(np.diff(forging_traces[:, :, ANompos_id], axis=-1))
    #
    # def get_peaks(arr_, threshold=1):
    #     peaks, _ = find_peaks(arr_, height=threshold)
    #     return peaks
    #
    # peaksR = np.apply_along_axis(
    #     func1d=get_peaks, axis=1, arr=RNompos_id_absdiff, threshold=1
    # )
    # peaksA = np.apply_along_axis(
    #     func1d=get_peaks, axis=1, arr=ANompos_id_absdiff, threshold=10
    # )
    # segmented_forging_traces = []
    # all_lens = []
    # for peak_r, peak_a in zip(peaksR, peaksA):
    #     all_lens.append(peak_a[-1] - peak_r[1])
    # min_len = np.min(all_lens)
    # for trace, peak_r, peak_a in zip(forging_traces, peaksR, peaksA):
    #     segmented_forging_traces.append(trace[peak_r[1] : peak_a[-1]][:min_len])
    # forging_traces = np.array(segmented_forging_traces)

    ### select sensors
    target_dim = [target_dim] if not isinstance(target_dim, list) else target_dim
    heating_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)
    forging_sensors = np.array([sensor_i] if isinstance(sensor_i, int) else sensor_i)

    # # get heating/forging traces
    x_heating = heating_traces[:, :, heating_sensors]
    x_forging = forging_traces[:, :, forging_sensors]
    print(x_heating.shape)
    print(x_forging.shape)

    # censor some samples?
    # censor_args = np.array(
    #     [
    #         i
    #         for i in np.arange(len(cmm_data_abs_err))
    #         # if i not in [0, 1, 2, 3, 4, 5, 71, 72, 73]
    #         # if i not in [0, 1, 2]
    #         # if i not in [0, 1, 2, 69, 70]
    #         if i not in [0, 1]
    #     ]
    # )
    # cmm_data_abs_err = cmm_data_abs_err[censor_args]
    # x_forging = x_forging[censor_args]
    # x_heating = x_heating[censor_args]

    ## Truncate sequence to relevant stages
    # truncate_seq = np.arange(400, 4000)  # ss3
    # truncate_seq = np.arange(400, 4100)  # ss3
    # truncate_seq = np.arange(400, 4300)  # ss9
    # x_forging = x_forging[:, truncate_seq]
    # x_heating = x_heating[:, truncate_seq]

    # get ood and id
    tukey_flags = np.apply_along_axis(
        flag_tukey_fence,
        arr=cmm_data_abs_err,
        axis=0,
        level=tukey_threshold,
        adjusted=tukey_adjusted,
    )
    tukey_flags = tukey_flags[:, target_dim] if len(target_dim) > 0 else tukey_flags
    ood_args = np.unique(np.argwhere(tukey_flags.sum(1) >= 1)[:, 0])
    id_args = np.unique(np.argwhere(tukey_flags.sum(1) == 0)[:, 0])

    # drop some
    # drop_early_args = 1111
    # drop_last_args = 900
    # ood_args = [i for i in ood_args if ((i > drop_early_args) and (i < drop_last_args))]
    # id_args = [i for i in id_args if ((i > drop_early_args) and (i < drop_last_args))]

    # ood_args = np.arange(len(cmm_data_abs_err))[54:70]
    # id_args = np.arange(len(cmm_data_abs_err))[3:54]

    # get id and ood args
    x_heating_train = x_heating[id_args]
    x_forging_train = x_forging[id_args]
    x_heating_ood = x_heating[ood_args]
    x_forging_ood = x_forging[ood_args]

    # move axis
    x_heating_train = np.moveaxis(x_heating_train, 1, 2)
    x_forging_train = np.moveaxis(x_forging_train, 1, 2)
    x_heating_ood = np.moveaxis(x_heating_ood, 1, 2)
    x_forging_ood = np.moveaxis(x_forging_ood, 1, 2)

    if mode == "forging":
        x_id_train = x_forging_train
        x_ood_test = x_forging_ood

    elif mode == "heating":
        x_id_train = x_heating_train
        x_ood_test = x_heating_ood

    # data splitting
    x_id_train, x_id_test = train_test_split(
        x_id_train, train_size=train_size, shuffle=True, random_state=random_seed
    )
    id_train_args, id_test_args = train_test_split(
        id_args, train_size=train_size, shuffle=True, random_state=random_seed
    )
    # x_id_test = np.moveaxis(x_forging, 1, 2)[id_test_args]
    # x_id_train = np.moveaxis(x_forging, 1, 2)[id_train_args]

    # option to apply fft
    if apply_fft:
        x_id_train = FFT_Sensor().transform(x_id_train)
        x_id_test = FFT_Sensor().transform(x_id_test)
        x_ood_test = FFT_Sensor().transform(x_ood_test)

    # resample
    if resample_factor_ > 1:
        downsample_type = "decimate"
        # downsample_type = "agg"
        x_id_train = Resample_Sensor().transform(
            x_id_train.copy(), n=resample_factor_, downsample_type=downsample_type
        )
        x_id_test = Resample_Sensor().transform(
            x_id_test.copy(), n=resample_factor_, downsample_type=downsample_type
        )
        x_ood_test = Resample_Sensor().transform(
            x_ood_test.copy(), n=resample_factor_, downsample_type=downsample_type
        )

    # Min max
    if standardise == "minmax":
        sensor_scaler = MinMaxSensor(
            num_sensors=x_id_train.shape[1], axis=1, clip=min_max_clip
        )
        # sensor_scaler = MinMaxSeqSensor(
        #     num_sensors=x_id_train.shape[1], clip=min_max_clip
        # )
        # sensor_scaler = FlattenMinMaxScaler()
    elif standardise == "standard":
        sensor_scaler = StandardiseSensor(num_sensors=x_id_train.shape[1], axis=1)

    # apply wavelet?
    # wt_type = "db2"
    wt_type = "haar"
    # wt_level = None
    wt_level = 3
    # wt_summarise = True
    wt_summarise = False
    wt_partial = False

    # x_id_train = extract_wt_feats(
    #     x_id_train,
    #     wt_type=wt_type,
    #     wt_level=wt_level,
    #     wt_summarise=wt_summarise,
    #     wt_partial=wt_partial,
    # )
    # x_id_test = extract_wt_feats(
    #     x_id_test,
    #     wt_type=wt_type,
    #     wt_level=wt_level,
    #     wt_summarise=wt_summarise,
    #     wt_partial=wt_partial,
    # )
    # x_ood_test = extract_wt_feats(
    #     x_ood_test,
    #     wt_type=wt_type,
    #     wt_level=wt_level,
    #     wt_summarise=wt_summarise,
    #     wt_partial=wt_partial,
    # )
    x_id_train = sensor_scaler.fit_transform(x_id_train)
    x_id_test = sensor_scaler.transform(x_id_test)
    x_ood_test = sensor_scaler.transform(x_ood_test)

    # trace_ss = x_id_train[0, 0]
    # temp_dwt = pywt.wavedec(trace_ss, wavelet=wt_type, level=wt_level)
    # len_dwt = np.cumsum([len(dwt_) for dwt_ in temp_dwt])
    # x_id_train = extract_wt_feats(
    #     x_id_train,
    #     wt_type=wt_type,
    #     wt_level=wt_level,
    #     wt_summarise=wt_summarise,
    #     wt_partial=wt_partial,
    # )
    # x_id_test = extract_wt_feats(
    #     x_id_test,
    #     wt_type=wt_type,
    #     wt_level=wt_level,
    #     wt_summarise=wt_summarise,
    #     wt_partial=wt_partial,
    # )
    # x_ood_test = extract_wt_feats(
    #     x_ood_test,
    #     wt_type=wt_type,
    #     wt_level=wt_level,
    #     wt_summarise=wt_summarise,
    #     wt_partial=wt_partial,
    # )
    # x_id_train = sensor_scaler.fit_transform(x_id_train, len_dwt=len_dwt)
    # x_id_test = sensor_scaler.transform(x_id_test)
    # x_ood_test = sensor_scaler.transform(x_ood_test)

    # # =====CONVERT TO DWT WITH MIN MAX SCALING=======
    # trace_ss = x_id_train[0,0]
    # temp_dwt = pywt.wavedec(trace_ss, wavelet=wt_type, level=wt_level)
    # len_dwt = [len(dwt_) for dwt_ in temp_dwt]
    #
    # for len_ in len_dwt:
    #
    # dwt_scaler =

    # dwt_id_train = []
    # dwt_id_test = []
    # dwt_ood_test = []
    #
    # for (x_traces, dwt_res) in zip(
    #     [x_id_train, x_id_test, x_ood_test], [dwt_id_train, dwt_id_test, dwt_ood_test]
    # ):
    #     for trace in x_traces:
    #         dwt_ = [
    #             pywt.wavedec(trace_ss, wavelet=wt_type, level=wt_level)
    #             for trace_ss in trace
    #         ]
    #         dwt_res.append(dwt_)
    #
    # scaled_dwt_id_train =
    #
    #     for dwt_trace in dwt_id_train:
    #
    #
    # cwt_id_train = np.array(dwt_id_train)
    # cwt_id_test = np.array(dwt_id_test)
    # cwt_ood_test = np.array(dwt_ood_test)
    # x_id_train = sensor_scaler.fit_transform(cwt_id_train)
    # x_id_test = sensor_scaler.transform(cwt_id_test)
    # x_ood_test = sensor_scaler.transform(cwt_ood_test)

    # =====CONVERT TO CWT=======
    # cwt_id_train = []
    # cwt_id_test = []
    # cwt_ood_test = []
    #
    # for (x_traces, cwt_res) in zip(
    #     [x_id_train, x_id_test, x_ood_test], [cwt_id_train, cwt_id_test, cwt_ood_test]
    # ):
    #     for trace in x_traces:
    #         cwt_ = [
    #             pywt.cwt(trace_ss, np.arange(1, len(trace_ss) // 2), "mexh")[0]
    #             for trace_ss in trace
    #         ]
    #         cwt_res.append(cwt_)
    # cwt_id_train = np.array(cwt_id_train)
    # cwt_id_test = np.array(cwt_id_test)
    # cwt_ood_test = np.array(cwt_ood_test)
    # x_id_train = sensor_scaler.fit_transform(cwt_id_train)
    # x_id_test = sensor_scaler.transform(cwt_id_test)
    # x_ood_test = sensor_scaler.transform(cwt_ood_test)

    # ========================

    return (
        x_id_train,
        x_id_test,
        x_ood_test,
        id_train_args,
        id_test_args,
        ood_args,
        cmm_data_abs_err,
    )


def get_bae_model(
    exp_params,
    x_id_train,
    activation="leakyrelu",
    se_block=False,
    bias=False,
    use_cuda=False,
    lr=0.01,
    dropout_rate=0.01,
    mean_prior_loss=False,
    collect_grads=False,
):
    # unpack exp_params variables
    random_seed = exp_params["random_seed"]
    bae_type = exp_params["bae_type"]
    latent_factor = exp_params["latent_factor"]
    layer_norm = exp_params["layer_norm"]
    full_likelihood_i = exp_params["full_likelihood"]
    skip = exp_params["skip"]
    weight_decay = exp_params["weight_decay"]
    n_bae_samples = exp_params["n_bae_samples"]
    standardise = "standard" if exp_params["full_likelihood"] == "std-mse" else "minmax"
    bae_set_seed(random_seed)

    # size of network
    n_dense_layers = (
        exp_params["n_dense_layers"] if "n_dense_layers" in exp_params.keys() else 1
    )
    n_conv_layers = (
        exp_params["n_conv_layers"] if "n_conv_layers" in exp_params.keys() else 1
    )
    n_enc_capacity = (
        exp_params["n_enc_capacity"] if "n_enc_capacity" in exp_params.keys() else 1
    )
    n_conv_filters = n_enc_capacity * 1
    # n_dense_nodes = n_enc_capacity * 100
    n_dense_nodes = 1000

    # continue unpacking based on chosen likelihood param
    twin_output = twin_output_map[full_likelihood_i]
    homoscedestic_mode = homoscedestic_mode_map[full_likelihood_i]
    likelihood = likelihood_map[full_likelihood_i]
    if n_bae_samples == -1:  # resort to default
        n_bae_samples = n_bae_samples_map[bae_type]

    # unpack this function params
    dropout_params = {"dropout_rate": dropout_rate} if bae_type == "mcd" else {}

    # get input dimensions
    # required to scale the latent dim and specify architecture input nodes
    input_dim = x_id_train.shape[-1]
    if isinstance(x_id_train, torch.utils.data.dataloader.DataLoader):
        shapes = list(next(iter(x_id_train))[0].shape)
        latent_dim = int(np.product(shapes) * latent_factor)
    else:  # numpy
        latent_dim = int(np.product(x_id_train.shape[1:]) * latent_factor)
    latent_dim = np.clip(latent_dim, 1, None)

    # specify architecture
    chain_params = [
        # {
        #     "base": "conv1d",
        #     "input_dim": input_dim,
        #     "conv_channels": [x_id_train.shape[1], 10, 20],
        #     "conv_stride": [2, 2],
        #     "conv_kernel": [8, 2],
        #     "activation": activation,
        #     "norm": layer_norm,
        #     "se_block": se_block,
        #     "order": ["base", "norm", "activation"],
        #     "bias": bias,
        #     "last_norm": layer_norm,
        # },
        # {
        #     "base": "linear",
        #     "architecture": [1000] * n_dense_layers + [latent_dim],
        #     "activation": activation,
        #     "norm": "none",
        #     "last_norm": "none",
        # },
        {
            "base": "conv1d",
            "input_dim": input_dim,
            "conv_channels": [x_id_train.shape[1], 10]
            + [n_conv_filters] * n_conv_layers,
            "conv_stride": [2] + [2] * n_conv_layers,
            "conv_kernel": [8] + [2] * n_conv_layers,
            "activation": activation,
            "norm": layer_norm,
            "se_block": se_block,
            "order": ["base", "norm", "activation"],
            "bias": bias,
            "last_norm": layer_norm,
        },
        {
            "base": "linear",
            "architecture": [n_dense_nodes] * n_dense_layers + [latent_dim],
            "activation": activation,
            "norm": "none",
            "last_norm": "none",
            "bias": bias,
        },
    ]

    bae_model = bae_type_classes[bae_type](
        chain_params=chain_params,
        last_activation="sigmoid" if standardise == "minmax" else "none",
        last_norm="none",
        twin_output=twin_output,
        twin_params={"activation": "none", "norm": "none"},
        skip=skip,
        use_cuda=use_cuda,
        scaler_enabled=False,
        homoscedestic_mode=homoscedestic_mode,
        likelihood=likelihood,
        weight_decay=weight_decay,
        num_samples=n_bae_samples,
        anchored=True if bae_type == "ens" else False,
        l1_prior=True if bae_type == "sae" else False,
        learning_rate=lr,
        stochastic_seed=random_seed,
        mean_prior_loss=mean_prior_loss,
        collect_grads=collect_grads,
        **dropout_params,
    )

    return bae_model
