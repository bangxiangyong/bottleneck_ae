import numpy as np
from scipy.fft import fft
import pandas as pd
from scipy.signal import resample, decimate
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pywt
from scipy.stats import kurtosis
from scipy.stats import skew


def apply_along_sensor(sensor_data, func1d, seq_axis=2, *args, **kwargs):
    """
    Apply a func1d on every independent sequence with data of shape (batch,sensor,seq)
    """
    if seq_axis == 2 or seq_axis == -1:
        sensor_axis = 1
    else:
        sensor_axis = 2

    transformed_x = []
    for sensor_i in range(sensor_data.shape[sensor_axis]):
        transform_batch = np.apply_along_axis(
            func1d=func1d,
            arr=np.take(sensor_data, sensor_i, sensor_axis),
            axis=1,
            *args,
            **kwargs
        )
        transformed_x.append(transform_batch)

    transformed_x = np.array(transformed_x)
    transformed_x = np.moveaxis(transformed_x, 0, sensor_axis)

    return transformed_x


class Resample_Sensor:
    """
    Resample by grouping into equal-sized bins and calculate mean of each bin
    """

    def downsample_data(self, data_series, n=10):
        temp_df = pd.DataFrame(data_series)
        resampled_data = (
            temp_df.groupby(np.arange(len(temp_df)) // n).mean().values.squeeze(-1)
        )
        return resampled_data

    def decimate_data(self, data_series, n=10, order=8):
        data_series_copy = data_series.copy()
        if n > 10:
            first_n = n // 5
            second_n = 5
            return decimate(
                decimate(data_series_copy, q=first_n, ftype="iir", n=order),
                second_n,
                ftype="iir",
                n=order,
            )
        else:
            return decimate(data_series_copy, q=n, ftype="iir", n=order)

    def upsample_data(self, data_series, n=10):
        x_ori = np.linspace(0, 1, len(data_series))
        x_upsampled = np.linspace(0, 1, len(data_series) * n)
        inter_f = interp1d(x_ori, data_series)
        resampled_data = inter_f(x_upsampled)
        return resampled_data

    def transform(self, x, seq_axis=2, n=10, mode="down", downsample_type="agg"):
        # set mode
        if mode == "down":
            if downsample_type == "agg":
                func1d = self.downsample_data
            elif downsample_type == "decimate":
                func1d = self.decimate_data
            else:
                raise NotImplemented("downsample_type can only be 'agg' or 'decimate'")
        elif mode == "up":
            func1d = self.upsample_data
        else:
            raise NotImplemented("Resample mode can be either up or down only.")

        if len(x.shape) > 1:
            return apply_along_sensor(
                sensor_data=x,
                func1d=func1d,
                seq_axis=seq_axis,
                n=n,
            )
        else:
            return func1d(x, n)


class Resample_Sensor_Fourier:
    """
    Resample by grouping into equal-sized bins and calculate mean of each bin
    """

    def resample_data(self, data_series, n=10):
        temp_df = pd.DataFrame(data_series)
        resampled_data = (
            temp_df.groupby(np.arange(len(temp_df)) // n).mean().values.squeeze(-1)
        )
        return resampled_data

    def transform(self, x, seq_axis=2, seq_len=600):
        # specify seq and sens axis
        if seq_axis == 2:
            sens_axis = 1
            x_resampled = np.zeros((x.shape[0], x.shape[sens_axis], seq_len))
        elif seq_axis == 1:
            sens_axis = 2
            x_resampled = np.zeros((x.shape[0], seq_len, x.shape[sens_axis]))
        else:
            raise ValueError("Seq axis can be either 1 or 2.")

        # ignore if already same size as seq_len ?
        if seq_len == x.shape[seq_axis]:
            return np.copy(x)

        # apply transformation over sensor axis
        for sensor_id in range(x.shape[sens_axis]):
            if sens_axis == 1:
                x_resampled[:, sensor_id] = resample(x[:, sensor_id], seq_len, axis=1)
            else:
                x_resampled[:, :, sensor_id] = resample(
                    x[:, :, sensor_id], seq_len, axis=1
                )
        return x_resampled


class FFT_Sensor:
    def apply_fft(self, seq_1d):
        N = len(seq_1d)
        trace_fft = 2.0 / N * np.abs(fft(seq_1d)[: N // 2])
        trace_fft = trace_fft[1:]
        return trace_fft

    def transform(self, x, seq_axis=2):
        return apply_along_sensor(
            sensor_data=x, func1d=self.apply_fft, seq_axis=seq_axis
        )


class MinMaxSensor:
    def __init__(self, num_sensors=11, axis=1, clip=True):
        self.num_sensors = num_sensors
        self.max_sensors = []
        self.min_sensors = []
        self.axis = axis
        self.clip = clip

    def fit(self, x):
        self.max_sensors = []
        self.min_sensors = []
        for sensor in range(self.num_sensors):
            slice_ = np.take(x, sensor, axis=self.axis)
            self.max_sensors.append(np.max(slice_))
            self.min_sensors.append(np.min(slice_))

    def transform(self, x):
        trans = np.copy(x)
        for sensor in range(self.num_sensors):
            slice_ = np.take(trans, sensor, axis=self.axis)
            slice_ = (slice_ - self.min_sensors[sensor]) / (
                self.max_sensors[sensor] - self.min_sensors[sensor]
            )
            if self.clip:
                slice_ = np.clip(slice_, 0, 1)
            if self.axis == 1:
                trans[:, sensor] = slice_
            elif self.axis == 2:
                trans[:, :, sensor] = slice_
            else:
                raise NotImplemented("Axis for sensor must be 1 or 2")
        return trans

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


# MSE N(0,1) with standardised features?
class StandardiseSensor:
    def __init__(self, num_sensors=11, axis=1):
        self.num_sensors = num_sensors
        self.mean_sensors = []
        self.std_sensors = []
        self.axis = axis

    def fit(self, x):
        self.mean_sensors = []
        self.std_sensors = []
        for sensor in range(self.num_sensors):
            slice_ = np.take(x, sensor, axis=self.axis)
            self.mean_sensors.append(np.mean(slice_))
            self.std_sensors.append(np.std(slice_))

    def transform(self, x):
        trans = np.copy(x)
        for sensor in range(self.num_sensors):
            slice_ = np.take(trans, sensor, axis=self.axis)
            slice_ = (slice_ - self.mean_sensors[sensor]) / self.std_sensors[sensor]
            if self.axis == 1:
                trans[:, sensor] = slice_
            elif self.axis == 2:
                trans[:, :, sensor] = slice_
            else:
                raise NotImplemented("Axis for sensor must be 1 or 2")
        return trans

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


# MSE N(0,1) with standardised features?
class FlattenStandardiseScaler:
    """
    Flattens data and fits standard scaler.
    """

    def __init__(self):
        self.std_scaler = StandardScaler()

    def fit(self, x):
        self.original_shape = x.shape
        flatten_x = x.reshape(x.shape[0], -1)
        self.std_scaler.fit(flatten_x)

    def transform(self, x):
        self.original_shape = x.shape
        flatten_x = x.reshape(x.shape[0], -1)
        new_x = self.std_scaler.transform(flatten_x)
        new_x = new_x.reshape(*self.original_shape)

        return new_x

    def fit_transform(self, x):
        self.original_shape = x.shape
        flatten_x = x.reshape(x.shape[0], -1)
        self.std_scaler.fit(flatten_x)
        new_x = self.std_scaler.transform(flatten_x)
        new_x = new_x.reshape(*self.original_shape)
        return new_x


# MSE N(0,1) with standardised features?
class FlattenMinMaxScaler:
    """
    Flattens data and fits MIN MAX scaler.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, x):
        self.original_shape = x.shape
        flatten_x = x.reshape(x.shape[0], -1)
        self.scaler.fit(flatten_x)

    def transform(self, x):
        self.original_shape = x.shape
        flatten_x = x.reshape(x.shape[0], -1)
        new_x = self.scaler.transform(flatten_x)
        new_x = new_x.reshape(*self.original_shape)

        return new_x

    def fit_transform(self, x):
        self.original_shape = x.shape
        flatten_x = x.reshape(x.shape[0], -1)
        self.scaler.fit(flatten_x)
        new_x = self.scaler.transform(flatten_x)
        new_x = new_x.reshape(*self.original_shape)
        return new_x


def calc_entropy(dwt):
    energy_scale = np.sum(np.abs(dwt))
    t_energy = np.sum(energy_scale)
    prob = energy_scale / t_energy
    w_entropy = -np.sum(prob * np.log(prob))
    return w_entropy


def crest_factor(x):
    return np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x)))


def wavedec_mean(
    trace, wt_type="haar", wt_level=2, wt_summarise=False, wt_partial=False
):
    if wt_partial:
        # res = pywt.wavedec(trace, wt_type, level=wt_level)[:2]
        res = pywt.wavedec(trace, wt_type, level=wt_level)[:2]
    else:
        # res = pywt.wavedec(trace, wt_type, level=wt_level)[:-2]
        res = pywt.wavedec(trace, wt_type, level=wt_level)
    if wt_summarise:
        res = np.array(
            [
                [
                    # i.mean(),
                    np.abs(i).mean(),
                    i.var(),
                    kurtosis(i),
                    skew(i),
                    i.max(),
                    i.min(),
                    np.sqrt(np.sum(i ** 2)) / len(i),  # energy for power
                    crest_factor(i),
                    # i.var(),
                    # calc_entropy(i),
                    # np.ptp(i),
                    # np.sqrt(np.mean(i ** 2))
                ]
                for i in res
            ]
        ).flatten()
    else:
        # res = [
        #     (level - np.min(level)) / (np.max(level) - np.min(level)) for level in res
        # ]
        # res = [(level - np.mean(level)) / np.std(level) for level in res]
        res = np.concatenate(res)
    return np.array(res)


def extract_wt_feats(
    sensor_data, wt_type="haar", wt_level=2, wt_summarise=False, wt_partial=False
):
    final_data = []
    for trace in sensor_data:
        if wt_level == 1:
            feat = np.array([pywt.dwt(trace_ssid, wt_type) for trace_ssid in trace])
        else:
            feat = np.array(
                [
                    wavedec_mean(
                        trace_ssid,
                        wt_type=wt_type,
                        wt_level=wt_level,
                        wt_summarise=wt_summarise,
                        wt_partial=wt_partial,
                    )
                    for trace_ssid in trace
                ]
            )
        final_data.append(feat)
    return np.array(final_data)


class MinMaxSeqSensor:
    # EXPERIMENTAL: for dwt sequeces
    # len dwt is resulting lengths of wavedec
    # temp_dwt = pywt.wavedec(trace_ss, wavelet=wt_type, level=wt_level)
    # len_dwt = np.cumsum([len(dwt_) for dwt_ in temp_dwt])
    def __init__(self, num_sensors=1, clip=True):
        self.num_sensors = num_sensors
        self.clip = clip

    def fit(self, x_train, len_dwt):
        self.len_dwt = len_dwt
        self.dwt_min_max_ss = []
        # for all sensors
        for sensor_i in range(self.num_sensors):
            temp_min_max = []
            for dwt_i, len_ in enumerate(self.len_dwt):
                # fit
                if dwt_i == 0:
                    trunc_traces = x_train[:, sensor_i, :len_]
                else:
                    trunc_traces = x_train[:, sensor_i, self.len_dwt[dwt_i - 1] : len_]
                dwt_max = np.max(trunc_traces)
                dwt_min = np.min(trunc_traces)
                temp_min_max.append([dwt_min, dwt_max])
            self.dwt_min_max_ss.append(temp_min_max)
        self.dwt_min_max_ss = np.array(
            self.dwt_min_max_ss
        )  # (ss_id, dwt_level, min-max)

    def transform(self, x):
        x_temp = x.copy()
        for sensor_i in range(self.num_sensors):
            for dwt_i, len_ in enumerate(self.len_dwt):
                # transform
                dwt_min, dwt_max = self.dwt_min_max_ss[sensor_i, dwt_i]

                if dwt_i == 0:
                    trunc_traces = x_temp[:, sensor_i, :len_]
                    x_temp[:, sensor_i, :len_] = (trunc_traces - dwt_min) / (
                        dwt_max - dwt_min
                    )
                else:
                    trunc_traces = x_temp[:, sensor_i, self.len_dwt[dwt_i - 1] : len_]
                    x_temp[:, sensor_i, self.len_dwt[dwt_i - 1] : len_] = (
                        trunc_traces - dwt_min
                    ) / (dwt_max - dwt_min)
        if self.clip:
            x_temp = np.clip(x_temp, 0, 1)
        return x_temp

    def fit_transform(self, x, len_dwt):
        self.fit(x, len_dwt)
        return self.transform(x)
