import numpy as np


def custom_max(array):
    max_val = array[0]
    for i in range(1, len(array)):
        if array[i] > max_val:
            max_val = array[i]
    return max_val

def custom_mean(array):
    sum_val = 0.0
    for i in range(len(array)):
        sum_val += array[i]
    return sum_val / len(array)

def custom_argmax(array):
    max_index = 0
    max_val = array[0]
    for i in range(1, len(array)):
        if array[i] > max_val:
            max_val = array[i]
            max_index = i
    return max_index

def quality_peak(y_pred):
    peak_ratio = custom_max(y_pred) / custom_mean(y_pred)
    # return peak_ratio * (100 / 2.190064)
    return (1 / (1 + np.e**(-3.5*peak_ratio+7.5)))*(100/0.4499714351078607)

def quality_tail(y_pred):
    end_idx = int(len(y_pred) * 0.2)
    end_mean = custom_mean(y_pred[-end_idx:])
    quality = (1 - (end_mean / (1.1 * custom_mean(y_pred))) ** 2)
    return quality * (100 / 0.33436023529043213)

def quality_base_to_mean(y_pred):
    return (1 - (y_pred[0] / custom_mean(y_pred)) ** 2) * (100 / 0.8831850876454762)

def quality_peak_time(y_pred):
    peak_time = custom_argmax(y_pred)
    num_timeslices = len(y_pred)
    qpt = (num_timeslices - peak_time) / num_timeslices
    return qpt * (100 / 0.9081383928571428)

def quality_ultimate(y_pred):
    peak_ratio = quality_peak(y_pred)
    end_ratio = quality_tail(y_pred)
    base_to_mean = quality_base_to_mean(y_pred)
    peak_time = quality_peak_time(y_pred)

    return peak_ratio * 0.3 + end_ratio * 0.3 + base_to_mean * 0.3 + peak_time * 0.1
