import numpy as np


def normalized_mean_square_error(outputs, targets):
    """
    Calculate the normalized mean square error (NMSE) between two 4D arrays along dim 0.

    Parameters:
    outputs (np.ndarray): The model's output 4D array.
    targets (np.ndarray): The target or reference 4D array.

    Returns:
    float: The NMSE value.
    """
    if outputs.shape != targets.shape:
        raise ValueError("The input arrays must have the same shape.")

    # Calculate the MSE along the first dimension
    mse = np.sum((outputs - targets) ** 2, axis=(1, 2, 3))

    # Calculate the variance of the target array
    power = np.sum(targets ** 2, axis=(1, 2, 3))

    # Avoid division by zero by adding a small epsilon where variance is zero
    epsilon = 1e-10
    nmse = mse / (power + epsilon)

    # Calculate the mean of the NMSE values to get the final NMSE
    mean_nmse = np.mean(nmse)

    return mean_nmse


def cosine_similarity(predicted_outputs, target_outputs):
    target_complex = convert_to_complex(target_outputs)
    predicted_complex = convert_to_complex(predicted_outputs)

    target_norm = np.sqrt(np.sum(np.abs(np.conj(target_complex) * target_complex), axis=1))
    target_norm = target_norm.astype('float32')
    predicted_norm = np.sqrt(np.sum(np.abs(np.conj(predicted_complex) * predicted_complex), axis=1))
    predicted_norm = predicted_norm.astype('float32')

    dot_product = np.abs(np.sum(np.conj(target_complex) * predicted_complex, axis=1))

    average_similarity = np.mean(dot_product / (target_norm * predicted_norm), axis=1)

    rho = np.mean(average_similarity)

    return rho


def convert_to_complex(input_array):
    real_part = input_array[:, 0, :, :]
    imaginary_part = input_array[:, 1, :, :]
    complex_array = real_part + 1j * imaginary_part

    return complex_array
