# Changepoint Detection on 2*2 Synthetic Data
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker
import datetime
import getpass

tl.set_backend('numpy')
np.random.seed(42)

print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-08-28 16:38:49")
print(f"Current User's Login: mafi197")
print("-" * 60)

def create_matrix_hankel_tensor(X_matrix, order, window):
    d1, d2, T = X_matrix.shape
    
    if T < window:
        raise ValueError(f"Time series length {T} must be >= window size {window}")
    
    n_snapshots = window - order + 1
    H = np.zeros((d1, d2, order, n_snapshots))
    
    for q in range(order):
        for j in range(n_snapshots):
            H[:, :, q, j] = X_matrix[:, :, q + j]
    
    return H

def _sanitize_rank(tensor_shape, rank):
    if isinstance(rank, int):
        rank = (rank,) * len(tensor_shape)
    rank = tuple(min(tensor_shape[i], rank[i]) for i in range(len(tensor_shape)))
    return rank

class MatrixTensorDMD:
    def __init__(self, order=5, rank=(2,2,3,6)):
        self.order = order
        self.rank = rank

    def reconstruction_error(self, X_matrix, window):
        try:
            H = create_matrix_hankel_tensor(X_matrix, self.order, window)
            
            safe_rank = _sanitize_rank(H.shape, self.rank)
            
            core, factors = tucker(H, rank=safe_rank)
            
            H_hat = tl.tucker_to_tensor((core, factors))
            
            error = np.linalg.norm(H - H_hat) / (np.linalg.norm(H) + 1e-12)
            return error

        except Exception as e:
            print(f"Error in reconstruction: {str(e)}")
            return float('inf')

class AdaptiveEWMA:
    def __init__(self, gamma, ell):
        self.gamma = gamma
        self.ell = ell
        self.mean = 0
        self.variance = 1e-4
        self.statistic = 0
        self.n = 0
        self.detected_change = False

    def update(self, delta):
        self.n += 1

        old_mean = self.mean
        self.mean = old_mean + (delta - old_mean) / min(10, self.n)

        if self.n > 1:
            self.variance = max(1e-4, (1 - 1/min(10, self.n)) * self.variance +
                               (1/max(1, self.n-1)) * (delta - old_mean)**2)

        std = np.sqrt(self.variance)
        normalized_delta = (delta - self.mean) / std

        self.statistic = (1 - self.gamma) * self.statistic + self.gamma * normalized_delta

        if abs(self.statistic) > self.ell and self.n >= 3:
            self.detected_change = True

        return self.detected_change

def single_change_matrix_tensor(data_3d, burn_in, window, order, rank, gamma, ell):
    d1, d2, sequence_length = data_3d.shape

    if sequence_length < window + burn_in:
        print("Time series too short for the specified window and burn-in")
        return None

    tdmd = MatrixTensorDMD(order=order, rank=rank)
    previous_error = None
    ewma = AdaptiveEWMA(gamma, ell)

    for t in range(window, sequence_length):
        X = data_3d[:, :, t-window:t]

        error = tdmd.reconstruction_error(X, window)

        if previous_error is not None:
            delta = error - previous_error
            if t > window + 3:
                if ewma.update(delta) and t >= burn_in:
                    return max(t-1, burn_in)

        previous_error = error

    return None

def multiple_changes_matrix_tensor(data_3d, burn_in, window, order, rank, gamma, ell):
    d1, d2, T = data_3d.shape
    changepoints = []

    start_idx = burn_in

    print(f"Starting changepoint detection with parameters:")
    print(f"  window={window}, order={order}, rank={rank}, gamma={gamma}, ell={ell}, burn_in={burn_in}")

    while start_idx + window < T:
        end_idx = T
        segment_length = end_idx - start_idx

        if segment_length <= window + burn_in:
            break

        segment = data_3d[:, :, start_idx:end_idx]

        cp_idx = single_change_matrix_tensor(segment, burn_in, window, order, rank, gamma, ell)

        if cp_idx is not None:
            global_idx = start_idx + cp_idx

            if not changepoints or global_idx - changepoints[-1] >= 30:
                changepoints.append(global_idx)
                print(f"Detected changepoint at index {global_idx}")

            start_idx = global_idx + 50
        else:
            start_idx += 10

    return changepoints

def optimized_parameter_search(data_3d, t, original_params=None):
    if original_params is None:
        original_params = {
            'window': 25,
            'order': 4,
            'rank': (2, 2, 3, 6),
            'gamma': 0.5,
            'ell': 1.2,
            'burn_in': 40
        }

    window_values = [15, 20, 25, 30]
    order_values = [3, 4, 5]
    gamma_values = [0.4, 0.5, 0.6, 0.7]
    ell_values = [0.9, 1.0, 1.1, 1.2]

    best_params = original_params.copy()
    best_score = 0

    print("\n=== PARAMETER OPTIMIZATION ===")
    print(f"Starting with parameters: {original_params}")

    for window in window_values:
        for order in order_values:
            if order >= window:
                continue

            rank = (2, 2, min(order, 3), min(window-order+1, 6))

            try:
                changepoints = multiple_changes_matrix_tensor(
                    data_3d,
                    original_params['burn_in'],
                    window,
                    order,
                    rank,
                    original_params['gamma'],
                    original_params['ell']
                )

                score = len(changepoints)

                if score > best_score:
                    best_score = score
                    best_params.update({'window': window, 'order': order, 'rank': rank})
                    print(f"Found better parameters: window={window}, order={order}, rank={rank}")
                    print(f"Detected {score} changepoints")
            except Exception as e:
                print(f"Error with window={window}, order={order}: {str(e)}")

    for gamma in gamma_values:
        for ell in ell_values:
            try:
                changepoints = multiple_changes_matrix_tensor(
                    data_3d,
                    original_params['burn_in'],
                    best_params['window'],
                    best_params['order'],
                    best_params['rank'],
                    gamma,
                    ell
                )

                score = len(changepoints)

                if score > best_score:
                    best_score = score
                    best_params.update({'gamma': gamma, 'ell': ell})
                    print(f"Found better parameters: gamma={gamma}, ell={ell}")
                    print(f"Detected {score} changepoints")
            except Exception as e:
                print(f"Error with gamma={gamma}, ell={ell}: {str(e)}")

    print(f"\nOptimized parameters: {best_params}")
    return best_params

def robust_changepoint_detection(data_3d, t):
    print("\n=== ROBUST CHANGEPOINT DETECTION ===")

    params = {
        'window': 15,
        'order': 3,
        'rank': (2, 2, 3, 6),
        'gamma': 0.5,
        'ell': 1.2,
        'burn_in': 40
    }

    matrix_changepoints = multiple_changes_matrix_tensor(
        data_3d,
        params['burn_in'],
        params['window'],
        params['order'],
        params['rank'],
        params['gamma'],
        params['ell']
    )

    if len(matrix_changepoints) < 3:
        print(f"\nFound {len(matrix_changepoints)} changepoints, trying parameter optimization")
        optimized_params = optimized_parameter_search(data_3d, t, params)

        matrix_changepoints = multiple_changes_matrix_tensor(
            data_3d,
            optimized_params['burn_in'],
            optimized_params['window'],
            optimized_params['order'],
            optimized_params['rank'],
            optimized_params['gamma'],
            optimized_params['ell']
        )

    if len(matrix_changepoints) < 3:
        print(f"\nMatrix approach found {len(matrix_changepoints)} changepoints, trying vector approach")

        data_2d = data_3d.reshape(data_3d.shape[0] * data_3d.shape[1], data_3d.shape[2])

        vector_changepoints = vector_based_detection(data_2d, params)

        all_changepoints = list(matrix_changepoints)
        for cp in vector_changepoints:
            if not any(abs(cp - existing) < 30 for existing in all_changepoints):
                all_changepoints.append(cp)

        all_changepoints.sort()
        return all_changepoints

    return matrix_changepoints

def vector_based_detection(data_2d, params):
    window = params['window']
    order = params['order']
    burn_in = params['burn_in']
    gamma = params['gamma']
    ell = params['ell']

    class VectorTensorDMD:
        def __init__(self, order=5, rank=(3,3,6)):
            self.order = order
            self.rank = rank

        def reconstruction_error(self, X):
            num_features, window = X.shape
            n_snapshots = window - self.order + 1
            tensor = np.zeros((self.order, num_features, n_snapshots))

            for i in range(self.order):
                tensor[i] = X[:, i:i + n_snapshots]

            safe_rank = _sanitize_rank(tensor.shape, self.rank)

            core, factors = tucker(tensor, rank=safe_rank)

            tensor_hat = tl.tucker_to_tensor((core, factors))

            error = np.linalg.norm(tensor - tensor_hat) / (np.linalg.norm(tensor) + 1e-12)
            return error

    tdmd = VectorTensorDMD(order=order, rank=(min(data_2d.shape[0], 4), min(data_2d.shape[0], 4), 6))
    previous_error = None
    ewma = AdaptiveEWMA(gamma, ell)
    changepoints = []

    for t in range(window, data_2d.shape[1]):
        X = data_2d[:, t-window:t]

        error = tdmd.reconstruction_error(X)

        if previous_error is not None:
            delta = error - previous_error
            if t > window + 3:
                if ewma.update(delta) and t >= burn_in:
                    changepoints.append(t-1)
                    print(f"Vector approach detected changepoint at index {t-1}")

                    ewma = AdaptiveEWMA(gamma, ell)
                    t += 50

        previous_error = error

    return changepoints

def generate_3d_eeg_data(cp1=2, cp2=6, cp3=8):
    dt = 0.01
    t = np.arange(0, 10, dt)
    data = np.zeros((2, 2, len(t)))

    idx1 = t < cp1
    data[0, 0, idx1] = 1.0 * np.sin(2 * np.pi * 10 * t[idx1])
    data[0, 1, idx1] = 1.0 * np.sin(2 * np.pi * 9 * t[idx1])
    data[1, 0, idx1] = 1.0 * np.sin(2 * np.pi * 11 * t[idx1])
    data[1, 1, idx1] = 1.0 * np.sin(2 * np.pi * 10 * t[idx1])
    noise_level1 = 0.05
    data[:, :, idx1] += noise_level1 * np.random.randn(*data[:, :, idx1].shape)

    idx2 = (t >= cp1) & (t < cp2)
    data[0, 0, idx2] = 1.5 * np.sin(2 * np.pi * 20 * t[idx2])
    data[0, 1, idx2] = 1.5 * np.sin(2 * np.pi * 18 * t[idx2])
    data[1, 0, idx2] = 1.5 * np.sin(2 * np.pi * 22 * t[idx2])
    data[1, 1, idx2] = 1.5 * np.sin(2 * np.pi * 20 * t[idx2])
    noise_level2 = 0.1
    data[:, :, idx2] += noise_level2 * np.random.randn(*data[:, :, idx2].shape)

    idx3 = (t >= cp2) & (t < cp3)
    data[0, 0, idx3] = 1.8 * np.sin(2 * np.pi * 6 * t[idx3])
    data[0, 1, idx3] = 1.8 * np.sin(2 * np.pi * 5 * t[idx3])
    data[1, 0, idx3] = 1.8 * np.sin(2 * np.pi * 7 * t[idx3])
    data[1, 1, idx3] = 1.8 * np.sin(2 * np.pi * 6 * t[idx3])
    noise_level3 = 0.15
    data[:, :, idx3] += noise_level3 * np.random.randn(*data[:, :, idx3].shape)

    idx4 = t >= cp3
    data[0, 0, idx4] = 2.0 * np.sin(2 * np.pi * 40 * t[idx4])
    data[0, 1, idx4] = 2.0 * np.sin(2 * np.pi * 38 * t[idx4])
    data[1, 0, idx4] = 2.0 * np.sin(2 * np.pi * 42 * t[idx4])
    data[1, 1, idx4] = 2.0 * np.sin(2 * np.pi * 40 * t[idx4])
    noise_level4 = 0.2
    data[:, :, idx4] += noise_level4 * np.random.randn(*data[:, :, idx4].shape)

    return data, t, [cp1, cp2, cp3]

def plot_3d_matrix_data(data, t, title, changepoints=None, detected_cps=None):
    plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16)

    channel_names = [['Upper Left', 'Upper Right'],
                     ['Lower Left', 'Lower Right']]

    for i in range(2):
        for j in range(2):
            plt.subplot(2, 2, i*2 + j + 1)
            plt.plot(t, data[i, j], linewidth=1)

            if changepoints is not None:
                for cp in changepoints:
                    plt.axvline(x=cp, color='r', linestyle='--', alpha=0.7, label='True CP')

            if detected_cps is not None:
                for cp in detected_cps:
                    plt.axvline(x=cp, color='g', linestyle='-', alpha=0.7, label='Detected CP')

            plt.title(f'Channel: {channel_names[i][j]}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')

            if i == 0 and j == 0:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def evaluate_changepoints(detected_idx, true_cp_times, t, tol=0.3):
    if not detected_idx:
        return dict(tp=0, fp=0, fn=len(true_cp_times), precision=0, recall=0, f1=0)

    detected_times = t[detected_idx]
    matched = set()
    tp = 0

    for dt in detected_times:
        for j, tt in enumerate(true_cp_times):
            if j in matched:
                continue
            if abs(dt - tt) <= tol:
                tp += 1
                matched.add(j)
                break

    fp = len(detected_times) - tp
    fn = len(true_cp_times) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

def plot_error_dynamics(data_3d, t, window, order, rank, true_cps, detected_cps):
    errors = []
    deltas = []
    time_points = []
    tdmd = MatrixTensorDMD(order=order, rank=rank)
    previous_error = None

    for i in range(window, len(t)):
        X = data_3d[:, :, i-window:i]
        error = tdmd.reconstruction_error(X, window)
        errors.append(error)
        time_points.append(t[i])

        if previous_error is not None:
            deltas.append(error - previous_error)
        else:
            deltas.append(0)

        previous_error = error

    plt.figure(figsize=(12,6))

    plt.subplot(2,1,1)
    plt.plot(time_points, errors)
    for cp in true_cps:
        plt.axvline(cp, color='r', linestyle='--', alpha=0.7)
    for cp in detected_cps:
        plt.axvline(cp, color='g', alpha=0.7)
    plt.title("Reconstruction Error")

    plt.subplot(2,1,2)
    plt.plot(time_points, deltas)
    for cp in true_cps:
        plt.axvline(cp, color='r', linestyle='--', alpha=0.7)
    for cp in detected_cps:
        plt.axvline(cp, color='g', alpha=0.7)
    plt.title("Error Increments")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cp1, cp2, cp3 = 2.3, 3.5, 5.8

    data_3d, t, true_cps = generate_3d_eeg_data(cp1, cp2, cp3)
    print(f"2×2 Matrix Time Series Shape: {data_3d.shape}")
    print(f"True changepoints at times: {[round(cp, 2) for cp in true_cps]}")
    true_cp_indices = [np.argmin(np.abs(t - cp)) for cp in true_cps]
    print(f"True changepoint indices: {true_cp_indices}")

    plot_3d_matrix_data(data_3d, t, "Original 2×2 Matrix-Valued Time Series", changepoints=true_cps)

    changepoints_indices = robust_changepoint_detection(data_3d, t)

    changepoints_times = t[changepoints_indices] if changepoints_indices else []

    print(f"\nDetected changepoints at times: {[round(cp, 2) for cp in changepoints_times]}")
    print(f"Detected changepoint indices: {changepoints_indices}")

    plot_3d_matrix_data(data_3d, t, "2×2 Matrix-Valued Time Series with Detected Changepoints",
                       changepoints=true_cps, detected_cps=changepoints_times)

    window = 15
    order = 3
    rank = (2, 2, 3, 6)
    plot_error_dynamics(data_3d, t, window, order, rank, true_cps, changepoints_times)

    metrics = evaluate_changepoints(changepoints_indices, true_cps, t, tol=0.3)
    print("\n=== CHANGEPOINT DETECTION RESULTS ===")
    print("TRUE CHANGEPOINTS:")
    for i, cp in enumerate(true_cps):
        print(f"  {i+1}. Time: {round(cp,2)}s (Index: {true_cp_indices[i]})")

    print("\nDETECTED CHANGEPOINTS:")
    for i, idx in enumerate(changepoints_indices):
        print(f"  {i+1}. Time: {round(t[idx],2)}s (Index: {idx})")

    print("\nMETRICS:")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  F1 Score: {metrics['f1']:.2f}")

    print("\nDETECTION ACCURACY:")
    for i, true_cp in enumerate(true_cps):
        closest_idx = np.argmin([abs(t[cp] - true_cp) for cp in changepoints_indices])
        closest_cp = t[changepoints_indices[closest_idx]]
        error = closest_cp - true_cp
        error_abs = abs(error)
        error_percent = (error_abs / true_cp) * 100 if true_cp > 0 else float('inf')

        if error_abs <= 0.3:
            status = "DETECTED"
        else:
            status = "MISSED"

        print(f"  True CP {round(true_cp, 2)}s: {status}")
        if status == "DETECTED":
            print(f"    • Detected at: {round(closest_cp, 2)}s")
            print(f"    • Time error: {'+' if error >= 0 else ''}{round(error, 3)}s ({round(error_percent, 1)}%)")
