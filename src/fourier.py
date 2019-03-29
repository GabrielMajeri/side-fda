import numpy as np

SIGNAL_COUNT = 1605

def merge_ranges(*ranges):
    """Returns an iterator which goes through multiple ranges, one-by-one."""
    for rng in ranges:
        yield from rng

def generate_location_signals():
    mesh_size = np.ceil(np.sqrt(2 * SIGNAL_COUNT))

    mesh_y, mesh_x = np.mgrid[0:(mesh_size), (-mesh_size + 1):(mesh_size)]

    mesh_y, mesh_x = mesh_y.flatten(), mesh_x.flatten()

    mesh_dist = (mesh_y ** 2 + mesh_x ** 2) + mesh_y / mesh_size + mesh_x / (mesh_size ** 2)
    mesh_dist = mesh_dist + SIGNAL_COUNT * np.float32((mesh_y == 0) & (mesh_x < 0))
    mesh_dist_sorted = np.sort(mesh_dist)

    location_signals = np.zeros((SIGNAL_COUNT, 2))

    for i in range(SIGNAL_COUNT):
        location = np.where(mesh_dist == mesh_dist_sorted[i])

        location_signals[i] = mesh_y[location], mesh_x[location]

    return location_signals

def frequency_domain_combine(candidates, weight_term, bias_term):
    location_signals = generate_location_signals()

    candidate_size = candidates[0].shape

    index_candidate = np.concatenate((
        np.zeros(7, dtype=np.int), range(17, 0, -2),
        np.zeros(7, dtype=np.int), range(18, 0, -2)
    ))
    fft_map = np.zeros((*candidate_size, 32), dtype=np.complex)

    for index_scale in merge_ranges(range(7, 16), range(23, 32)):
        transformed = np.fft.fft2(candidates[index_candidate[index_scale] - 1])
        shifted = np.fft.fftshift(transformed)
        fft_map[:, :, index_scale] = shifted

    center = np.round(np.array(candidate_size) / 2 + 1)

    weight_map_real = np.ones_like(fft_map) / 2
    weight_map_imag = np.ones_like(fft_map) / 2

    indices = np.array(tuple(merge_ranges(range(0, 15), range(16, 31))), dtype=np.int)
    weight_map_real[:, :, indices] = 0
    weight_map_imag[:, :, indices] = 0

    for index_signal in range(SIGNAL_COUNT):
        location_signal = location_signals[index_signal]

        for index_scale in merge_ranges(range(7, 16), range(23, 32)):
            row = int(center[0] + location_signal[0])
            col = int(center[1] + location_signal[1])
            weight_map_real[row, col, index_scale] = weight_term[index_signal, 0, index_scale]

            row = int(center[0] - location_signal[0])
            col = int(center[1] - location_signal[1])
            weight_map_real[row, col, index_scale] = weight_term[index_signal, 0, index_scale]

            row = int(center[0] + location_signal[0])
            col = int(center[1] + location_signal[1])
            weight_map_imag[row, col, index_scale] = weight_term[index_signal, 1, index_scale]

            row = int(center[0] - location_signal[0])
            col = int(center[1] - location_signal[1])
            weight_map_imag[row, col, index_scale] = weight_term[index_signal, 1, index_scale]

        row = int(center[0] + location_signal[0])
        col = int(center[1] + location_signal[1])
        fft_map[row, col, indices] = fft_map[row, col, indices] - bias_term[index_signal, indices]

        row = int(center[0] - location_signal[0])
        col = int(center[1] - location_signal[1])
        fft_map[row, col, indices] = fft_map[row, col, indices] - bias_term[index_signal, indices]

    real_part = np.real(fft_map) * weight_map_real
    imag_part = np.imag(fft_map) * weight_map_imag

    fft_map_new = np.sum(real_part + 1j * imag_part, axis=2)

    return np.real(np.fft.ifft2(np.fft.fftshift(fft_map_new)))
