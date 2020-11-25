import matplotlib.pyplot as plt
import numpy as np


def rolling_smooth(data, window_size, mode='mean'):
    # rolling/moving average (specifically, this is a trailing average. i.e. the mean of (s_t) = mean(s_t-n, ..., s_t-1, s_t)
    # ok for simple moving average (SMA)
    assert data.ndim == 1
    # generate a kernel e.g. size 3 kernel = [1., 1., 1.]
    kernel = np.ones(window_size)
    # numerator is the convolved (? why not correlate)
    # denominator is the element counter
    smooth_data_mean = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    if mode == 'mean':
        # return the smoothes data and abandon the tail
        return smooth_data_mean[: -window_size + 1]
    elif mode == 'mean-var':
        # compute the estimated E(X^2)
        smooth_data_2_mean = np.convolve(data**2, kernel) / np.convolve(np.ones_like(data**2), kernel)
        # compute the estimated varinace: Var(X) = E(X^2) - (E(X))^2
        smooth_data_var = smooth_data_2_mean - smooth_data_mean ** 2
        return smooth_data_mean[: -window_size + 1], smooth_data_var[: -window_size + 1]
    elif mode == 'mean-std':
        smooth_data_2_mean = np.convolve(data**2, kernel) / np.convolve(np.ones_like(data**2), kernel)
        # compute the standard deviation
        smooth_data_std = np.sqrt(np.abs(smooth_data_2_mean - smooth_data_mean ** 2))
        return smooth_data_mean[: -window_size + 1], smooth_data_std[: -window_size + 1]
    else:
        raise Exception(f"Invalid mode input. Expect mean, mean-var, or mean-std, but get {mode}")


def rolling_variance(data, window_size):
    assert data.ndim == 1
    # first generate the kernel
    kernel = np.ones(window_size)
    # Second compute the moving average
    sma = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    # Third compute the sma^2
    data_2 = data ** 2
    sma_2 = np.convolve(data_2, kernel) / np.convolve(np.ones_like(data_2), kernel)
    var = sma_2 - sma ** 2
    return var


def plot_compared_learning_curve(data_list, win_size, plot_configs):
    smoothed_data = {'mean': [], 'var': []}
    for data in data_list:
        d_mean, d_var = rolling_smooth(data, win_size, mode='mean-std')
        smoothed_data['mean'].append(d_mean)
        smoothed_data['var'].append(d_var)

    # set the settings
    curve_num = len(data_list)
    # plot the learning curves
    fig, ax = plt.subplots(1, figsize=(plot_configs['width'], plot_configs['height']))
    ax.set_title(plot_configs['title'], fontsize=plot_configs['font_size'])
    for i in range(curve_num):
        mu = smoothed_data['mean'][i]
        var = smoothed_data['var'][i]
        t = np.arange(mu.shape[0])
        ax.plot(t, mu, lw=2, label=plot_configs['legend'][i], color=plot_configs['color'][i])
        ax.fill_between(t, mu + var, mu - var, lw=2, facecolor=plot_configs['color'][i], alpha=0.5)
        ax.legend(loc=plot_configs['legend_pos'], fontsize=plot_configs['font_size'])
        ax.set_xlabel(plot_configs['x_label'], fontsize=plot_configs['font_size'])
        ax.set_ylabel(plot_configs['y_label'], fontsize=plot_configs['font_size'])
    ax.grid()
    #     plt.savefig('./corl_results/camera-ready/' + plot_configs['title'] + '.png', dpi=100)
    plt.show()


def load_data(root_path, file_names):
    data_all = []
    for file in file_names:
        file = root_path + file
        data = np.load(file, allow_pickle=True)
        data_all.append(data)
    return data_all


if __name__ == '__main__':
    root_dir = './'
    file_names = ['returns.npy', 'w3_returns.npy'
                  ]

    data_list = load_data(root_dir, file_names)

    plot_configs = {
        'width': 14,
        'height': 8,
        'x_label': 'episode',
        'y_label': 'discounted return',
        'font_size': 20,
        'title': 'vanilla and double DQN: CartPole-v0',
        'color': ['tab:purple', 'tab:orange'],
        'legend': ['1-worker', '3-workers'],
        'legend_pos': 'lower right'
    }

    win_size = 100

    plot_compared_learning_curve(data_list, win_size, plot_configs)