import abcpy #pyabc
import matplotlib.pyplot as plt

def plot_history(history, ground_truth):
    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        abcpy.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=1000,
            x='num_value_ag',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()

    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        abcpy.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=1000,
            x='num_momentum_ag',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()

    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        abcpy.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=8000,
            x='num_noise_ag',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()
    
    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        abcpy.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=1,
            x='mm_pov',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()