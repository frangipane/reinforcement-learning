import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_test_returns(returns_path_dir, returns_path_fname):
    df = pd.read_table(os.path.join(returns_path_dir, returns_path_fname))
    x = df.Epoch
    y = df.AverageTestEpRet
    std = df.StdTestEpRet

    fig, ax = plt.subplots(1,1)
    ax.plot(x, y)
    ax.fill_between(x, y-std, y+std, alpha=0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('average return')
    ax.set_title('Average returns per epoch for test agent')
    #plt.show()
    fig.savefig(os.path.join(returns_path_dir, 'avg_test_returns'), fmt='png')
