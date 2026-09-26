"""Figures are saved headlessly, making exploration reproducible."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import write_json_new
from .data import NUMERIC_COLUMNS


def explore_training(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    summary = {'rows': len(frame), 'class_counts': {str(k): int(v) for k,v in frame.Churn.value_counts().items()},
               'missing': frame.isna().sum().to_dict(),
               'numeric': frame[NUMERIC_COLUMNS].describe().astype(object).where(
                   frame[NUMERIC_COLUMNS].describe().notna(), None).to_dict()}
    write_json_new(output_dir/'summary.json', summary)
    fig, axes = plt.subplots(1,3,figsize=(12,3.5), layout='constrained')
    for ax,column in zip(axes, NUMERIC_COLUMNS):
        ax.hist(frame[column].dropna(), bins=25, color='#23677d', edgecolor='white')
        ax.set(xlabel=column, ylabel='Customers')
    fig.suptitle('Training partition only — numeric distributions')
    fig.savefig(output_dir/'numeric_distributions.png', dpi=150)
    plt.close(fig)
    for column in ('Contract', 'InternetService'):
        groups = frame.groupby(column).Churn.agg(['mean','count']).sort_index()
        fig,ax = plt.subplots(figsize=(7,4.5), layout='constrained')
        ax.bar(np.arange(len(groups)), groups['mean'], color='#23677d')
        ax.set_xticks(np.arange(len(groups)), [f'{label}\nn={count}' for label,count in zip(groups.index,groups['count'])])
        ax.set(ylim=(0,1), ylabel='Observed churn fraction', title=f'Training partition — {column}')
        ax.text(0,-0.2,'Association does not establish causation.', transform=ax.transAxes, fontsize=9)
        fig.savefig(output_dir/f'churn_by_{column}.png', dpi=150)
        plt.close(fig)
