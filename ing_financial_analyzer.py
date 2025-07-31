# ing_financial_analyzer.py
"""
Copyright (c) 2025 M.R. Hiemstra

This file is part of the ING Financial Analyzer, a simple tool for analyzing personal financial transactions.

Licensed under the MIT License. See LICENSE file in the project root for full license information.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tabulate import tabulate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
from typing import List, Dict, Optional


def validate_csv(filename: str) -> None:
    if not filename.endswith(".csv"):
        raise NameError("The provided file is not a CSV file. Please provide a valid CSV file.")


def find_column(df: pd.DataFrame, possible_names: List[str]) -> str:
    for name in df.columns:
        if name.strip().lower() in [n.strip().lower() for n in possible_names]:
            return name
    raise ValueError(f"None of the possible column names {possible_names} found in CSV.")


def preprocess_dataframe(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    df[column_map['date']] = pd.to_datetime(df[column_map['date']], format="%Y%m%d", errors='coerce')
    df['year_month'] = df[column_map['date']].dt.to_period("M")

    df[column_map['amount']] = (
        df[column_map['amount']]
        .astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    df['spending'] = df.apply(
        lambda row: row[column_map['amount']] if row[column_map['debit_credit']].strip().lower() == 'af' else 0,
        axis=1
    )
    df['earnings'] = df.apply(
        lambda row: row[column_map['amount']] if row[column_map['debit_credit']].strip().lower() == 'bij' else 0,
        axis=1
    )
    df['description_lower'] = df[column_map['description']].str.lower()
    return df


def extract_salary(group: pd.DataFrame, column_map: Dict[str, str]) -> Optional[float]:
    bij_rows = group[group[column_map['debit_credit']].str.lower() == 'bij']
    if bij_rows.empty:
        return None
    keywords = ['salaris', 'salary', 'loon', 'payroll']
    salary_rows = bij_rows[
        bij_rows[column_map['description']].str.lower().str.contains('|'.join(keywords), na=False)
    ]
    return salary_rows[column_map['amount']].max() if not salary_rows.empty else bij_rows[column_map['amount']].max()


def compute_monthly_summary(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    monthly = df.groupby('year_month').agg({'spending': 'sum', 'earnings': 'sum'})
    monthly['salary'] = df.groupby('year_month').apply(
        lambda group: extract_salary(group, column_map), include_groups=False
    )    
    def filter_sum(keywords, debit_credit_value):
        return df[
            (df[column_map['debit_credit']].str.lower() == debit_credit_value) &
            (df['description_lower'].str.contains('|'.join(keywords), na=False))
        ].groupby('year_month')[column_map['amount']].sum()

    monthly['to_savings'] = filter_sum([r'naar .*spaarrekening'], 'af')
    monthly['from_savings'] = filter_sum([r'van .*spaarrekening'], 'bij')
    monthly['savings'] = filter_sum(['spaarrekening', 'savings', 'spaar', 'sparen'], 'af')
    monthly['investment'] = filter_sum(['belegging', 'investment', 'beleggingen'], 'af')

    monthly = monthly.fillna(0)
    monthly['cumulative_savings'] = (monthly['savings'] - monthly['from_savings']).cumsum()
    monthly['cumulative_investment'] = monthly['investment'].cumsum()
    monthly['net_balance'] = monthly['earnings'] - monthly['spending']
    monthly['adjusted_earnings'] = monthly['earnings'] - monthly['from_savings']
    monthly['adjusted_spending'] = monthly['spending'] - monthly['savings'] - monthly['investment']

    monthly['earnings_trend'] = rolling_trend_interpolated(monthly['adjusted_earnings'])
    monthly['spending_trend'] = rolling_trend_interpolated(monthly['adjusted_spending'])

    return monthly


def rolling_trend_interpolated(series: pd.Series, window: int = 3, z_thresh: float = 1.5) -> pd.Series:
    z_scores = (series - series.mean()) / series.std()
    cleaned = series.where(z_scores.abs() < z_thresh)
    interpolated = cleaned.interpolate(method='linear', limit_direction='both')
    return interpolated.rolling(window=window, center=True, min_periods=1).mean()


def plot_summary(monthly: pd.DataFrame, filename: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(24, 18), sharex=True)
    axes = axes.flatten()
    xtick_labels = monthly.index.astype(str)
    xtick_positions = np.arange(len(xtick_labels))
    skip = max(1, len(xtick_labels) // 18)
    xtick_show = xtick_positions[::skip]

    def set_axis_format(ax, title, ylabel='EUR'):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xtick_show)
        ax.set_xticklabels(xtick_labels[xtick_show], rotation=45)
        ax.grid(axis='both', linestyle='--', alpha=0.5)

    # Plot definitions...
    axes[0].bar(xtick_labels, monthly['salary'], label='Salary', alpha=0.7)
    axes[0].bar(xtick_labels, monthly['adjusted_earnings'], label='Income (excl. savings)', alpha=0.4)
    axes[0].plot(xtick_labels, monthly['earnings_trend'], '--', color='black', label='Trend')
    set_axis_format(axes[0], 'Monthly Salary & Income')
    axes[0].legend()

    axes[1].bar(xtick_labels, monthly['adjusted_spending'], color='tomato')
    axes[1].plot(xtick_labels, monthly['spending_trend'], '--', color='black', label='Trend')
    set_axis_format(axes[1], 'Monthly Expenses')
    axes[1].legend()

    axes[2].bar(xtick_labels, monthly['net_balance'], color='seagreen')
    set_axis_format(axes[2], 'Net Balance Change')
    axes[2].legend(['Net Balance'])

    axes[3].bar(xtick_labels, monthly['investment'], color='royalblue')
    axes[3].plot(xtick_labels, monthly['cumulative_investment'], color='orange', marker='o')
    set_axis_format(axes[3], 'Investments')
    axes[3].legend(['Monthly', 'Cumulative'])

    axes[4].bar(xtick_labels, monthly['savings'], label='Savings', color='slateblue')
    axes[4].plot(xtick_labels, monthly['cumulative_savings'], color='gold', marker='o')
    axes[4].bar(xtick_labels, monthly['from_savings'], label='From Savings', color='lightcoral', alpha=0.5)
    set_axis_format(axes[4], 'Savings')
    axes[4].legend()

    axes[5].plot(xtick_labels, monthly['earnings_trend'], label='Income Trend', color='mediumseagreen')
    axes[5].plot(xtick_labels, monthly['spending_trend'], label='Spending Trend', color='tomato')
    set_axis_format(axes[5], 'Income vs Spending Trends')
    axes[5].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Financial Trends: {filename}', fontsize=16)
    plt.subplots_adjust(top=0.92, bottom=0.08)

    try:
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')
    except:
        pass

    plt.show()


def main(filename: str) -> None:
    validate_csv(filename)
    df = pd.read_csv(filename, sep=';')

    print("Detected columns:", df.columns.tolist())

    column_map = {
        'date': find_column(df, ['Datum', 'Date']),
        'amount': find_column(df, ['Bedrag (EUR)', 'Amount (EUR)', 'Bedrag', 'Amount']),
        'debit_credit': find_column(df, ['Af Bij', 'Debit Credit', 'AfBij']),
        'description': find_column(df, ['Mededelingen', 'Description', 'Omschrijving']),
    }

    df = preprocess_dataframe(df, column_map)
    monthly = compute_monthly_summary(df, column_map)

    pd.set_option('display.max_columns', None)
    print(tabulate(monthly[['salary', 'adjusted_earnings', 'adjusted_spending',
                            'earnings_trend', 'spending_trend']],
                   headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))

    plot_summary(monthly, filename)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python financial_analyzer.py <filename.csv>")
        sys.exit(1)
    main(sys.argv[1])