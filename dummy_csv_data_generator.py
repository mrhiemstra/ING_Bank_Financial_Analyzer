"""
Copyright (c) 2025 M.R. Hiemstra

This file is part of Financial Analyzer, a simple tool for analyzing personal financial transactions.

Licensed under the MIT License. See LICENSE file in the project root for full license information.

Description:
    Generates synthetic daily financial transaction data from 2015 to 2025.
    Includes salaries, random income, expenses, savings, and investments.
    Output: dummy_financial_data.csv (semicolon-separated, UTF-8)
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def generate_transaction_row(date: pd.Timestamp, amount: float, af_bij: str, description: str) -> Dict[str, str]:
    amount_str = f"{amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return {
        'Datum': date.strftime('%Y%m%d'),
        'Bedrag (EUR)': amount_str,
        'Af Bij': af_bij,
        'Mededelingen': description
    }


def simulate_financial_data(start_date: str = '2015-01-01', end_date: str = '2025-12-31') -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    salary_days = pd.date_range(start=start_date, end=end_date, freq='MS')  # 1st of each month
    data: List[Dict[str, str]] = []

    for date in dates:
        transactions = []

        # Monthly salary
        if date in salary_days:
            salary = np.random.normal(loc=2000, scale=100)
            transactions.append(generate_transaction_row(date, salary, 'Bij', 'Salaris maand'))

        # Random bonus/gift (1% chance)
        if np.random.rand() < 0.01:
            bonus = np.random.normal(loc=500, scale=100)
            transactions.append(generate_transaction_row(date, bonus, 'Bij', 'Vergoeding of gift'))

        # Random expense (50% chance)
        if np.random.rand() < 0.5:
            spending = np.random.exponential(scale=50)
            description = np.random.choice([
                'Boodschappen', 'Huur', 'Uit eten', 'Benzine', 'Kleding', 'Entertainment'
            ])
            transactions.append(generate_transaction_row(date, spending, 'Af', description))

        # Savings transfer (5% chance)
        if np.random.rand() < 0.05:
            savings = np.random.uniform(50, 200)
            direction = np.random.choice(['Naar spaarrekening', 'Van spaarrekening'], p=[0.7, 0.3])
            af_bij = 'Af' if direction.startswith('Naar') else 'Bij'
            transactions.append(generate_transaction_row(date, savings, af_bij, direction))

        # Investment transfer (3% chance)
        if np.random.rand() < 0.03:
            investment = np.random.uniform(100, 500)
            transactions.append(generate_transaction_row(date, investment, 'Af', 'Belegging'))

        data.extend(transactions)

    return pd.DataFrame(data, columns=['Datum', 'Bedrag (EUR)', 'Af Bij', 'Mededelingen'])


def save_to_csv(df: pd.DataFrame, filename: str = 'dummy_financial_data.csv') -> None:
    df.to_csv(filename, sep=';', index=False, encoding='utf-8')
    print(f"Dummy financial data CSV generated: {filename}")


def main():
    df = simulate_financial_data()
    save_to_csv(df)


if __name__ == '__main__':
    main()