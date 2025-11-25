import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Configuration
portfolio_list = ["AMZN", "APO", "GEHC", "GOOGL", "NVDA", "UNH", "PYPL", "MSFT"]
risk_free_ticker = "TNX"
benchmark_ticker = "^GSPC"
start_date = "2015-01-01"
end_date = "2025-11-14"

print(f"Downloading Data from {start_date} to {end_date}...")

try:
    # 2. Download Data
    price_data = yf.download(portfolio_list, start=start_date, end=end_date)["Close"]
    rf_data = yf.download(risk_free_ticker, start=start_date, end=end_date)["Close"]
    bench_data = yf.download(benchmark_ticker, start=start_date, end=end_date)["Close"]

    # 3. Clean Data
    if isinstance(bench_data, pd.DataFrame):
        bench_data = bench_data.squeeze()

    if isinstance(rf_data, pd.DataFrame):
        rf_data = rf_data.squeeze()

    price_data = price_data.dropna()
    bench_data = bench_data.dropna()
    rf_data = rf_data.dropna()

    # 4. Calculate Risk-Free Rate
    daily_rf_series = (rf_data / 100) / 252
    avg_daily_rf = daily_rf_series.mean()

    if isinstance(avg_daily_rf, (pd.Series, np.ndarray)):
        avg_daily_rf = avg_daily_rf.item()

    # 5. Calculate Returns
    returns_data = price_data.pct_change().dropna()
    benchmark_returns = bench_data.pct_change().dropna()

    print("Data Processing complete. Starting Analysis...\n")

    
    # 6. OPTIMIZATION PHASE
   
    print("--- Running Portfolio Optimizer ---")

    # A. Covariance Matrix (Annualized)
    cov_matrix = returns_data.cov() * 252


    # B. Define Optimization Function (Negative Sharpe)
    def negative_sharpe(weights):
        # Portfolio Annual Return
        p_ret = np.dot(weights, returns_data.mean()) * 252
        # Portfolio Volatility (Sqrt of Weights * Covariance * Weights)
        p_var = weights.T @ cov_matrix @ weights
        p_vol = np.sqrt(p_var)
        # Return Negative Sharpe
        return - ((p_ret - (avg_daily_rf * 252)) / p_vol)


    # C. Constraints & Bounds
    # Sum of weights must equal 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Weights must be between 0% and 25% (0.25)
    bounds = tuple((0, 0.25) for _ in range(len(portfolio_list)))
    # Initial guess (equal weights)
    init_guess = [1 / len(portfolio_list)] * len(portfolio_list)

    # D. Run Optimizer
    opt_results = minimize(
        negative_sharpe,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # E. Store Optimal Results
    optimal_weights = opt_results.x

    # Print Optimal Weights nicely
    print("\nOptimal Weights Found:")
    for ticker, weight in zip(portfolio_list, optimal_weights):
        if weight > 0.001:  # Only print if weight is significant
            print(f"{ticker}: {weight:.2%}")

    # F. Create "Optimal Portfolio" Returns Column
    opt_portfolio_returns = (returns_data * optimal_weights).sum(axis=1)

    # G. Calculate Metrics for Optimal Portfolio
    opt_volatility = opt_portfolio_returns.std() * np.sqrt(252)

    # Regression (Beta/Alpha)
    opt_reg_df = pd.DataFrame({'Port': opt_portfolio_returns, 'Bench': benchmark_returns}).dropna()
    slope, intercept, r, p, std_err = stats.linregress(opt_reg_df['Bench'], opt_reg_df['Port'])
    opt_beta = slope
    opt_alpha = intercept * 252
    opt_r_sq = r ** 2

    # Sharpe
    opt_mean_ret = opt_portfolio_returns.mean() * 252
    opt_sharpe = (opt_mean_ret - (avg_daily_rf * 252)) / opt_volatility

    # VaR
    opt_var_99 = (opt_portfolio_returns.mean() + (-2.33 * opt_portfolio_returns.std())) * np.sqrt(21)

    print("\n--- Optimal Portfolio Metrics ---")
    print(f"Beta: {opt_beta:.4f} | Alpha: {opt_alpha:.4f} | Sharpe: {opt_sharpe:.4f}")
    print(f"Volatility: {opt_volatility:.4f} | VaR (99%): {opt_var_99:.4f}")
    print("-" * 40 + "\n")

    
    # 7. MONTE CARLO SIMULATION & CHART
   
    print("--- Running Monte Carlo Simulation ---")

    # 1. Setup
    simulations = 10000
    start_capital = 10000
    time_horizons = [0.25, 0.5, 1.0]
    horizon_names = ["3 Months", "6 Months", "1 Year"]
    monte_carlo_results = []

    # --- CHARTING SECTION ---
    print("\nGenerating Monte Carlo Chart (1 Year)...")

    sims_to_chart = 500
    trading_days = 252

    daily_mu = opt_mean_ret / 252
    daily_sigma = opt_volatility / np.sqrt(252)

    price_paths = np.zeros((trading_days + 1, sims_to_chart))
    price_paths[0] = start_capital

    for t in range(1, trading_days + 1):
        Z = np.random.normal(0, 1, sims_to_chart)
        daily_factor = np.exp((daily_mu - 0.5 * daily_sigma ** 2) + daily_sigma * Z)
        price_paths[t] = price_paths[t - 1] * daily_factor

    plt.figure(figsize=(12, 6))
    plt.plot(price_paths, color='blue', alpha=0.1, linewidth=0.5)
    plt.title(f'Monte Carlo Simulation: {sims_to_chart} Possible Paths over 1 Year')
    plt.xlabel('Trading Days (0 to 252)')
    plt.ylabel('Portfolio Value ($)')
    plt.axhline(y=start_capital, color='r', linestyle='--', label="Start Value")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    # --- STATISTICS SECTION ---
    for T, name in zip(time_horizons, horizon_names):
        random_shocks = np.random.normal(0, 1, simulations)
        drift = (opt_mean_ret - 0.5 * opt_volatility ** 2) * T
        shock = opt_volatility * np.sqrt(T) * random_shocks

        simulated_end_values = start_capital * np.exp(drift + shock)

        percentile_5 = np.percentile(simulated_end_values, 5)
        percentile_50 = np.percentile(simulated_end_values, 50)
        percentile_95 = np.percentile(simulated_end_values, 95)

        monte_carlo_results.append({
            "Horizon": name,
            "Worst Case (5%)": percentile_5,
            "Expected (50%)": percentile_50,
            "Best Case (95%)": percentile_95
        })

    mc_df = pd.DataFrame(monte_carlo_results)
    pd.options.display.float_format = '${:,.2f}'.format
    print(f"\nSimulation Results (Start Capital: ${start_capital:,.0f}):")
    print(mc_df)
    pd.options.display.float_format = '{:.4f}'.format

   
    # 8. INDIVIDUAL TICKER ANALYSIS
   
    results_list = []

    for ticker in returns_data.columns:
        # Volatility
        annual_volatility = returns_data[ticker].std() * np.sqrt(252)

        # Regression
        regression_df = pd.DataFrame({
            'Ticker': returns_data[ticker],
            'Benchmark': benchmark_returns
        }).dropna()

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            regression_df['Benchmark'],
            regression_df['Ticker']
        )

        beta = slope
        alpha = intercept * 252
        r_squared = r_value ** 2

        # Sharpe
        ticker_annual_return = returns_data[ticker].mean() * 252
        avg_annual_rf = avg_daily_rf * 252
        sharpe = (ticker_annual_return - avg_annual_rf) / annual_volatility

        # VaR
        ticker_mean = returns_data[ticker].mean()
        ticker_std = returns_data[ticker].std()
        var_99 = (ticker_mean + (-2.33 * ticker_std)) * np.sqrt(21)

        results_list.append({
            'Ticker': ticker,
            'Beta': beta,
            'Alpha': alpha,
            'R_Squared': r_squared,
            'Volatility': annual_volatility,
            'Sharpe_Ratio': sharpe,
            'VaR_99_Month': var_99
        })

    final_df = pd.DataFrame(results_list)
    pd.options.display.float_format = '{:.4f}'.format

    print("--- Individual Ticker Analysis ---")
    print(final_df)

    print("\n--- Correlation Matrix ---")
    print(returns_data.corr())


    print("\nSaving results to Excel...")

    # A. Optimal Weights DataFrame
    weights_df = pd.DataFrame({
        'Ticker': portfolio_list,
        'Optimal_Weight': optimal_weights
    })
    weights_df = weights_df.sort_values(by='Optimal_Weight', ascending=False)

    # B. Optimized Portfolio Metrics
    port_metrics_df = pd.DataFrame([{
        'Beta': opt_beta,
        'Alpha': opt_alpha,
        'R_Squared': opt_r_sq,
        'Volatility': opt_volatility,
        'Sharpe_Ratio': opt_sharpe,
        'VaR_99': opt_var_99
    }], index=['Total_Portfolio'])

    # C. Correlation Matrix
    correlation_matrix = returns_data.corr()

    # Write to Excel
    output_filename = "Portfolio_Analysis.xlsx"

    with pd.ExcelWriter(output_filename) as writer:
        weights_df.to_excel(writer, sheet_name="Optimal_Weights", index=False)
        port_metrics_df.to_excel(writer, sheet_name="Portfolio_Summary")
        final_df.to_excel(writer, sheet_name="Ticker_Metrics", index=False)
        correlation_matrix.to_excel(writer, sheet_name="Correlation")

    print(f"File saved successfully as: {output_filename}")

except PermissionError:
    print(f"\nERROR: Could not save Excel file. Please close '{output_filename}' and try again.")
except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")