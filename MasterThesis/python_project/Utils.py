import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import time

def get_pairs(form_ret, n_pairs):
    # replace first return with 0 so cum returns start at 1
    form_ret.iloc[0, :] = 0

    # create object with cumulative returns
    prc = (form_ret + 1.).cumprod()

    # define number of stocks and total number of pairs
    num_stocks = prc.shape[1]
    total_pairs = int(prc.shape[1] * (prc.shape[1] - 1) / 2)

    # create column with pair combinations to calculate distances
    pairs = np.asarray(list(itertools.permutations(prc.columns, 2)))

    # keep only one set of combinations
    pairs = pd.DataFrame(pairs[pairs[:, 1] > pairs[:, 0]], columns=["leg_1", "leg_2"])

    # calculate distances between normalized prices
    pairs_dist = pairwise_distances(prc.transpose(), prc.transpose())
    pairs_dist = pd.Series(pairs_dist[np.triu_indices(num_stocks, k=1)])
    pairs["dist"] = pairs_dist

    # remove pairs with 0 distance if any
    pairs = pairs[pairs.dist > 0]

    # order according to distance and select pairs
    pairs = pairs.sort_values("dist", ascending=True)
    pairs = pairs.loc[pairs.index[0:min(n_pairs, pairs.shape[0])]]

    # for these pairs, store the standard deviation of the spread
    pairs["spread_std"] = np.std(np.asarray(prc.loc[:, pairs.leg_1]) - np.asarray(prc.loc[:, pairs.leg_2]), axis=0,
                                 ddof=1)
    pairs.index = np.arange(pairs.shape[0])
    
    # returns selected pairs
    return pairs
import numpy as np
import pandas as pd
import itertools

import itertools
import numpy as np
import pandas as pd

def get_pairs_triplet_heron(
    form_ret: pd.DataFrame,
    n_pairs: int,
    index_col: str = "NIFTY50",
    require_complete: bool = True,
) -> pd.DataFrame:
    """
    Select pairs by minimizing Heron's triangle area built from L2 distances
    among (stock i, stock j, index). Returns top n_pairs with spread_std.

    Parameters
    ----------
    form_ret : DataFrame
        Formation-window daily returns in decimal (Date index, columns=tickers incl. index_col).
    n_pairs : int
        Number of pairs to return.
    index_col : str
        Column name for the market index in form_ret.
    require_complete : bool
        If True, restrict to columns fully non-NaN in the formation window.
        If False, distances are computed on overlapping (non-NaN) dates.

    Returns
    -------
    DataFrame with columns:
        leg_1, leg_2, area (Heron), spread_std
    """
    if index_col not in form_ret.columns:
        raise KeyError(f"index_col '{index_col}' not found in form_ret columns")

    # Work on a copy; do not mutate caller
    ret = form_ret.copy()

    # Optionally enforce complete data over the formation window
    if require_complete:
        complete_cols = ret.columns[ret.notna().all(axis=0)]
        if index_col not in complete_cols:
            raise ValueError(f"Index column '{index_col}' has missing values in formation window.")
        ret = ret[complete_cols]

    # Build price paths and normalize each series to start at 1.0
    prc = (1.0 + ret).cumprod()

    if require_complete:
        # Safe if no NaNs: divide all columns by their first-row value
        prc = prc / prc.iloc[0]
    else:
        # Normalize each series by its own first valid observation
        def _normalize(s: pd.Series) -> pd.Series:
            fv = s.first_valid_index()
            if fv is None or s.loc[fv] == 0:
                return s
            return s / s.loc[fv]
        prc = prc.apply(_normalize, axis=0)

    # Separate stocks & index
    stocks = [c for c in prc.columns if c != index_col]
    if len(stocks) < 2:
        raise ValueError("Not enough valid stocks in the formation window to form pairs.")

    idx_series = prc[index_col]

    # L2 (Euclidean) distance on overlapping dates
    def l2_distance(a: pd.Series, b: pd.Series) -> float:
        ab = pd.concat([a, b], axis=1).dropna()
        if ab.shape[0] == 0:
            return np.inf
        diff = ab.iloc[:, 0].to_numpy() - ab.iloc[:, 1].to_numpy()
        return float(np.sqrt(np.sum(diff * diff)))

    # Heron's formula with numeric safeguards
    def heron_area(a: float, b: float, c: float) -> float:
        s = 0.5 * (a + b + c)
        # Guard against tiny negatives due to floating-point
        val = s * max(s - a, 0.0) * max(s - b, 0.0) * max(s - c, 0.0)
        return float(np.sqrt(max(val, 0.0)))

    rows = []
    for i, j in itertools.combinations(stocks, 2):
        si, sj = prc[i], prc[j]
        dij  = l2_distance(si, sj)
        diM  = l2_distance(si, idx_series)
        djM  = l2_distance(sj, idx_series)

        area = heron_area(dij, diM, djM)

        # Spread std over formation window (normalized prices), robust to NaNs
        spread_std = float((si - sj).dropna().std(ddof=1))

        rows.append({
            "leg_1": i,
            "leg_2": j,
            "area": area,
            "spread_std": spread_std
        })

    pairs = (
        pd.DataFrame(rows)
        .sort_values("area", ascending=True)
        .head(n_pairs)
        .reset_index(drop=True)
    )
    return pairs


# def get_pairs_triplet_heron(form_ret: pd.DataFrame,
def get_pairs_triplet_heron_0(
    form_ret: pd.DataFrame,
    n_pairs: int,
    index_col: str = "NIFTY50",
    require_complete: bool = True,
) -> pd.DataFrame:
    """
    Select pairs using an L1 area-like metric that rewards path closeness
    among stock i, stock j, and the index:
        score(i,j) = L1(si, sj) + 0.5 * [ L1(si, idx) + L1(sj, idx) ]
    where L1(a,b) = sum_t |a_t - b_t| on the overlapping dates.

    Parameters
    ----------
    form_ret : DataFrame
        Formation-window daily returns in decimal (Date index, columns=tickers incl. index_col).
    n_pairs : int
        Number of pairs to return.
    index_col : str
        Column name of the market index series within form_ret.
    require_complete : bool
        If True, restrict to columns fully non-NaN over the formation window.
        If False, distances are computed on overlapping (non-NaN) dates.

    Returns
    -------
    DataFrame with columns:
        leg_1, leg_2, area_metric, spread_std
    """
    #print('col names are ', form_ret.columns) 
    if index_col not in form_ret.columns:
        raise KeyError(f"index_col '{index_col}' not found in form_ret columns")

    # Work on a copy; never mutate the caller
    ret = form_ret.copy()

    # Optionally enforce complete data over the window
    if require_complete:
        complete_cols = ret.columns[ret.notna().all(axis=0)]
        if index_col not in complete_cols:
            raise ValueError(f"Index column '{index_col}' has missing values in formation window.")
        ret = ret[complete_cols]

    # Build price paths and normalize each to start at 1.0
    prc = (1.0 + ret).cumprod()

    if require_complete:
        # Safe if no NaNs: divide all series by row 0
        prc = prc / prc.iloc[0]
    else:
        # Normalize each series by its own first valid price
        def _normalize(s: pd.Series) -> pd.Series:
            fv = s.first_valid_index()
            if fv is None or s.loc[fv] == 0:
                return s
            return s / s.loc[fv]
        prc = prc.apply(_normalize, axis=0)

    # Separate stocks and index
    stocks = [c for c in prc.columns if c != index_col]
    if len(stocks) < 2:
        raise ValueError("Not enough valid stocks in the formation window to form pairs.")

    idx_series = prc[index_col]

    # L1 "area" on overlapping dates
    def l1_area(a: pd.Series, b: pd.Series) -> float:
        ab = pd.concat([a, b], axis=1).dropna()
        if ab.shape[0] == 0:
            return np.inf
        diff = np.abs(ab.iloc[:, 0].to_numpy() - ab.iloc[:, 1].to_numpy())
        return float(np.sum(diff))

    rows = []
    for i, j in itertools.combinations(stocks, 2):
        si, sj = prc[i], prc[j]

        score = l1_area(si, sj) + 0.5 * (l1_area(si, idx_series) + l1_area(sj, idx_series))

        # Spread std over the formation window (normalized prices), robust to NaNs
        spread_std = float((si - sj).dropna().std(ddof=1))

        rows.append({
            "leg_1": i,
            "leg_2": j,
            "area_metric": score,
            "spread_std": spread_std
        })

    pairs = (pd.DataFrame(rows)
             .sort_values("area_metric", ascending=True)
             .head(n_pairs)
             .reset_index(drop=True))

    return pairs


def get_pairs_area_based(form_ret, n_pairs, index_col='NIFTY50'):
    """
    Select pairs based on minimum area between price paths, using proper positional indexing.
    """
    # Initialize prices starting at 1.0
    #form_ret.iloc[0, :] = 0
    #prc = (form_ret + 1.).cumprod()

    ret = form_ret.copy()

    # Optionally enforce complete data over the window
    #if require_complete:
    complete_cols = ret.columns[ret.notna().all(axis=0)]
    if index_col not in complete_cols:
        raise ValueError(f"Index column '{index_col}' has missing values in formation window.")
    ret = ret[complete_cols]

    # Build price paths and normalize each to start at 1.0
    prc = (1.0 + ret).cumprod()

    #if require_complete:
        # Safe if no NaNs: divide all series by row 0
    prc = prc / prc.iloc[0]
    
    # Separate stocks and index
    stocks = [col for col in prc.columns if col != index_col]
    index_prc = prc[index_col]
    
    # Generate all possible stock pairs
    pairs = pd.DataFrame(list(itertools.combinations(stocks, 2)), 
                        columns=["leg_1", "leg_2"])
    
    def calculate_areas(pair):
        # Use .loc[] for label-based access or .iloc[] for positional
        s1 = prc[pair['leg_1']]  # Label-based access
        s2 = prc[pair['leg_2']]  # Label-based access
        
        # Area calculation using numpy
        def area(y1, y2):
            return np.sum(np.abs(y1 - y2))  # “Discrete L1 path distance (sum of absolute deviations over time)”
        
        return (
            area(s1, s2) + 
            0.5 * (area(s1, index_prc) + area(s2, index_prc))
        )
    
    # Apply area calculation
    pairs['area_metric'] = pairs.apply(calculate_areas, axis=1)
    
    # Select top pairs and calculate spread std
    pairs = pairs.sort_values('area_metric').head(n_pairs)
    pairs['spread_std'] = pairs.apply(
        lambda x: np.std(prc[x['leg_1']] - prc[x['leg_2']], ddof=1),
        axis=1
    )
    
    return pairs.reset_index(drop=True)


def preprocess_trades(trade_ret):
    """Clean and prepare returns data"""
    trade_ret_dates = trade_ret.index
    trade_ret.index = np.arange(trade_ret.shape[0])
    trade_ret.iloc[0, :] = 0

    # Handle delisted stocks
    last_valid_ret_ind = trade_ret.apply(pd.Series.last_valid_index)
    for idx, column in enumerate(trade_ret.columns):
        r = trade_ret.loc[trade_ret.index[0]:last_valid_ret_ind.iloc[idx], column]
        r = r.fillna(0)
        trade_ret.loc[trade_ret.index[0]:last_valid_ret_ind.iloc[idx], column] = r

    return trade_ret, trade_ret_dates


def calculate_normalized_prices(trade_ret):
    """Convert returns to normalized price series starting at 1"""
    return (trade_ret + 1.).cumprod()


def process_single_pair(pair, trade_prc, trade_ret, d_open, d_stop, wait1d, last_day, plot=False):
    """Process trading signals and payoffs for a single pair"""
    pair_calcs = pd.DataFrame(np.zeros((trade_prc.shape[0], 9)),
                              columns=["p_1", "p_2", "s", "direction", "w_1", "w_2", "r_1", "r_2", "payoff"])

    # Set up pair data
    leg_1 = pair.leg_1
    leg_2 = pair.leg_2
    pair_calcs.p_1 = trade_prc.loc[:, leg_1]
    pair_calcs.p_2 = trade_prc.loc[:, leg_2]
    pair_calcs.r_1 = trade_ret.loc[:, leg_1]
    pair_calcs.r_2 = trade_ret.loc[:, leg_2]

    # Calculate normalized spread
    pair_calcs.s = (pair_calcs.p_1 - pair_calcs.p_2) / pair.spread_std

    # Find open/close signals
    open_ids = np.array(trade_ret.index * (np.abs(pair_calcs.s) > d_open))
    open_ids = open_ids[open_ids != 0]
    open_ids = open_ids[open_ids <= last_day]

    close_ids = np.array(trade_ret.index[np.sign(pair_calcs.s).diff() != 0])

    close_ids = close_ids[~np.isnan(close_ids)]
    close_ids = np.append(close_ids, last_day)

    # Process trades
    t_open = open_ids[0] if len(open_ids) != 0 else np.nan
    connections = []
    if ~np.isnan(t_open):
        while ~np.isnan(t_open) & (t_open < last_day - wait1d):
            t_close = np.min(close_ids[close_ids > t_open + wait1d])

            # Set trade direction
            pair_calcs.loc[(t_open + wait1d + 1): (t_close + 1), "direction"] = -np.sign(
                pair_calcs.loc[t_open - wait1d, "s"])
            connections.append((t_open, t_close))
            # Calculate weights
            # pair_calcs.w_1[(t_open + wait1d):(t_close + 1)] = np.append(1., (
            #         1 + pair_calcs.r_1[(t_open + wait1d): (t_close)]).cumprod())
            rhs_values1 = np.append(1., (1 + pair_calcs.r_1[(t_open + wait1d):(t_close)]).cumprod())

            # Get the exact row indices we're modifying
            row_indices1 = range((t_open + wait1d), (t_close + 1))
            
            # Verify lengths match
            if len(rhs_values1) != len(row_indices1):
                raise ValueError(f"Length mismatch: {len(rhs_values1)} VIKRAM 1 values for {len(row_indices1)} rows")
            
            # Assign using .loc with explicit indices
            pair_calcs.loc[row_indices1, 'w_1'] = rhs_values1

            # pair_calcs.loc[(t_open + wait1d):(t_close + 1), 'w_1'] = np.append(1., (1 + pair_calcs.r_1[(t_open + wait1d): (t_close)]).cumprod())

            # pair_calcs.w_2[(t_open + wait1d):(t_close + 1)] = np.append(1., (
            #         1 + pair_calcs.r_2[(t_open + wait1d): (t_close)]).cumprod())
            rhs_values2 = np.append(1., (1 + pair_calcs.r_2[(t_open + wait1d): (t_close)]).cumprod())
            row_indices2 = range((t_open + wait1d), (t_close + 1))

            # Verify lengths match
            if len(rhs_values2) != len(row_indices2):
                raise ValueError(f"Length mismatch: {len(rhs_values2)} VIKRAM 2 values for {len(row_indices2)} rows")
            
            pair_calcs.loc[row_indices2, 'w_2'] = rhs_values2

            # Move to next trade
            t_open = open_ids[open_ids > t_close][0] if any(open_ids > t_close) else np.nan

    # Calculate payoffs
    pair_calcs["payoff"] = pair_calcs.direction * (
            pair_calcs.w_1 * pair_calcs.r_1 - pair_calcs.w_2 * pair_calcs.r_2)

    # Optional, graph individual pair open,close positions
    if (plot):
        plt.figure(figsize=(15, 5))
        plt.plot(pair_calcs.s, 'k-', alpha=0.3, label='Spread')
    
        for i, (op, cl) in enumerate(connections):
            plt.plot([op, cl],
                     [pair_calcs.s.loc[op], pair_calcs.s.loc[cl]],
                     'b-', alpha=0.5)
            plt.scatter(op, pair_calcs.s.loc[op], color='green',
                        marker='^', s=100, label='Open' if i == 0 else "")
            plt.scatter(cl, pair_calcs.s.loc[cl], color='red',
                        marker='v', s=100, label='Close' if i == 0 else "")
    
        plt.title(f'Trade Signals for Pair {pair.leg_1}-{pair.leg_2}')
        plt.xlabel('Date')
        plt.ylabel('Normalized Spread')
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()

        # --- Build a trade log from connections ---
    trade_rows = []
    idx = trade_ret.index  # index (int or datetime)
    is_dt = hasattr(idx, "dtype") and "datetime64" in str(idx.dtype)

    for (op, cl) in connections:
        # Execution begins after the 1-day wait: start = op + wait1d + 1
        start = int(op + wait1d + 1)
        end   = int(cl)  # inclusive
        if end < start:
            continue

        # P&L over the trade window (sum of daily payoff)
        pnl = float(np.nansum(pair_calcs.loc[start:end, "payoff"].values))

        # Direction at entry
        dir_entry = float(pair_calcs.loc[start, "direction"]) if start in pair_calcs.index else np.nan

        # Duration in trading days
        if is_dt:
            open_date = pd.to_datetime(idx[op])
            close_date = pd.to_datetime(idx[cl])
            duration_days = (close_date - open_date).days  # calendar days
            # also store trading-bar count
            bar_count = end - start + 1
        else:
            open_date = int(op)
            close_date = int(cl)
            duration_days = int(cl - op)
            bar_count = end - start + 1

        trade_rows.append({
            "leg_1": leg_1,
            "leg_2": leg_2,
            "open_id": int(op),
            "close_id": int(cl),
            "open_date": open_date,
            "close_date": close_date,
            "entry_spread": float(pair_calcs.loc[op, "s"]) if op in pair_calcs.index else np.nan,
            "exit_spread":  float(pair_calcs.loc[cl, "s"]) if cl in pair_calcs.index else np.nan,
            "direction": dir_entry,
            "duration_bars": int(bar_count),
            "duration_days": int(duration_days),
            "pnl": pnl,  # in return units (decimal)
        })
    trade_log = pd.DataFrame(trade_rows)

    return pair_calcs["payoff"], pair_calcs["direction"], trade_log



def calculate_cc_and_fi_returns(payoffs, directions, trade_ret_dates):
    """Calculate committed capital and fully invested returns"""
    payoffs.index = trade_ret_dates
    directions.index = trade_ret_dates

    # Committed capital approach
    returns_cc = payoffs.mean(axis=1)

    # Fully invested approach
    num_open_pairs = (directions != 0).sum(axis=1)
    num_open_pairs = num_open_pairs.astype(float)  # Convert to float64 upfront
    num_open_pairs[num_open_pairs > 0] = 1.0 / num_open_pairs[num_open_pairs > 0]
    weights_fi = pd.concat([num_open_pairs] * payoffs.shape[1], axis=1)
    returns_fi = (weights_fi * payoffs).sum(axis=1)

    return returns_cc, returns_fi


def calculate_pairs_returns(trade_ret, pairs, d_open=2, d_close=3, wait1d=1):
    """Main function to calculate pairs trading returns"""
    # Preprocess returns
    trade_ret, trade_ret_dates = preprocess_trades(trade_ret)

    # Calculate normalized prices
    trade_prc = calculate_normalized_prices(trade_ret)
    trading_days = trade_prc.shape[0]
    num_pairs = pairs.shape[0]

    # Initialize storage
    payoffs = pd.DataFrame(np.zeros((trading_days, num_pairs)))
    directions = pd.DataFrame(np.zeros((trading_days, num_pairs)))
    trade_logs = []  # <-- collect each pair's trade log

    # Process each pair
    for idx_pair, pair in pairs.iterrows():
        last_day = max(trade_prc[pair.leg_1].last_valid_index(),
                       trade_prc[pair.leg_2].last_valid_index())
        p_payoffs, p_directions, p_trades = process_single_pair(
            pair, trade_prc, trade_ret, d_open, d_close, wait1d, last_day)
        payoffs.loc[:, idx_pair] = p_payoffs
        directions.loc[:, idx_pair] = p_directions
        if p_trades is not None and len(p_trades):
            # add pair id so we can trace back
            p_trades["pair_id"] = idx_pair
            trade_logs.append(p_trades)
        

    # Calculate portfolio returns
    returns_cc, returns_fi = calculate_cc_and_fi_returns(payoffs, directions, trade_ret_dates)
    
    # === Build Table 4.2 metrics from the trade log ===
    if trade_logs:
        trade_log = pd.concat(trade_logs, ignore_index=True)
        total_trades = len(trade_log)
        avg_duration_days = float(trade_log["duration_bars"].mean())  # or "duration_days" if you prefer calendar
        win_rate = float((trade_log["pnl"] > 0).mean() * 100.0)
        avg_profit_per_trade_pct = float(trade_log["pnl"].mean() * 100.0)  # pnl is decimal → %

        perf_table_4_2 = pd.DataFrame({
            "Value": [
                total_trades,
                round(avg_duration_days, 2),
                round(win_rate, 2),
                round(avg_profit_per_trade_pct, 4),
            ]
        }, index=[
            "Total Number of Trades",
            "Average Trade Duration (days)",
            "Win Rate (%)",
            "Average Profit per Trade (%)"
        ])
    else:
        trade_log = pd.DataFrame()
        perf_table_4_2 = pd.DataFrame({
            "Value": [0, np.nan, np.nan, np.nan]
        }, index=[
            "Total Number of Trades",
            "Average Trade Duration (days)",
            "Win Rate (%)",
            "Average Profit per Trade (%)"
        ])

    return {
        "pairs": pairs,
        "directions": directions,
        "payoffs": payoffs,
        "returns_cc": returns_cc,
        "returns_fi": returns_fi,
        "trade_log": trade_log,           # NEW: per-trade details
        "table_4_2": perf_table_4_2       # NEW: summary for thesis
    }


##################### these are functions for backtesting ########################
def initialize_backtest_parameters():
    """Returns configuration parameters for backtest"""
    return {
        'n_formation': 12,  # Formation period in months
        'n_trading': 6,  # Trading period in months
        'num_pairs': 5,  # Number of pairs to select
        'd_open': 2,  # Z-score threshold for opening trades
        'd_close': 3,  # stop loss rule
        'wait1d': 1  # Days to wait before executing
    }


def initialize_results_storage(dates, n_trading):
    """Initialize DataFrames for storing results"""
    columns = [f"P_{i + 1}" for i in range(n_trading)]
    zeros_df = pd.DataFrame(np.zeros((len(dates), n_trading)),
                            index=dates,
                            columns=columns)

    return {
        'returns_cc': zeros_df.copy(),  # Committed capital returns
        'returns_fi': zeros_df.copy(),  # Fully invested returns
        'num_open_pairs': zeros_df.copy()  # Count of open pairs
    }


def prepare_monthly_indices(dates):
    """Create monthly indices for formation/trading period segmentation"""
    month_id = pd.Series(dates.month)
    month_id = (month_id.diff() != 0).cumsum()
    month_id[0] = 0
    return month_id


def run_portfolio_backtest(dates, port_num, month_id, params, ret, vol, results, portfolios, all_trade_logs, distance=True):
    """Run backtest for a single portfolio"""
    port_name = f"P_{port_num + 1}"
    print(f"Running portfolio {port_num + 1} of {params['n_trading']}")

    unique_months = month_id.unique()
    date_range = np.arange(
        start=params['n_formation'] + port_num,
        stop=len(unique_months) - params['n_trading'] + 1,
        step=params['n_trading']
    )

    for month_idx in date_range:
        # Get formation and trading periods
        form_months = unique_months[month_idx - params['n_formation']:month_idx]
        trade_months = unique_months[month_idx:month_idx + params['n_trading']]

        # Get date ranges
        form_dates = get_date_range(month_id, form_months, dates)
        trade_dates = get_date_range(month_id, trade_months, dates)

        portfolios.append({
                    'portfolio': port_name,
                    'formation_start': form_dates.min(),
                    'formation_end': form_dates.max(),
                    'trading_start': trade_dates.min(),
                    'trading_end': trade_dates.max(),
                    'formation_days': (form_dates.max() - form_dates.min()).days + 1,
                    'trading_days': (trade_dates.max() - trade_dates.min()).days + 1
                })

        # Select eligible stocks
        form_ret, trade_ret = select_eligible_stocks(ret, vol, form_dates, trade_dates)

        # Select pairs and calculate returns
        if (distance):
            pairs =  get_pairs(form_ret, params['num_pairs'])
        else:
            pairs =  get_pairs_area_based(form_ret, params['num_pairs'])
        trades = calculate_pairs_returns( trade_ret, pairs, params['d_open'], params['d_close'], params['wait1d'])
        if "trade_log" in trades and len(trades["trade_log"]) > 0:
            tl = trades["trade_log"].copy()
            tl["portfolio"] = port_name
            tl["form_start"] = form_dates[0]
            tl["form_end"]   = form_dates[-1]
            tl["trade_start"] = trade_dates[0]
            tl["trade_end"]   = trade_dates[-1]
            all_trade_logs.append(tl)

        # Store results
        store_results(results, trades, trade_dates, port_name)


def get_date_range(month_id, target_months, dates):
    """Get start and end dates for given month indices"""
    start_date = dates[month_id == target_months.min()][0]
    end_date = dates[month_id == target_months.max()][-1]
    return pd.date_range(start_date, end_date)


def select_eligible_stocks(ret, vol, form_dates, trade_dates):
    """Filter stocks with complete data and sufficient volume"""
    form_ret = ret[form_dates[0]:form_dates[-1]].copy()
    # form_vol = vol[form_dates[0]:form_dates[-1]].fillna(0)

    # Stocks with no missing returns and all volumes > 0
    # ava_stocks = (form_ret.isna().sum() == 0) & ((form_vol == 0).sum() == 0)
    ava_stocks = form_ret.isna().sum() == 0

    return (
        ret.loc[form_dates[0]:form_dates[-1], ava_stocks],
        ret.loc[trade_dates[0]:trade_dates[-1], ava_stocks]
    )


def store_results(results, trades, trade_dates, port_name):
    """Store backtest results in preallocated DataFrames"""
    results['returns_cc'].loc[trade_dates[0]:trade_dates[-1], port_name] = trades["returns_cc"].values
    results['returns_fi'].loc[trade_dates[0]:trade_dates[-1], port_name] = trades["returns_fi"].values
    results['num_open_pairs'].loc[trade_dates[0]:trade_dates[-1], port_name] = (trades["directions"] != 0).sum(
        axis=1).values


def main_backtest(ret, vol=None, distance=True):
    """Complete backtest execution"""
    dates = ret.index
    params = initialize_backtest_parameters()
    results = initialize_results_storage(dates, params['n_trading'])
    month_id = prepare_monthly_indices(dates)
    portfolios = []
    all_trade_logs = []   # collect trade logs from every window & portfolio
    for port_num in range(params['n_trading']):
        tic = time.perf_counter()
        run_portfolio_backtest(dates, port_num, month_id, params, ret, vol, results, portfolios, all_trade_logs, distance)
        toc = time.perf_counter()
        print(f"Portfolio completed in {(toc - tic) / 60.:0.2f} minutes")
    portfolios_df = pd.DataFrame(portfolios)
    trade_log_all = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()

    # print(portfolios_df)
    return results, all_trade_logs, portfolios


############################# functions to aggregate returns ###################
def calculate_aggregate_returns(results):
    strat_returns_cc_w1d = results['returns_cc']
    strat_returns_fi_w1d = results['returns_fi']

    # Average across all portfolios
    ret_cc = strat_returns_cc.mean(axis=1)
    ret_fi = strat_returns_fi.mean(axis=1)

    # Create DataFrames
    ret_daily = pd.DataFrame({
        'ret_cc': ret_cc,
        'ret_fi': ret_fi
    })

    # Monthly compounded returns
    ret_monthly = ret_daily.resample('M').agg(lambda x: (x + 1).prod() - 1)

    # 12-month moving averages
    ret_12_month = ret_monthly.rolling(12).mean()

    return {
        'daily': ret_daily,
        'monthly': ret_monthly,
        '12_month_ma': ret_12_month
    }


def plot_strategy_returns(returns_data, title_suffix=""):
    """
    Plot strategy returns with both CC and FI versions
    Args:
        returns_data: Dictionary from calculate_aggregate_returns()
        title_suffix: Optional string to append to title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Committed Capital plot
    ax1.plot(returns_data['monthly']["ret_cc"],
             color="gray",
             label="Monthly return (CC)")
    ax1.plot(returns_data['12_month_ma']["ret_cc"],
             color="blue",
             label="12-month MA (CC)")
    ax1.set_title(f"Monthly Returns - Committed Capital {title_suffix}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Fully Invested plot
    ax2.plot(returns_data['monthly']["ret_fi"],
             color="lightcoral",
             label="Monthly return (FI)")
    ax2.plot(returns_data['12_month_ma']["ret_fi"],
             color="darkred",
             label="12-month MA (FI)")
    ax2.set_title(f"Monthly Returns - Fully Invested {title_suffix}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


# Usage Example:
# returns_data = calculate_aggregate_returns(strat_returns_cc_w1d, strat_returns_fi_w1d)
# plot_strategy_returns(returns_data, title_suffix="GGR Strategy with 20 Pairs")
# plt.show()

import pandas as pd
import numpy as np


def calculate_sharpe(returns, risk_free_rate=0.0):
    """Annualized Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_cagr(returns):
    """Compound Annual Growth Rate"""
    cum_return = (1 + returns).prod()
    years = len(returns) / 252
    return cum_return ** (1 / years) - 1


def max_dd(returns):
    """Maximum Drawdown"""
    p = (1 + returns).cumprod()
    roll_max = p.cummax()
    drawdown = p / roll_max - 1.0
    return -drawdown.min()


def calculate_calmar(returns):
    """Calmar Ratio (CAGR/MaxDD)"""
    cagr = calculate_cagr(returns)
    dd = max_dd(returns)
    return cagr / abs(dd) if dd != 0 else np.nan


def analyze_performance(daily_returns):
    """Comprehensive performance metrics"""
    return {
        'Sharpe': calculate_sharpe(daily_returns),
        'CAGR': calculate_cagr(daily_returns),
        'MaxDD': max_dd(daily_returns),
        'Calmar': calculate_calmar(daily_returns),
        'AnnualVol': daily_returns.std() * np.sqrt(252)
    }

def downside_deviation(r, rf=0.0):
    """r: daily returns in decimal; downside wrt rf/0"""
    neg = np.minimum(r - rf/252.0, 0.0)
    return float(np.sqrt((neg**2).mean()))

def create_trading_performance(backtest_results, trade_log_all, invet_type):
    daily_port_ret = backtest_results[invet_type].mean(axis=1) # invet_type == returns_cc or returns_fi
    # ---- Portfolio-level metrics ----
    r = daily_port_ret.dropna().astype(float)
    T = len(r)
    cum_curve = (1.0 + r).cumprod()
    cum_ret = cum_curve.iloc[-1] - 1.0
    ann_ret = cum_curve.iloc[-1]**(252 / T) - 1.0
    ann_vol = float(r.std(ddof=0) * np.sqrt(252))
    sharpe  = float((r.mean() / r.std(ddof=0)) * np.sqrt(252)) if r.std(ddof=0) > 0 else np.nan
    ann_downside = downside_deviation(r) * np.sqrt(252)
    sortino = ann_ret / ann_downside
    calmar  = calculate_calmar(r)
    mdd     = max_dd(r)
    var_95_np = float(np.quantile(r, 0.05))
    cvar_95_np = float(r[r <= var_95_np].mean()) if (r <= var_95_np).any() else np.nan
    # Average concurrent open pairs (sanity metric)
    num_open_pairs_w1d = backtest_results['num_open_pairs']
    avg_open_pairs = float(num_open_pairs_w1d.mean(axis=1).mean())
    
    # ---- Trade-level metrics (from the global trade log) ----
    if not trade_log_all.empty:
        total_trades = int(len(trade_log_all))
        avg_dur_days = float(trade_log_all["duration_bars"].mean())  # or "duration_days" if you want calendar days
        win_rate = float((trade_log_all["pnl"] > 0).mean() * 100.0)
        avg_profit_per_trade_pct = float(trade_log_all["pnl"].mean() * 100.0)  # pnl is decimal → %
    else:
        total_trades = 0
        avg_dur_days = np.nan
        win_rate = np.nan
        avg_profit_per_trade_pct = np.nan
    
    # ---- Assemble Table 4.2 ----
    table_4_2 = pd.DataFrame({
        "Value": [
            total_trades,
            round(avg_dur_days, 2),
            round(win_rate, 2),
            round(avg_profit_per_trade_pct, 4),
            round((cum_ret * 100.0), 2),
            round((ann_ret * 100.0), 2),
            round((ann_vol * 100.0), 2),
            round(sharpe, 2),
            round(sortino, 2),
            round((mdd * 100.0), 2),
            round((calmar), 2),
            round((var_95_np * 100.0), 2),
            round((cvar_95_np * 100.0), 2),
            round(avg_open_pairs, 2),
        ]
    }, index=[
        "Total Number of Trades",
        "Average Trade Duration (days)",
        "Win Rate (%)",
        "Average Profit per Trade (%)",
        "Cumulative Return (%)",
        "Annualized Return (%)",
        "Annualized Volatility (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown (%)",
        "Calmar Ratio",
        "VaR (95%) — non-parametric (%)",
        "CVaR (95%) — non-parametric (%)",
        "Avg Concurrent Open Pairs"
    ])
    return table_4_2
    

def create_performance_summary(backtest_results):
    """Generate professional performance report"""

    # Aggregate returns
    cc_returns = backtest_results['returns_cc'].mean(axis=1)
    fi_returns = backtest_results['returns_fi'].mean(axis=1)

    # Calculate metrics
    cc_metrics = analyze_performance(cc_returns)
    fi_metrics = analyze_performance(fi_returns)

    # Create summary table
    summary = pd.DataFrame({
        'Committed Capital': cc_metrics,
        'Fully Invested': fi_metrics
    }).T

    # Formatting
    summary['CAGR'] = summary['CAGR'].map('{:.2%}'.format)
    summary['MaxDD'] = summary['MaxDD'].map('{:.2%}'.format)
    summary['AnnualVol'] = summary['AnnualVol'].map('{:.2%}'.format)
    summary['Sharpe'] = summary['Sharpe'].map('{:.2f}'.format)
    summary['Calmar'] = summary['Calmar'].map('{:.2f}'.format)

    return summary.sort_index()

# Usage Example:
# results = main_backtest(dates, ret, vol)  # Your existing backtest
# performance_report = create_performance_summary(results)

# print("PAIRS TRADING PERFORMANCE REPORT")
# print("="*40)
# print(performance_report)

def plot_portfolio_timeline_scaled(
    portfolio: str,
    start_year: int,
    timeline_df: pd.DataFrame,
    years: int = 3,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    
    timeline_df = timeline_df.copy()
    
    # Convert date columns
    for col in ['formation_start', 'formation_end', 'trading_start', 'trading_end']:
        timeline_df[col] = pd.to_datetime(timeline_df[col])
    
    # Filter data for the specified 3-year window
    end_date = pd.Timestamp(f'{start_year+2}-12-31')
    port_data = timeline_df[
        (timeline_df['portfolio'] == portfolio) &
        (timeline_df['formation_end'] >= pd.Timestamp(f'{start_year}-01-01')) &
        (timeline_df['trading_start'] <= end_date)
    ].copy()
    
    # Sort by formation start
    port_data = port_data.sort_values('formation_start').reset_index(drop=True)
    
    # Create figure with dynamic height
    fig, ax = plt.subplots(figsize=(12, max(4, len(port_data)*0.8)))
    
    # Plot continuous blocks without gaps
    for i, row in port_data.iterrows():
        y_pos = len(port_data) - i  # Reverse order (newest at top)
        
        # Combined period (formation + trading)
        ax.broken_barh(
            [
                (row['formation_start'], row['trading_end'] - row['formation_start'])
            ],
            (y_pos-0.4, 0.8),
            facecolors='steelblue',  # Base color
            edgecolor='white'
        )
        
        # Overlay trading period
        ax.broken_barh(
            [
                (row['trading_start'], row['trading_end'] - row['trading_start'])
            ],
            (y_pos-0.4, 0.8),
            facecolors='coral',  # Trading color overlay
            edgecolor='white'
        )
        
        # Add labels
        ax.text(
            row['formation_start'] + (row['formation_end'] - row['formation_start'])/2,
            y_pos,
            'Formation',
            ha='center', va='center', color='white', fontweight='bold'
        )
        ax.text(
            row['trading_start'] + (row['trading_end'] - row['trading_start'])/2,
            y_pos,
            'Trading',
            ha='center', va='center', color='white', fontweight='bold'
        )
    
    # Formatting
    ax.set_title(f'{portfolio} Continuous Timeline ({start_year}-{start_year+2})', pad=20)
    ax.set_xlabel('Date')
    ax.set_ylabel('Period Sequence')
    
    # Y-axis (period numbers)
    ax.set_yticks(np.arange(1, len(port_data)+1))
    ax.set_yticklabels([f"Period {i}" for i in range(len(port_data), 0, -1)])
    ax.set_ylim(0.5, len(port_data)+0.5)
    
    # X-axis formatting - continuous monthly ticks
    ax.set_xlim(pd.Timestamp(f'{start_year}-01-01'), end_date)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    # plt.show()
    return fig

