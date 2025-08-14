"""
siMMMulator

1. Define Basic Parameters (Python Implementation)
2. Simulate Daily Baseline Sales
3. Generate Ad Spend
4. Generate Media Variables
5. Generate Noisy CVRs
6. Transforming Media Variables: 
    6.1 Pivoting the table to an MMM format
    6.2 Applying Adstock
    6.3 Applying Diminishing Returns to Media Variables
7. Calculating Conversions
8. Generate Final DataFrame


Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
import pandas as pd
from tabulate import tabulate
import numpy as np


def define_basic_parameters(
    years: int = 5,
    channels_impressions: Optional[List[str]] = None,
    channels_clicks: Optional[List[str]] = None,
    channels_sessions: Optional[List[str]] = None,
    frequency_of_campaigns: int = 1,
    true_cvr: List[float] = None,
    revenue_per_conv: float = None,
    start_date: str = "2020/1/1"
) -> Dict[str, Any]:
    """
    Define Basic Parameters
    
    User inputs basic parameters that will be used in subsequent steps to simulate 
    the data set. This function initializes variables for marketing mix modeling.
    
    Args:
        years: Number of years to generate data for. Must be >= 1 and whole number.
        channels_impressions: List of channel names using impressions metric 
                            (e.g., Facebook, TV, Long-Form Video).
        channels_clicks: List of channel names using clicks metric (e.g., Search).
        frequency_of_campaigns: How often campaigns occur. Must be >= 1 and whole number.
        true_cvr: List of underlying conversion rates (0-1) for each channel in order
                 (impressions channels first, then clicks channels).
        revenue_per_conv: Revenue per conversion. Must be > 0.
        start_date: Start date in format "yyyy/mm/dd" or "yyyy-mm-dd".
    
    Returns:
        Dict containing validated parameters for use in subsequent MMM steps.
        
    Raises:
        ValueError: If any parameter validation fails.
        TypeError: If parameter types are incorrect.
    
    Examples:
        >>> params = step_0_define_basic_parameters(
        ...     years=5,
        ...     channels_impressions=["Facebook", "TV"],
        ...     channels_clicks=["Search"],
        ...     frequency_of_campaigns=1,
        ...     true_cvr=[0.001, 0.002, 0.003],
        ...     revenue_per_conv=1,
        ...     start_date="2017/1/1"
        ... )
    """
    
    # Handle None values for optional list parameters
    if channels_impressions is None:
        channels_impressions = []
    if channels_clicks is None:
        channels_clicks = []
    if channels_sessions is None:
        channels_sessions = []
    if true_cvr is None:
        true_cvr = []
    
    # Store variables
    years_var = years
    channels_impressions_var = channels_impressions
    channels_clicks_var = channels_clicks
    frequency_of_campaigns_var = frequency_of_campaigns
    true_cvr_var = true_cvr
    revenue_per_conv_var = revenue_per_conv
    
    # Validate and parse start_date
    try:
        # Handle both "/" and "-" separators
        date_str = start_date.replace("/", "-")
        start_date_parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        raise ValueError("You didn't enter a correct format for the date. Enter as a string yyyy/mm/dd or yyyy-mm-dd")
    
    # Type validation
    if not isinstance(years_var, int):
        raise TypeError("You did not enter a number for years. You must have years be a numeric type.")
    
    if not isinstance(frequency_of_campaigns, int):
        raise TypeError("You did not enter a number for frequency_of_campaigns. You must enter a numeric")
    
    if not isinstance(true_cvr, list) or not all(isinstance(x, (int, float)) for x in true_cvr):
        raise TypeError("You did not enter a number for the true conversion rates. Must enter a numeric.")
    
    if not isinstance(revenue_per_conv, (int, float)):
        raise TypeError("You did not enter a number for the revenue per conversion. Must enter a numeric.")
    
    # Value validation
    if years_var < 1:
        raise ValueError('You entered less than 1 year. Must generate at least 1 year worth of data')
    
    if years_var != int(years_var):
        raise ValueError("You entered a decimal for the number of years. Must choose a whole number of years")
    
    # Check conversion rates length matches number of channels
    total_channels = (
        len([ch for ch in channels_impressions if ch]) +
        len([ch for ch in channels_clicks if ch]) +
        len([ch for ch in channels_sessions if ch])  # NEW
    )

    if len(true_cvr) != total_channels:
        raise ValueError("Did not input enough numbers or input too many numbers for conversion rates. Must have a conversion rate for each channel specified.")
    
    # Validate conversion rates are between 0 and 1
    if not all(cvr > 0 for cvr in true_cvr):
        raise ValueError("You entered a negative conversion rate. Enter a conversion rate between 0 and 1")
    
    if not all(cvr <= 1 for cvr in true_cvr):
        raise ValueError("You entered a conversion rate greater than 1. Enter conversion rate between 0 and 1.")
    
    # Validate frequency of campaigns
    if frequency_of_campaigns != int(frequency_of_campaigns):
        raise ValueError("You entered a decimal for the frequency of campaigns. You must enter a whole number")
    
    if frequency_of_campaigns < 1:
        raise ValueError("You entered a frequency of campaign less than 1. You must enter a number greater than 1")
    
    # Validate revenue per conversion
    if revenue_per_conv <= 0:
        raise ValueError("You entered a negative or zero revenue per conversion. You must enter a positive number")
    
    # Create return dictionary
    parameters = {
        'years': years_var,
        'channels_impressions': channels_impressions_var,
        'channels_clicks': channels_clicks_var,
        'channels_sessions': channels_sessions,
        'frequency_of_campaigns': frequency_of_campaigns_var,
        'true_cvr': true_cvr_var,
        'revenue_per_conv': revenue_per_conv_var,
        'start_date': start_date_parsed
    }
    
    # Print confirmation (matching R output format)
    summary_data = [
    ["Years of Data to generate", parameters['years']],
    ["Channels (impressions)", ", ".join(parameters['channels_impressions'])],
    ["Channels (clicks)", ", ".join(parameters['channels_clicks'])],
    ["Channels (sessions)", ", ".join(parameters['channels_sessions'])],    
    ["Campaign frequency", parameters['frequency_of_campaigns']],
    ["True CVRs", ", ".join(map(str, parameters['true_cvr']))],
    ["Revenue per conversion", parameters['revenue_per_conv']],
    ["Start date", parameters['start_date']]
    ]

    print("***** Defining Basic Parameters *****")
    print(tabulate(summary_data, headers=["Parameter", "Value"], tablefmt="github"))
    
    return parameters



def create_baseline(
    setup_variables: Dict[str, Any],
    base_p: float,
    trend_p: float = 2,
    temp_var: float = 8,
    temp_coef_mean: float = 50000,
    temp_coef_sd: float = 5000,
    error_std: float = 100
) -> pd.DataFrame:
    """
    Simulate Daily Baseline Sales
    
    Generates daily baseline sales (sales not due to ad spend) for the number of years specified.
    Takes user input and adds statistical noise.
    
    Args:
        setup_variables: Dictionary that was created after running basic setup. Stores the inputs specified.
        base_p: Amount of baseline sales we get in a day (sales not due to ads).
        trend_p: How much baseline sales is going to grow over the whole period of our data.
        temp_var: How big the height of the sine function is for temperature -- i.e. how much 
                 temperature varies (used to inject seasonality into our data).
        temp_coef_mean: The average of how important seasonality is in our data (the larger 
                       this number, the more important seasonality is for sales).
        temp_coef_sd: The standard deviation of how important seasonality is in our data 
                     (the larger this number, the more variable the importance of seasonality is for sales).
        error_std: Amount of statistical noise added to baseline sales (the larger this number, 
                  the noisier baseline sales will be).
    
    Returns:
        DataFrame with daily baseline sales data including components.
        
    Raises:
        TypeError: If parameter types are incorrect.
        ValueError: If parameter values are invalid.
    
    Examples:
        >>> params = define_basic_parameters(years=5, ...)
        >>> baseline_df = create_baseline(
        ...     setup_variables=params,
        ...     base_p=10000,
        ...     trend_p=1.8,
        ...     temp_var=8,
        ...     temp_coef_mean=50000,
        ...     temp_coef_sd=5000,
        ...     error_std=100
        ... )
    """
    
    # Extract necessary variables from Step 0's output
    years = setup_variables['years']
    
    # Type validation
    if not isinstance(base_p, (int, float)):
        raise TypeError("You did not enter a number for base_p. You must have base_p be a numeric type.")
    
    if not isinstance(trend_p, (int, float)):
        raise TypeError("You did not enter a number for trend_p. You must enter a numeric")
    
    if not isinstance(temp_var, (int, float)):
        raise TypeError("You did not enter a number for temp_var. Must enter a numeric.")
    
    if not isinstance(temp_coef_mean, (int, float)):
        raise TypeError("You did not enter a number for temp_coef_mean. Must enter a numeric.")
    
    if not isinstance(temp_coef_sd, (int, float)):
        raise TypeError("You did not enter a number for temp_coef_sd. Must enter a numeric.")
    
    if not isinstance(error_std, (int, float)):
        raise TypeError("You did not enter a number for error_std. Must enter a numeric.")
    
    # Warning for large error relative to baseline
    if error_std > base_p:
        print("Warning: You entered an error much larger than your baseline sales. "
              "As a result, you may get negative numbers for baseline sales. "
              "We have corrected these negative baseline sales to 0. "
              "To not get this warning, set an error_std much lower than base_p.")
    
    # Number of days to generate data for
    total_days = years * 365
    day = np.arange(1, total_days + 1)
    
    # Base sales of base_p units
    base = np.full(total_days, base_p)
    
    # Trend of trend_p extra units per day
    trend_cal = (trend_p / total_days) * base_p
    trend = trend_cal * day
    
    # Temperature generated by a sin function and we can manipulate how much 
    # the sin function goes up or down with temp_var
    temp = temp_var * np.sin(day * np.pi / 182.5)
    
    # Coefficient of temperature's effect on sales will be a random variable with normal distribution
    seasonality_coef = np.random.normal(temp_coef_mean, temp_coef_sd)
    seasonality = seasonality_coef * temp
    
    # Add some noise to the trend
    error = np.random.normal(0, error_std, total_days)
    
    # Generate series for baseline sales
    baseline_sales = base + trend + seasonality + error
    
    # If error term makes baseline_sales negative, make it 0
    baseline_sales = np.maximum(baseline_sales, 0)
    
    # Create output DataFrame
    output = pd.DataFrame({
        'day': day,
        'baseline_sales': baseline_sales,
        'base': base,
        'trend': trend,
        'temp': temp,
        'seasonality': seasonality,
        'error': error
    })
    
    print("Generating baseline sales: Done!")
    print("\nDescriptive statistics:")
    print(output.describe().T)  # transpose for easier reading
    
    return output


def create_ads_spend(
    setup_variables: Dict[str, Any],
    campaign_spend_mean: float,
    campaign_spend_std: float,
    min_max_proportion: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Generate Ad Spend

    Simulates how much to spend on each ad campaign and channel.
    
    Args:
        setup_variables: Dict from define_basic_parameters()
        campaign_spend_mean: Mean spend per campaign (float)
        campaign_spend_std: Std deviation spend per campaign (float)
        min_max_proportion: Dict {channel_name: (min_prop, max_prop)}
                            Last channel is omitted (gets remainder automatically)
    
    Returns:
        DataFrame with simulated ad spend per campaign per channel.

    Raises:
        TypeError: If parameter types are incorrect.
        ValueError: If parameter values are invalid.
    
    Examples:
        >>> params = define_basic_parameters(years=5, ...)
        >>> min_max_proportion = {
                "Programmatic": (0.45, 0.55),
                "Google.SEM": (0.15, 0.25),
                "TikTok": (0.10, 0.20),
                "SEO.Non.Brand": (0.05, 0.15),
                "Facebook": (0.05, 0.15),
                "CRM": (0.05, 0.10),
                "Affiliates": (0.02, 0.08),
                "Direct": (0, 0),  
                # last channel 'Unassigned' will get remainder
            }
        >>> # Global spend parameters
        >>> spend_df = create_ads_spend(
        ...     setup_variables=params,
        ...     campaign_spend_mean=329000,
        ...     campaign_spend_std=100000,
        ...     min_max_proportion=min_max_proportion
        ... )
    """
    years = setup_variables['years']
    freq = setup_variables['frequency_of_campaigns']
    start_date = pd.to_datetime(setup_variables['start_date'])

    # Gather all channels with types in the correct order
    channels = []
    for ch in setup_variables.get('channels_impressions', []):
        channels.append({"name": ch, "type": "impressions"})
    for ch in setup_variables.get('channels_clicks', []):
        channels.append({"name": ch, "type": "clicks"})
    for ch in setup_variables.get('channels_sessions', []):
        channels.append({"name": ch, "type": "sessions"})

    n_days = years * 365
    n_campaigns = n_days // freq
    campaign_days = pd.date_range(start=start_date, periods=n_campaigns, freq=f"{freq}D")

    # Validate spend params
    if not isinstance(campaign_spend_mean, (int, float)):
        raise TypeError("campaign_spend_mean must be numeric")
    if not isinstance(campaign_spend_std, (int, float)):
        raise TypeError("campaign_spend_std must be numeric")
    if campaign_spend_mean < 0:
        raise ValueError("Mean spend must be positive")
    if campaign_spend_std > campaign_spend_mean:
        print("Warning: std deviation > mean — may cause negative spends (will be set to 0)")

    # Simulate spends
    rows = []
    for day in campaign_days:
        proportions = []
        total_prop = 0

        for idx, ch in enumerate(channels):
            name = ch["name"]

            if idx < len(channels) - 1:  # all but last channel
                if name not in min_max_proportion:
                    raise ValueError(f"Missing min/max proportion for channel '{name}'")
                min_p, max_p = min_max_proportion[name]
                if not (0 <= min_p <= 1 and 0 <= max_p <= 1):
                    raise ValueError(f"Proportions for '{name}' must be between 0 and 1")
                p = np.random.uniform(min_p, max_p)
                proportions.append(p)
                total_prop += p
            else:
                proportions.append(max(0, 1 - total_prop))  # last channel gets remainder

        # Spend per channel
        for ch, prop in zip(channels, proportions):
            spend_total = np.random.normal(campaign_spend_mean, campaign_spend_std)
            spend_total = max(spend_total, 0)  # ensure no negative spends
            spend_channel = spend_total * prop

            rows.append({
                "day": day,
                "channel": ch["name"],
                "channel_type": ch["type"],
                "total_campaign_spend": spend_total,
                "channel_prop_spend": prop,
                "spend_channel": spend_channel
            })

    df = pd.DataFrame(rows)
    print("Simulated ad spend: Done!")
    return df

def _make_week_effect(
    dow: np.ndarray,
    cfg: Dict[str, Any],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Construct a weekly seasonality effect for session-driven channels.

    This function produces a repeating 7-day pattern (Monday=0 ... Sunday=6),
    scaled by a given weekly strength and centered so that the mean effect over
    a week is approximately zero (avoiding inflation of totals).

    Weekly patterns can be specified in three ways:
        1. List/array of 7 floats: explicit daily multipliers (Mon..Sun).
        2. "sin": smooth sinusoidal cycle over the week.
        3. "random": random weekly shape generated from a normal distribution.

    Args:
        dow (np.ndarray):
            Day-of-week indices for each observation (0=Monday ... 6=Sunday).
        cfg (dict):
            Weekly effect configuration:
                - weekly_strength (float): Amplitude of weekly variation.
                - weekly_pattern (str|list|np.ndarray):
                    * "sin": smooth sine wave over the week.
                    * "random": random shape with mean 0 and max abs 1.
                    * list/array of 7 floats: explicit daily pattern.
                - seed (int, optional): Used by caller to init `rng` for reproducibility.
        rng (np.random.Generator):
            Numpy random generator (already seeded if reproducibility is desired).

    Returns:
        np.ndarray:
            Array of weekly effects (same shape as `dow`), scaled to `weekly_strength`.

    Raises:
        ValueError:
            If `weekly_pattern` is a list/array but length is not 7.
            If `weekly_pattern` is an invalid string.
        TypeError:
            If `weekly_pattern` type is unsupported.
    """

    strength = float(cfg.get("weekly_strength", 0.12))
    pattern = cfg.get("weekly_pattern", "sin")

    if isinstance(pattern, (list, tuple, np.ndarray)):
        pat = np.array(pattern, dtype=float)
        if pat.shape[0] != 7:
            raise ValueError("sessions weekly_pattern list must have 7 elements (Mon..Sun).")
        # center to mean 0, scale to desired strength
        pat = pat - pat.mean()
        # normalize to max abs 1 to make strength meaningful
        maxabs = np.max(np.abs(pat)) if np.max(np.abs(pat)) > 0 else 1.0
        pat = (pat / maxabs) * strength
        return pat[dow]
    elif isinstance(pattern, str):
        p = pattern.lower()
        if p == "sin":
            # smooth weekly sinusoid
            return strength * np.sin(2 * np.pi * dow / 7.0)
        elif p == "random":
            # random shape per channel; mean 0, max abs 1, then scale
            base = rng.normal(0.0, 1.0, 7)
            base -= base.mean()
            maxabs = np.max(np.abs(base)) if np.max(np.abs(base)) > 0 else 1.0
            base = (base / maxabs) * strength
            return base[dow]
        else:
            raise ValueError("sessions weekly_pattern must be 'random', 'sin', or a list of 7 floats.")
    else:
        raise TypeError("sessions weekly_pattern must be 'random', 'sin', or a list of 7 floats.")

def generate_media(
    setup_variables: Dict[str, Any],
    df_ads_spends: pd.DataFrame,
    true_cpm: Dict[str, Optional[float]],
    true_cpc: Dict[str, Optional[float]],
    mean_noisy: Optional[Dict[str, float]] = None,
    std_noisy: Optional[Dict[str, float]] = None,
    sessions_config: Optional[Dict[str, Any]] = None,  # per-channel config (see doc above)
) -> pd.DataFrame:
    """
    Generate daily media metrics (impressions, clicks, sessions, and spend allocation)
    for each channel, including seasonality and trend for session-based channels.

    This function simulates the raw marketing input data used in MMM pipelines.
    For spend-based channels, impressions and clicks are derived from CPM/CPC values.
    For session-only channels, traffic is generated from a baseline plus seasonal
    patterns and optional trend.

    Args:
        setup_variables (dict):
            Dictionary of MMM setup parameters containing:
                - "channels_impressions" (list[str])
                - "channels_clicks" (list[str])
                - "channels_sessions" (list[str])
                - "frequency_of_campaigns" (int)
                - "start_date" (str: YYYY-MM-DD)
                - "years" (int)
        df_ads_spends (pd.DataFrame):
            DataFrame with at least:
                - 'day' (datetime-like) or 'campaign_id' (int)
                - 'channel' (str)
                - 'spend_channel' (float)
        true_cpm (dict):
            Mapping {channel: CPM in currency units} for impression-based channels.
            Use None for channels without CPM.
        true_cpc (dict):
            Mapping {channel: CPC in currency units} for click/session-based channels.
            Use None for channels without CPC.
        mean_noisy (dict, optional):
            Mapping {channel: mean noise in CPM/CPC}, defaults to 0.0 if missing.
        std_noisy (dict, optional):
            Mapping {channel: standard deviation of noise}, defaults to 0.0 if missing.
        sessions_config (dict, optional):
            Mapping {channel: {...}} for session-only channels with keys:
                - base_sessions (float): mean sessions/day
                - trend_per_year (float): multiplicative annual trend (e.g., 0.05 = +5% per year)
                - weekly_strength (float): amplitude of weekly cycle
                - annual_strength (float): amplitude of annual cycle
                - noise_cv (float): coefficient of variation for random noise

    Returns:
        pd.DataFrame:
            Original df_ads_spends with additional columns:
                - impressions_{channel}_after_running_day_{n}
                - clicks_{channel}_after_running_day_{n}
                - sessions_{channel}_after_running_day_{n}
                - spend_{channel}_after_running_day_{n}
                - true_cpm, noisy_cpm, true_cpc, noisy_cpc
    """
    # ----- unpack setup -----
    channels_impressions = setup_variables.get("channels_impressions", [])
    channels_clicks      = setup_variables.get("channels_clicks", [])
    channels_sessions    = setup_variables.get("channels_sessions", [])
    frequency            = int(setup_variables["frequency_of_campaigns"])

    channels = channels_impressions + channels_clicks + channels_sessions

    # ----- ensure day/campaign_id -----
    df = df_ads_spends.copy()
    if "campaign_id" in df.columns and "day" in df.columns:
        day_order = pd.to_datetime(df["day"]).sort_values().drop_duplicates().reset_index(drop=True)
        day_to_idx = {d: i+1 for i, d in enumerate(day_order)}
        df["campaign_id"] = pd.to_datetime(df["day"]).map(day_to_idx).astype(int)
    elif "campaign_id" not in df.columns:
        if "day" not in df.columns:
            raise ValueError("df_ads_spends must contain either 'day' or 'campaign_id'.")
        df["day"] = pd.to_datetime(df["day"])
        day_order = df["day"].sort_values().drop_duplicates().reset_index(drop=True)
        day_to_idx = {d: i+1 for i, d in enumerate(day_order)}
        df["campaign_id"] = df["day"].map(day_to_idx).astype(int)
    else:
        df["campaign_id"] = df["campaign_id"].astype(int)

    # ----- CPM/CPC & noise defaults -----
    for ch in channels:
        true_cpm.setdefault(ch, np.nan)
        true_cpc.setdefault(ch, np.nan)

    if mean_noisy is None:
        mean_noisy = {ch: 0.0 for ch in channels}
    else:
        for ch in channels:
            mean_noisy.setdefault(ch, 0.0)

    if std_noisy is None:
        std_noisy = {ch: 0.0 for ch in channels}
    else:
        for ch in channels:
            std_noisy.setdefault(ch, 0.0)

    # map channel-wise params to rows
    cpm_vec = df["channel"].map(true_cpm).values
    cpc_vec = df["channel"].map(true_cpc).values
    mu_vec  = df["channel"].map(mean_noisy).values
    sd_vec  = df["channel"].map(std_noisy).values

    # noisy CPM/CPC
    global_seed = (sessions_config or {}).get("default", {}).get("seed", None)
    rng_global = np.random.default_rng(global_seed)
    noisy_cpm = cpm_vec + np.where(sd_vec > 0, rng_global.normal(mu_vec, sd_vec), mu_vec)
    noisy_cpc = cpc_vec + np.where(sd_vec > 0, rng_global.normal(mu_vec, sd_vec), mu_vec)
    noisy_cpm = np.where(np.isnan(noisy_cpm) | (noisy_cpm <= 0), np.nan, noisy_cpm)
    noisy_cpc = np.where(np.isnan(noisy_cpc) | (noisy_cpc <= 0), np.nan, noisy_cpc)

    df["true_cpm"]  = cpm_vec
    df["noisy_cpm"] = noisy_cpm
    df["true_cpc"]  = cpc_vec
    df["noisy_cpc"] = noisy_cpc

    # impressions/clicks only where applicable
    df["lifetime_impressions"] = np.where(
        df["channel"].isin(channels_impressions),
        (df["spend_channel"] / df["noisy_cpm"]) * 1000,
        np.nan
    )
    df["lifetime_clicks"] = np.where(
        df["channel"].isin(channels_clicks),
        df["spend_channel"] / df["noisy_cpc"],
        np.nan
    )

    # ----- sessions synthesis (per-channel config) -----
    if channels_sessions:
        # default config if not supplied
        default_cfg = {
            "base": 10000.0,
            "trend": 0.0,
            "weekly_strength": 0.12,
            "weekly_pattern": "sin",   # "random" | "sin" | [7 floats]
            "annual_strength": 0.15,
            "noise_cv": 0.08,
            "seed": None,
        }
        sessions_config = sessions_config or {}
        default_cfg.update(sessions_config.get("default", {}))

        # ensure we have dates
        df["day"] = pd.to_datetime(df.get("day", pd.NaT))
        if df["day"].isna().all():
            start = pd.Timestamp(setup_variables.get("start_date", "2020-01-01"))
            date_map = df.groupby("campaign_id").size().index.to_series().map(
                lambda cid: start + pd.Timedelta(days=int(cid)-1)
            )
            df["day"] = df["campaign_id"].map(date_map.to_dict())

        dates = pd.to_datetime(df["day"])
        n = len(dates)
        years_from_start = (dates - dates.min()).dt.days.values / 365.0
        dow = dates.dt.dayofweek.values
        doy = dates.dt.dayofyear.values

        lifetime_sessions = np.full(n, np.nan, dtype=float)

        chan_cfgs = sessions_config.get("channels", {}) or {}
        for ch in channels_sessions:
            # build channel cfg = defaults overridden by channel-specific
            cfg = default_cfg.copy()
            cfg.update(chan_cfgs.get(ch, {}))

            # RNG per channel for reproducibility if seed given
            rng = np.random.default_rng(cfg.get("seed", default_cfg.get("seed", None)))

            base = float(cfg.get("base", 10000.0))
            trend_per_year = float(cfg.get("trend", 0.0))
            weekly_strength = float(cfg.get("weekly_strength", 0.12))
            annual_strength = float(cfg.get("annual_strength", 0.15))
            noise_cv = float(cfg.get("noise_cv", 0.08))

            trend_factor = 1.0 + trend_per_year * years_from_start
            week_effect = _make_week_effect(dow, cfg, rng) if weekly_strength != 0 else 0.0
            annual = annual_strength * np.sin(2 * np.pi * doy / 365.25) if annual_strength != 0 else 0.0

            expected = base * trend_factor * (1.0 + week_effect + annual)
            expected = np.maximum(expected, 0.0)
            noise = rng.normal(loc=0.0, scale=noise_cv, size=n) if noise_cv > 0 else 0.0
            series = np.maximum(expected * (1.0 + noise), 0.0)

            mask = (df["channel"] == ch).values
            lifetime_sessions[mask] = series[mask]

        df["lifetime_sessions"] = lifetime_sessions
    else:
        df["lifetime_sessions"] = np.nan

    # ----- spread uniformly across campaign length -----
    frequency = max(1, int(frequency))
    for ch in channels_impressions:
        for j in range(1, frequency + 1):
            df[f"impressions_{ch}_after_running_day_{j}"] = np.where(
                df["channel"] == ch, np.nan_to_num(df["lifetime_impressions"]) / frequency, 0.0
            )
    for ch in channels_clicks:
        for j in range(1, frequency + 1):
            df[f"clicks_{ch}_after_running_day_{j}"] = np.where(
                df["channel"] == ch, np.nan_to_num(df["lifetime_clicks"]) / frequency, 0.0
            )
    for ch in channels_sessions:
        for j in range(1, frequency + 1):
            df[f"sessions_{ch}_after_running_day_{j}"] = np.where(
                df["channel"] == ch, np.nan_to_num(df["lifetime_sessions"]) / frequency, 0.0
            )
    for ch in channels:
        for j in range(1, frequency + 1):
            df[f"spend_{ch}_after_running_day_{j}"] = np.where(
                df["channel"] == ch, df["spend_channel"] / frequency, 0.0
            )

    print("Simulating media variables: Done!")
    return df

def generate_noisy_cvr(
    setup_variables: Dict[str, Any],
    df_media: pd.DataFrame,
    mean_noisy_cvr: Optional[Dict[str, float]] = None,
    std_noisy_cvr: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Generate Noisy CVRs (Python port)

    - Uses true CVR from setup_variables['true_cvr'] (ordered as impressions + clicks + sessions).
    - Applies noise per channel (dicts by channel); defaults to 0 if not provided.
    - Produces columns: noisy_cvr_<channel>_after_running_day_<n> for impressions + clicks channels.

    Args:
        setup_variables: dict, must include:
            'years', 'channels_impressions', 'channels_clicks', 'channels_sessions',
            'frequency_of_campaigns', 'true_cvr'
        df_ads_step: DataFrame (must include 'channel')
        mean_noisy_cvr: optional dict {channel: mean}, defaults to 0 for missing channels
        std_noisy_cvr: optional dict {channel: std},  defaults to 0 for missing channels

    Returns:
        DataFrame with per-row 'true_cvr', 'noisy_cvr' and
        per-day columns 'noisy_cvr_<channel>_after_running_day_<n>' (for impressions+clicks).
    Raises:
        TypeError / ValueError on invalid inputs.
    """
    channels_impr  = setup_variables.get("channels_impressions", [])
    channels_click = setup_variables.get("channels_clicks", [])
    channels_sess  = setup_variables.get("channels_sessions", [])
    freq           = int(setup_variables["frequency_of_campaigns"])
    true_cvr_list  = setup_variables["true_cvr"]

    all_channels = channels_impr + channels_click + channels_sess
    if len(true_cvr_list) != len(all_channels):
        raise ValueError("true_cvr length must equal total channels (impr+clicks+sessions).")

    true_cvr_map = {ch: float(cv) for ch, cv in zip(all_channels, true_cvr_list)}
    media_channels = set(channels_impr + channels_click + channels_sess)

    # defaults for noise
    if mean_noisy_cvr is None:
        mean_noisy_cvr = {ch: 0.0 for ch in all_channels}
    else:
        for ch in all_channels:
            mean_noisy_cvr.setdefault(ch, 0.0)

    if std_noisy_cvr is None:
        std_noisy_cvr = {ch: 0.0 for ch in all_channels}
    else:
        for ch in all_channels:
            std_noisy_cvr.setdefault(ch, 0.0)

    df = df_media.copy()
    if "campaign_id" not in df.columns:
        # derive from day if needed
        if "day" in df.columns:
            s = pd.to_datetime(df["day"]).sort_values().drop_duplicates().reset_index(drop=True)
            map_idx = {d: i+1 for i, d in enumerate(s)}
            df["campaign_id"] = pd.to_datetime(df["day"]).map(map_idx).astype(int)
        else:
            raise ValueError("df_media must contain 'campaign_id' or 'day'.")

    # true & noisy CVR per row (only for media channels)
    df["true_cvr"] = df["channel"].map(true_cvr_map).astype(float)
    mu = df["channel"].map(mean_noisy_cvr).astype(float)
    sd = df["channel"].map(std_noisy_cvr).astype(float)

    noise = np.where(sd.values > 0, np.random.normal(mu.values, sd.values), mu.values)
    noisy = df["true_cvr"].values + noise
    noisy = np.maximum(noisy, 0.0)  # truncate at 0
    # Non-media channels (e.g., sessions) don't get CVR
    noisy = np.where(df["channel"].isin(media_channels), noisy, np.nan)
    df["noisy_cvr"] = noisy

    # spread CVR uniformly across campaign days for media channels
    for ch in media_channels:
        for n in range(1, freq + 1):
            col = f"noisy_cvr_{ch}_after_running_day_{n}"
            df[col] = np.where(df["channel"] == ch, df["noisy_cvr"], 0.0)

    print("You have completed running step 4: Simulating conversion rates.")
    return df



def pivot_to_mmm_format(
    setup_variables: Dict[str, Any],
    df_ads: pd.DataFrame
) -> pd.DataFrame:
    """
    Pivot from campaign+channel to a daily MMM format.

    Mirrors the R implementation:
      - Build a DATE index from start_date across n_days
      - For each channel, find columns like:
          impressions_<channel>_after_running_day_<n>
          clicks_<channel>_after_running_day_<n>
          sessions_<channel>_after_running_day_<n>
          spend_<channel>_after_running_day_<n>
          noisy_cvr_<channel>_after_running_day_<n>
        * select + sum by campaign_id
        * transpose so rows = running-day, cols = campaign_<id>
        * pad to n_days rows
        * shift each campaign col by (freq*i - freq) where i is 1-based campaign index
        * row-sum -> per-day totals
      - Append to output with same column names as R:
          sum_n_<channel>_imps_this_day
          sum_n_<channel>_clicks_this_day
          sum_n_<channel>_sessions_this_day
          sum_spend_<channel>_this_day
          cvr_<channel>_this_day

    Args:
        setup_variables: dict from step 0. Required keys:
            ['years','channels_impressions','channels_clicks', 'channels_sessions',
             'frequency_of_campaigns','start_date']
        df_ads: pandas DataFrame (must include 'campaign_id').

    Returns:
        pandas DataFrame with column 'DATE' and daily aggregated metric columns.
    """
    # ---- validate inputs ----
    req_keys = ["years", "channels_impressions", "channels_clicks",
                "frequency_of_campaigns", "start_date"]
    for k in req_keys:
        if k not in setup_variables:
            raise ValueError(f"Missing '{k}' in setup_variables.")

    if not isinstance(df_ads, pd.DataFrame):
        raise TypeError("df_ads must be a pandas DataFrame.")
    if "campaign_id" not in df_ads.columns:
        raise ValueError("df_ads must contain 'campaign_id'.")

    years = int(setup_variables["years"])
    channels_impressions: List[str] = list(setup_variables.get("channels_impressions", []))
    channels_clicks: List[str] = list(setup_variables.get("channels_clicks", []))
    channels_sessions: List[str] = list(setup_variables.get("channels_sessions", []))
    frequency_of_campaigns = int(setup_variables["frequency_of_campaigns"])
    start_date = pd.to_datetime(setup_variables["start_date"])

    n_days = years * 365
    channels = channels_impressions + channels_clicks + channels_sessions

    if not channels_impressions and not channels_clicks and not channels_sessions:
        raise ValueError("At least one channel must be specified in setup_variables.")

    # Output skeleton with DATE
    df_out = pd.DataFrame({
        "DATE": pd.date_range(start=start_date, periods=n_days, freq="D")
    })

    # helper: from a set of *_after_running_day_* cols -> per-day summed Series
    def _daily_from_after_running_columns(prefix_cols: List[str], out_name: str) -> pd.Series:
        """
        prefix_cols: list of column names belonging to one (metric, channel) like:
                     impressions_Facebook_after_running_day_1, _2, ...
        out_name: name for the resulting daily series.
        """
        if len(prefix_cols) == 0:
            return pd.Series(np.zeros(n_days, dtype=float), name=out_name)

        # select campaign_id + target columns and sum by campaign_id
        tmp = df_ads[["campaign_id"] + prefix_cols].fillna(0)
        summed = tmp.groupby("campaign_id")[prefix_cols].sum()

        # transpose: rows = running_day index, cols = campaign_<id>
        transposed = summed.T
        transposed.columns = [f"campaign_{cid}" for cid in transposed.columns]

        # pad to n_days rows (R code added NAs then shifted; here we add zeros)
        if transposed.shape[0] < n_days:
            pad = n_days - transposed.shape[0]
            transposed = pd.concat(
                [transposed, pd.DataFrame(0, index=range(pad), columns=transposed.columns)],
                axis=0,
                ignore_index=True
            )

        # shift each campaign column by (frequency*(i) - frequency), i is 1-based
        for i, col in enumerate(transposed.columns, start=1):
            offset = (frequency_of_campaigns * i) - frequency_of_campaigns
            if offset > 0:
                transposed[col] = transposed[col].shift(offset, fill_value=0)

        # row-sum to daily totals; clip to n_days
        daily = transposed.sum(axis=1).iloc[:n_days].reset_index(drop=True)
        daily.name = out_name
        return daily

    # ----- impressions -> sum_n_<channel>_imps_this_day -----
    for ch in channels_impressions:
        cols = [c for c in df_ads.columns
                if c.startswith(f"impressions_{ch}_after_running_day_")]
        series = _daily_from_after_running_columns(
            cols, f"sum_n_{ch}_imps_this_day"
        )
        df_out[series.name] = series

    # ----- clicks -> sum_n_<channel>_clicks_this_day -----
    for ch in channels_clicks:
        cols = [c for c in df_ads.columns
                if c.startswith(f"clicks_{ch}_after_running_day_")]
        series = _daily_from_after_running_columns(
            cols, f"sum_n_{ch}_clicks_this_day"
        )
        df_out[series.name] = series

    # ----- sessions -> sum_n_<channel>_sessions_this_day -----
    for ch in channels_sessions:
        cols = [c for c in df_ads.columns
                if c.startswith(f"sessions_{ch}_after_running_day_")]
        series = _daily_from_after_running_columns(
            cols, f"sum_n_{ch}_sessions_this_day"
        )
        df_out[series.name] = series

    # ----- spend (all media channels) -> sum_spend_<channel>_this_day -----
    for ch in channels:
        cols = [c for c in df_ads.columns
                if c.startswith(f"spend_{ch}_after_running_day_")]
        series = _daily_from_after_running_columns(
            cols, f"sum_spend_{ch}_this_day"
        )
        df_out[series.name] = series

    # ----- noisy CVR (all media channels) -> cvr_<channel>_this_day -----
    for ch in channels:
        cols = [c for c in df_ads.columns
                if c.startswith(f"noisy_cvr_{ch}_after_running_day_")]
        series = _daily_from_after_running_columns(
            cols, f"cvr_{ch}_this_day"
        )
        df_out[series.name] = series

    print("Pivoting the data frame to an MMM format: Done!")
    return df_out


def apply_adstock(
    setup_variables: dict,
    df_daily: pd.DataFrame,
    true_lambda_decay: dict
) -> pd.DataFrame:
    """
    Apply geometric adstock to daily media variables.

    Args:
        setup_variables: Dict (define_basic_parameters)
            Must include:
                - channels_impressions
                - channels_clicks
                - channels_sessions
        df_daily: DataFrame (daily MMM format).
        true_lambda_decay: dict of {channel_name: decay_rate}, where
            decay_rate is between 0 and 1.

    Returns:
        DataFrame with adstocked media variables.
    """
    channels_impressions = setup_variables.get("channels_impressions", [])
    channels_clicks = setup_variables.get("channels_clicks", [])
    channels_sessions = setup_variables.get("channels_sessions", [])

    all_channels = channels_impressions + channels_clicks + channels_sessions

    # --- Validation ---
    if not isinstance(true_lambda_decay, dict):
        raise TypeError("true_lambda_decay must be a dict of {channel: decay_rate}.")

    # Ensure all channels are covered
    missing = [ch for ch in all_channels if ch not in true_lambda_decay]
    if missing:
        raise ValueError(f"Missing decay rates for channels: {missing}")

    # Ensure all decay rates are valid
    if not all(0 <= true_lambda_decay[ch] <= 1 for ch in all_channels):
        raise ValueError("All decay rates must be between 0 and 1.")

    # --- Column mappings for adstock ---
    channel_metric_map = {}
    for ch in channels_impressions:
        channel_metric_map[ch] = f"sum_n_{ch}_imps_this_day"
    for ch in channels_clicks:
        channel_metric_map[ch] = f"sum_n_{ch}_clicks_this_day"
    for ch in channels_sessions:
        channel_metric_map[ch] = f"sum_n_{ch}_sessions_this_day"

    df_out = df_daily.copy()

    # --- Apply geometric decay per channel ---
    for ch in all_channels:
        col = channel_metric_map[ch]
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found in df_daily.")

        decay = true_lambda_decay[ch]
        adstocked_col_name = f"{col}_adstocked"
        adstocked_values = np.zeros(len(df_out))

        # First value is raw metric
        adstocked_values[0] = df_out[col].iloc[0]

        # Apply carryover
        for t in range(1, len(df_out)):
            adstocked_values[t] = df_out[col].iloc[t] + decay * adstocked_values[t - 1]

        df_out[adstocked_col_name] = adstocked_values

    print("Applied geometric adstock to all channels: Done!")
    return df_out


def apply_diminishing_returns(
    setup_variables: Dict[str, Any],
    df_adstock: pd.DataFrame,
    alpha_saturation: Union[float, Dict[str, float]],
    gamma_saturation: Union[float, Dict[str, float]],
    x_marginal: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply diminishing returns (S-curve per R spec) to adstocked media.

    Formula (per channel):
        gammaTrans = quantile( linspace(min(x), max(x), 100), q=gamma )
        s(x)       = x^alpha / ( x^alpha + gammaTrans^alpha )
        y          = x * s(x)
    If x_marginal is provided, s(x) uses x_marginal in place of x.

    Args
    ----
    setup_variables : dict from initial step.
    df_adstock   : DataFrame with adstocked columns, e.g.
                      sum_n_<ch>_imps_this_day_adstocked,
                      sum_n_<ch>_clicks_this_day_adstocked,
                      sum_n_<ch>_sessions_this_day_adstocked (optional).
    alpha_saturation: float or {channel: alpha}, must be > 0.
    gamma_saturation: float or {channel: gamma}, in (0, 1].
    x_marginal      : optional float; if provided, the S-curve factor is constant
                      over time (per channel) using this x.
    include_sessions: whether to also transform session channels if present.

    Returns
    -------
    DataFrame with added columns:
      '<adstocked_col>_decay_diminishing'
    """
    # --- channels ---
    ch_imps = setup_variables.get("channels_impressions", []) or []
    ch_click = setup_variables.get("channels_clicks", []) or []
    ch_sess = setup_variables.get("channels_sessions", []) or []
    all_channels = ch_imps + ch_click + ch_sess

    if not isinstance(df_adstock, pd.DataFrame):
        raise TypeError("df_adstock must be a pandas DataFrame.")

    # --- normalize per-channel params ---
    def per_channel(param, name):
        if isinstance(param, (int, float)):
            if name == "alpha" and param <= 0:
                raise ValueError("alpha must be > 0.")
            if name == "gamma" and not (0 < param <= 1):
                raise ValueError("gamma must be in (0, 1].")
            return {ch: float(param) for ch in all_channels}
        if isinstance(param, dict):
            missing = [ch for ch in all_channels if ch not in param]
            if missing:
                raise ValueError(f"Missing {name} for channels: {missing}")
            out = {}
            for ch, val in param.items():
                v = float(val)
                if name == "alpha" and v <= 0:
                    raise ValueError(f"alpha for '{ch}' must be > 0.")
                if name == "gamma" and not (0 < v <= 1):
                    raise ValueError(f"gamma for '{ch}' must be in (0, 1].")
                out[ch] = v
            return out
        raise TypeError(f"{name}_saturation must be a float or dict {{channel: value}}")

    alpha_map = per_channel(alpha_saturation, "alpha")
    gamma_map = per_channel(gamma_saturation, "gamma")

    # --- columns to transform (must be *_adstocked) ---
    spec = []
    for ch in ch_imps:
        spec.append((ch, f"sum_n_{ch}_imps_this_day_adstocked"))
    for ch in ch_click:
        spec.append((ch, f"sum_n_{ch}_clicks_this_day_adstocked"))
    for ch in ch_sess:
        spec.append((ch, f"sum_n_{ch}_sessions_this_day_adstocked"))

    df = df_adstock.copy()
    eps = 1e-12

    for ch, col in spec:
        if col not in df.columns:
            raise KeyError(f"Required adstocked column not found: '{col}'")

        a = alpha_map[ch]
        g = gamma_map[ch]

        # series to transform
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        x_min, x_max = np.min(x), np.max(x)

        # build the evenly spaced sequence over the observed range and take the gamma-quantile
        seq = np.linspace(x_min, x_max, 100)
        gammaTrans = np.quantile(seq, g)  # matches R: quantile(seq(...), gamma)

        # S-curve factor
        if x_marginal is None:
            num = np.power(np.maximum(x, 0.0) + eps, a)
        else:
            # constant factor using a chosen marginal x
            xM = max(float(x_marginal), 0.0) + eps
            num = np.full_like(x, np.power(xM, a))

        den = num + np.power(max(gammaTrans, 0.0) + eps, a)
        s = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0.0)

        # final diminished series: x * s(x)
        out_col = f"{col}_decay_diminishing"
        df[out_col] = x * s

    print("Apply diminishing marginal returns: Done!")
    return df

def calculate_conversions(
    setup_variables: Dict[str, Any],
    df_daily: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate conversions per channel (impressions, clicks, and sessions channels).

    Args:
        setup_variables (dict):
            Must include:
                - channels_impressions: list[str]
                - channels_clicks: list[str]
                - channels_sessions: list[str]
        df_daily (pd.DataFrame):
            DataFrame after step 5c, containing adstocked + diminishing columns and CVR columns.

    Returns:
        pd.DataFrame: Copy of df_daily with added conv_<channel> columns.
    """
    channels_impressions = setup_variables.get("channels_impressions", [])
    channels_clicks = setup_variables.get("channels_clicks", [])
    channels_sessions = setup_variables.get("channels_sessions", [])

    df_out = df_daily.copy()

    # --- Define helper for vectorized calc ---
    def _calc_conv(ch: str, metric: str):
        base_col = f"sum_n_{ch}_{metric}_this_day_adstocked_decay_diminishing"
        cvr_col  = f"cvr_{ch}_this_day"
        conv_col = f"conv_{ch}"

        if base_col not in df_out.columns:
            raise KeyError(f"Missing column {base_col} for channel {ch}")
        if cvr_col not in df_out.columns:
            raise KeyError(f"Missing column {cvr_col} for channel {ch}")

        df_out[conv_col] = df_out[base_col] * df_out[cvr_col]

    # Impressions
    for ch in channels_impressions:
        _calc_conv(ch, "imps")
    # Clicks
    for ch in channels_clicks:
        _calc_conv(ch, "clicks")
    # Sessions
    for ch in channels_sessions:
        _calc_conv(ch, "sessions")

    print("Calculated conversions for all channel types: Done!")
    return df_out

def generate_final_df(
    setup_variables: dict,
    df_daily: pd.DataFrame,
    df_baseline: pd.DataFrame,
    self_claim_config: Dict[str, float] = None,
    daily_noise_std: float = 0.01,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate final DataFrame for MMM analysis with self-claim simulation and daily noise.

    Creates both actual_* and self_* metrics for:
      - impressions, clicks, sessions, spend, conversions
    Self-reported values are adjusted by:
      - a fixed total-period change (from self_claim_config)
      - daily random noise (daily_noise_std)
    Ensures total period change matches the specified adjustment.

    Args:
        setup_variables: Dict with simulation setup, including:
            - years: float, simulation length
            - channels_impressions: list[str]
            - channels_clicks: list[str]
            - channels_sessions: list[str]
            - revenue_per_conv: float
            - start_date: str | pd.Timestamp
        df_daily: DataFrame from step 6 (contains raw + conversions).
        df_baseline: DataFrame from step 1 (contains baseline_sales).
        self_claim_config: Dict {channel: adj}, adj > 0 means over-report, < 0 means under-report.
        daily_noise_std: Std dev of daily noise as proportion of value.
        seed: Optional random seed for reproducibility.

    Returns:
        pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    years = setup_variables.get("years")
    channels_impressions = setup_variables.get("channels_impressions", [])
    channels_clicks = setup_variables.get("channels_clicks", [])
    channels_sessions = setup_variables.get("channels_sessions", [])
    revenue_per_conversion = setup_variables.get("revenue_per_conv", 0.0)
    start_date = pd.to_datetime(setup_variables.get("start_date"))

    n_days = int(years * 365)

    df_mmm = pd.DataFrame({
        "DATE": pd.date_range(start=start_date, periods=n_days, freq="D")
    })

    # Metric categories
    metric_map = {
        "impressions": (channels_impressions, "_imps_this_day"),
        "clicks":      (channels_clicks,      "_clicks_this_day"),
        "sessions":    (channels_sessions,    "_sessions_this_day"),
    }

    # Helper function to apply adjustment with noise
    def apply_self_claim_with_noise(actual_series, adj):
        if adj == 0:
            return actual_series.copy()

        total_actual = actual_series.sum()
        if total_actual == 0:
            return actual_series.copy()

        # Generate daily noise (mean 0)
        noise = rng.normal(0, daily_noise_std, size=len(actual_series))
        noisy_factor = 1 + noise

        # Scale to match exact total target
        target_total = total_actual * (1 + adj)
        scaling_factor = target_total / (actual_series * noisy_factor).sum()

        return np.maximum(0.0, actual_series * noisy_factor * scaling_factor)

    # Process activity metrics
    for metric_prefix, (channel_list, suffix) in metric_map.items():
        for ch in channel_list:
            col_src = f"sum_n_{ch}{suffix}"
            actual_col = f"actual_{metric_prefix}_{ch}"
            df_mmm[actual_col] = df_daily[col_src] if col_src in df_daily.columns else 0.0

            adj = self_claim_config.get(ch, 0.0) if self_claim_config else 0.0
            df_mmm[f"self_{metric_prefix}_{ch}"] = apply_self_claim_with_noise(df_mmm[actual_col], adj)

    # Process spend
    all_channels = channels_impressions + channels_clicks + channels_sessions
    for ch in all_channels:
        col_src = f"sum_spend_{ch}_this_day"
        actual_col = f"actual_spend_{ch}"
        df_mmm[actual_col] = df_daily[col_src] if col_src in df_daily.columns else 0.0

        adj = self_claim_config.get(ch, 0.0) if self_claim_config else 0.0
        df_mmm[f"self_spend_{ch}"] = apply_self_claim_with_noise(df_mmm[actual_col], adj)

    # Process conversions
    conv_cols = [c for c in df_daily.columns if c.startswith("conv_")]
    for c in conv_cols:
        ch = c[len("conv_"):]
        actual_col = f"actual_conv_{ch}"
        df_mmm[actual_col] = df_daily[c]

        adj = self_claim_config.get(ch, 0.0) if self_claim_config else 0.0
        df_mmm[f"self_conv_{ch}"] = apply_self_claim_with_noise(df_mmm[actual_col], adj)

    # Totals
    df_mmm["total_conv_from_ads"] = df_mmm[[c for c in df_mmm.columns if c.startswith("actual_conv_")]].sum(axis=1)
    df_mmm["revenue_from_ads"] = df_mmm["total_conv_from_ads"] * revenue_per_conversion

    # Baseline revenue
    df_mmm["baseline_revenue"] = df_baseline.get("baseline_sales", pd.Series(0.0, index=df_mmm.index))

    # Total revenue
    df_mmm["total_revenue"] = df_mmm["revenue_from_ads"] + df_mmm["baseline_revenue"]

    print("Generate final dataframe with self-claim + daily noise — Done!")
    return df_mmm















