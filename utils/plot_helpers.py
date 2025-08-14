import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np


def plot_baseline_sales(df_baseline: pd.DataFrame, figsize: tuple = (1000, 700)):
    """
    Interactive plot of baseline sales (top) and components (bottom).
    
    Args:
        df_baseline (pd.DataFrame): Must have columns:
            'day', 'baseline_sales', and optionally 'trend', 'seasonality', 'error'.
        figsize (tuple): (width, height) in pixels for the entire figure.
    """
    required_cols = ['day', 'baseline_sales']
    if not all(col in df_baseline.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols} columns")

    # Ensure data is sorted by day
    df_plot = df_baseline.sort_values("day").copy()

    # Create 2-row subplot layout (shared x-axis)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.25,
        subplot_titles=("Baseline Sales", "Trend / Seasonality / Error")
    )

    # Top plot: Baseline sales
    fig.add_trace(
        go.Scatter(
            x=df_plot['day'],
            y=df_plot['baseline_sales'],
            mode='lines',
            name='Baseline Sales',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Bottom plot: components if available
    component_colors = {
        'trend': 'green',
        'seasonality': 'orange',
        'error': 'red'
    }
    for comp, color in component_colors.items():
        if comp in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot['day'],
                    y=df_plot[comp],
                    mode='lines',
                    name=comp.capitalize(),
                    line=dict(color=color, width=1, dash='dot')
                ),
                row=2, col=1
            )

    # Layout adjustments
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # X-axis formatting
    fig.update_xaxes(
        title_text="Day",
        row=2, col=1,
        tickmode='auto',
        showgrid=True
    )

    fig.update_yaxes(title_text="Baseline Sales", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    fig.show()

def plot_ads_spend(
    df_spend: pd.DataFrame,
    by: str = "channel",
    chart_type: str = "line",
    figsize: tuple = (1200, 800)
):
    """
    Interactive plot of ad spend with option for bar, line, or pie chart.

    Args:
        df_spend (pd.DataFrame): Output from create_ads_spend()
            Must have columns: 'day', 'channel', 'channel_type',
            'spend_channel', 'total_campaign_spend'
        by (str): "channel" = spend per channel over time,
                  "total"   = total spend over time (ignores channel breakdown).
        chart_type (str): "line", "bar", or "pie".
        figsize (tuple): (width, height) in pixels.
    """
    if 'day' not in df_spend.columns:
        raise ValueError("df_spend must contain a 'day' column for x-axis plotting.")
    if by not in ["channel", "total"]:
        raise ValueError("Argument 'by' must be 'channel' or 'total'")
    if chart_type not in ["line", "bar", "pie"]:
        raise ValueError("chart_type must be 'line', 'bar', or 'pie'")

    df_plot = df_spend.sort_values("day").copy()

    if by == "total":
        df_plot = df_plot.groupby("day", as_index=False)["total_campaign_spend"].sum()

    # Pick chart type
    if chart_type == "line":
        if by == "channel":
            fig = px.line(
                df_plot,
                x="day", y="spend_channel", color="channel",
                hover_data=["channel_type", "channel_prop_spend", "total_campaign_spend"],
                title="Ad Spend per Channel Over Time",
                labels={"day": "Day", "spend_channel": "Spend"}
            )
        else:
            fig = px.line(
                df_plot,
                x="day", y="total_campaign_spend",
                title="Total Campaign Spend Over Time",
                labels={"day": "Day", "total_campaign_spend": "Total Spend"}
            )

    elif chart_type == "bar":
        if by == "channel":
            fig = px.bar(
                df_plot,
                x="day", y="spend_channel", color="channel",
                hover_data=["channel_type", "channel_prop_spend", "total_campaign_spend"],
                title="Ad Spend per Channel Over Time",
                labels={"day": "Day", "spend_channel": "Spend"}
            )
        else:
            fig = px.bar(
                df_plot,
                x="day", y="total_campaign_spend",
                title="Total Campaign Spend Over Time",
                labels={"day": "Day", "total_campaign_spend": "Total Spend"}
            )

    elif chart_type == "pie":
        if by == "channel":
            # Aggregate spend across all days
            df_pie = df_plot.groupby("channel", as_index=False)["spend_channel"].sum()
            fig = px.pie(
                df_pie,
                names="channel", values="spend_channel",
                title="Total Spend Distribution by Channel"
            )
        else:
            raise ValueError("Pie chart is only meaningful when by='channel'.")

    # Layout adjustments
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified" if chart_type != "pie" else None,
        width=figsize[0],
        height=figsize[1],
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    fig.show()


def plot_channel_transforms(
    df,
    channel: str,
    date_col: str = "DATE",
    figsize: tuple = (1000, 600)
):
    """
    Interactive Plotly version of channel transforms plot.

    Plots raw vs. adstocked vs. adstocked+diminishing for a given channel,
    auto-detecting impressions/clicks/sessions.

    Args:
        df (pd.DataFrame): Data containing the columns.
        channel (str): Channel name (e.g., 'Programmatic').
        date_col (str): Column name for the date.
        figsize (tuple): Figure size in pixels (width, height).
    """
    # Determine which metric exists for this channel
    base_candidates = [
        f"sum_n_{channel}_imps_this_day",
        f"sum_n_{channel}_clicks_this_day",
        f"sum_n_{channel}_sessions_this_day",
    ]
    base_col = next((c for c in base_candidates if c in df.columns), None)
    if base_col is None:
        raise KeyError(
            f"No raw column found for channel '{channel}'. "
            f"Tried: {', '.join(base_candidates)}"
        )

    adstock_col = f"{base_col}_adstocked"
    dimret_col = f"{adstock_col}_decay_diminishing"

    missing = [c for c in [adstock_col, dimret_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing transformed columns for '{channel}': {missing}.\n"
            "Make sure you've run apply_adstock() and step_5c_diminishing_returns()."
        )

    if "_imps_" in base_col:
        pretty_metric = "Impressions"
    elif "_clicks_" in base_col:
        pretty_metric = "Clicks"
    else:
        pretty_metric = "Sessions"

    fig = go.Figure()

    # Raw
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[base_col],
        mode="lines", name="Raw", line=dict(color="red", width=1.5)
    ))
    # Adstocked
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[adstock_col],
        mode="lines", name="Adstocked", line=dict(color="blue", width=2)
    ))
    # Adstocked + Diminishing
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[dimret_col],
        mode="lines", name="Adstocked + Diminishing", line=dict(color="green", width=1.5)
    ))

    fig.update_layout(
        title=f"{pretty_metric} on {channel} — Raw and Transformed",
        xaxis_title="Date",
        yaxis_title=pretty_metric,
        hovermode="x unified",
        template="plotly_white",
        width=figsize[0],
        height=figsize[1],
        legend=dict(title="Series", x=1.02, y=1, xanchor="left", yanchor="top")
    )

    fig.show()


def plot_actual_vs_self_conversions(
    df,
    key="conversion",       # "activity" or "conversion"
    sort_by="actual",
    descending=True,
    figsize=(900, 500)      # width, height in px
):
    """
    Interactive bar chart comparing actual vs self-reported metrics for each channel.

    Args:
        df (pd.DataFrame): Final dataframe from generate_final_df()
        key (str): "activity" (impressions, clicks, sessions) or "conversion"
        sort_by (str): "actual" or "self" to determine sorting order
        descending (bool): Whether to sort in descending order
        figsize (tuple): (width_px, height_px) size of the plot
    """
    # --- Get column prefix based on key ---
    if key == "conversion":
        actual_prefix = "actual_conv_"
        self_prefix   = "self_conv_"
    elif key == "activity":
        # Combine impressions, clicks, sessions into one total activity
        metrics = ["impressions", "clicks", "sessions"]
        channels = sorted(set(
            c.replace(f"actual_{m}_", "")
            for m in metrics
            for c in df.columns if c.startswith(f"actual_{m}_")
        ))
        actual_sums = []
        self_sums = []
        for ch in channels:
            actual_total = 0
            self_total = 0
            for m in metrics:
                actual_col = f"actual_{m}_{ch}"
                self_col   = f"self_{m}_{ch}"
                if actual_col in df.columns:
                    actual_total += df[actual_col].sum()
                if self_col in df.columns:
                    self_total += df[self_col].sum()
            actual_sums.append(actual_total)
            self_sums.append(self_total)
    else:
        raise ValueError("key must be 'activity' or 'conversion'")

    # --- For conversion mode ---
    if key == "conversion":
        conv_cols = [c for c in df.columns if c.startswith(actual_prefix)]
        channels = [c.replace(actual_prefix, "") for c in conv_cols]
        actual_sums = [df[f"{actual_prefix}{ch}"].sum() for ch in channels]
        self_sums   = [df[f"{self_prefix}{ch}"].sum() for ch in channels]

    # --- Sorting ---
    if sort_by == "actual":
        order = np.argsort(actual_sums)[::-1] if descending else np.argsort(actual_sums)
    elif sort_by == "self":
        order = np.argsort(self_sums)[::-1] if descending else np.argsort(self_sums)
    else:
        raise ValueError("sort_by must be 'actual' or 'self'")

    channels_sorted = [channels[i] for i in order]
    actual_sorted   = [actual_sums[i] for i in order]
    self_sorted     = [self_sums[i] for i in order]

    # --- Plotly bar chart ---
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=channels_sorted,
        y=actual_sorted,
        name=f"Actual {key.capitalize()}",
        marker_color="steelblue",
        hovertemplate="Channel: %{x}<br>Actual: %{y:,.0f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=channels_sorted,
        y=self_sorted,
        name=f"Self-Reported {key.capitalize()}",
        marker_color="orange",
        hovertemplate="Channel: %{x}<br>Self: %{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Actual vs Self-Reported {key.capitalize()} by Channel",
        xaxis_title="Channel",
        yaxis_title=f"Total {key.capitalize()}",
        barmode="group",
        template="plotly_white",
        xaxis_tickangle=-45,
        width=figsize[0],
        height=figsize[1],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100)
    )

    fig.show()





