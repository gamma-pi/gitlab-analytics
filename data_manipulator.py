import argparse
import glob
import polars as pl
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from datetime import datetime, timezone

CORRECTIONS_MAP = {
    'Shipment': 'Deliver',
    'Demand Plan': 'Plan',
    'Supplier Attribution': 'Plan',
    'Design': 'Engineering And Design'
}


def load_data(csv_path: str) -> pl.DataFrame:
    """Load CSV into Polars and parse datetime columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pl.read_csv(csv_path)

    datetime_cols = [
        "pipeline_created_at", "pipeline_updated_at",
        "job_created_at", "job_started_at", "job_finished_at"
    ]
    for col in datetime_cols:
        if col in df.columns:
            df = df.with_columns(
            pl.Series(df[col]).str.strptime(pl.Datetime, strict=False).alias(col)
            )

    # Determine prefix based on CSV filename
    filename = os.path.basename(csv_path).lower()
    if "nprod" in filename:
        prefix = "[nprod]_"
    elif "prod" in filename:
        prefix = "[prod]_"
    else:
        prefix = ""

    print(f"Loaded {df.height:,} records from {csv_path}")
    return df, prefix

def clean_names(df: pl.DataFrame) -> pl.DataFrame:
    """Strip whitespace, replace underscores, title-case, and apply corrections."""

    # Clean domain and subdomain (still using expressions)
    df = df.with_columns([
        pl.col("domain")
          .str.strip_chars()
          .str.replace("_", " ")
          .str.to_titlecase(),
        pl.col("subdomain")
          .str.strip_chars()
          .str.replace("_", " ")
          .str.to_titlecase()
    ])

    # Apply domain corrections eagerly on Series
    subdomains = df["subdomain"].to_list()
    corrected_domain = [
        CORRECTIONS_MAP.get(s, s) if s in CORRECTIONS_MAP else d
        for s, d in zip(subdomains, df["domain"].to_list())
    ]

    df = df.with_columns(
        pl.Series("domain", corrected_domain)
    )

    return df

def analyze_project_ids(df: pl.DataFrame) -> None:
    """Detailed analysis of projectID column."""
    print("\n=== PROJECT ID ANALYSIS ===")
    
    # Basic stats
    total_rows = df.height
    null_count = df.select(pl.col('projectID').is_null().sum()).item()
    non_null_count = total_rows - null_count
    unique_count = df.select(pl.col('projectID').n_unique()).item()
    
    print(f"Total records: {total_rows:,}")
    print(f"Records with null projectID: {null_count:,} ({null_count/total_rows*100:.1f}%)")
    print(f"Records with valid projectID: {non_null_count:,} ({non_null_count/total_rows*100:.1f}%)")
    print(f"Unique projectIDs (including null): {unique_count:,}")
    
    if non_null_count > 0:
        unique_non_null = df.filter(pl.col('projectID').is_not_null()).select(pl.col('projectID').n_unique()).item()
        print(f"Unique projectIDs (excluding null): {unique_non_null:,}")
    
    # Show unique project IDs in a cleaner format
    print("\n--- Unique Project IDs ---")
    unique_ids = df.select(pl.col('projectID').unique().sort()).to_series().to_list()
    
    # Separate null and non-null values
    null_present = None in unique_ids
    non_null_ids = [id for id in unique_ids if id is not None]
    
    if null_present:
        print("null (missing projectID)")
    
    if non_null_ids:
        print("Valid projectIDs:")
        for project_id in non_null_ids:
            count = df.filter(pl.col('projectID') == project_id).height
            print(f"  {project_id}: {count:,} records")
    
    # Show which domains/subdomains have missing projectIDs
    if null_count > 0:
        print(f"\n--- Records with Missing ProjectID ---")
        null_breakdown = (
            df.filter(pl.col('projectID').is_null())
            .group_by(['domain', 'subdomain'])
            .agg(pl.len().alias('null_count'))
            .sort('null_count', descending=True)
        )
        print(null_breakdown)

def summarize_data(df: pl.DataFrame) -> None:
    print("\n--- Data Summary ---")
    print(f"Records: {df.height:,}")
    print(f"Domains: {df.select(pl.col('domain').n_unique()).item()}")
    print(f"Repositories: {df.select(pl.col('repo_name').n_unique()).item()}")
    print(f"Pipelines: {df.select(pl.col('pipeline_id').n_unique()).item()}")
    print(f"Branch flow violations: {df.select(pl.col('branch_flow_violation').sum()).item():,}")

    print("\nPipeline Status Counts:")
    print(
    df.group_by("pipeline_status")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    # Add project ID analysis
    analyze_project_ids(df)

    print("\n--- First 5 rows ---")
    print(df.head(5))

    print("\n--- Columns ---")
    print(df.columns)

def get_default_csv() -> str:
    files = glob.glob("reports/*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found in reports/")
    return max(files, key=os.path.getmtime)

def plot_branch_flow_violations(df: pl.DataFrame, prefix: str = "") -> None:
    """Clustered barplot directly from Polars."""
    os.makedirs("reports/plots", exist_ok=True)

    agg = (
        df.group_by(["domain", "subdomain"])
        .agg(pl.sum("branch_flow_violation").alias("branch_flow_violation"))
        .sort(["domain", "subdomain"])
    )

    domains = agg["domain"].unique().to_list()
    colors = sns.color_palette("tab10", len(domains))
    color_map = {domain: colors[i] for i, domain in enumerate(domains)}

    x_positions, labels = [], []
    current_pos = 0
    cluster_spacing, subdomain_spacing, bar_width = 1.0, 0.3, 0.2

    plt.figure(figsize=(16, 8))

    for domain in domains:
        subdf = agg.filter(pl.col("domain") == domain)
        n_sub = subdf.height
        positions = current_pos + np.arange(n_sub) * subdomain_spacing
        violations = subdf["branch_flow_violation"].to_numpy()

        bars = plt.bar(positions, violations, width=bar_width, color=color_map[domain])
        for bar, value in zip(bars, violations):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.1, str(value),
                     ha="center", va="bottom", fontsize=8)

        x_positions.extend(positions)
        labels.extend(subdf["subdomain"].to_list())
        current_pos = positions[-1] + cluster_spacing

    plt.xticks(x_positions, labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Total Branch Flow Violations")
    plt.title("Branch Flow Violations per Domain and Subdomain")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    handles = [plt.Rectangle((0,0),1,1,color=color_map[d]) for d in domains]
    plt.legend(handles, domains, title="Domain", bbox_to_anchor=(1.05,1), loc="upper left")

    plt.tight_layout()
    filename = f"reports/plots/{prefix}branch_flow_violations.png"
    plt.savefig(filename, dpi=400)
    print("Plot saved as reports/plots/branch_flow_violations.png")

def plot_pipeline_success_rate(df: pl.DataFrame, prefix: str = "") -> None:
    """Clustered + stacked barplot with domain-based colors: dark for success, light for failure."""

    os.makedirs("reports/plots", exist_ok=True)

    # Aggregate counts of pipeline_status per domain+subdomain
    counts = (
        df.group_by(["domain", "subdomain", "pipeline_status"])
          .len()
          .rename({"len": "status_count"})
    )

    # Pivot counts so that 'success' and 'failed' are columns
    pivoted = counts.pivot(
        values="status_count",
        index=["domain", "subdomain"],
        on="pipeline_status"
    ).fill_null(0)

    # Ensure columns exist
    for col in ["success", "failed"]:
        if col not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0).alias(col))

    # Totals and success rate
    pivoted = pivoted.with_columns([
        (pl.col("success") + pl.col("failed")).alias("total"),
        (pl.col("success") / (pl.col("success") + pl.col("failed"))).alias("success_rate")
    ]).sort(["domain", "subdomain"])

    # Unique domains and base colors
    unique_domains = list(dict.fromkeys(pivoted["domain"].to_list()))
    base_colors = sns.color_palette("tab10", len(unique_domains))
    domain_color_map = {domain: base_colors[i] for i, domain in enumerate(unique_domains)}

    # Helper to get dark/light shades
    def domain_colors(domain):
        base = np.array(domain_color_map[domain])
        dark = np.clip(base - 0.2, 0, 1)    # darker for success
        light = np.clip(base + 0.4, 0, 1)   # lighter for failure
        return dark, light

    # Plot positions
    x_positions, labels = [], []
    current_pos = 0
    cluster_spacing, subdomain_spacing, bar_width = 1.0, 0.3, 0.2

    plt.figure(figsize=(16, 8))

    for domain in unique_domains:
        subdf = pivoted.filter(pl.col("domain") == domain)
        n_sub = subdf.height
        positions = current_pos + np.arange(n_sub) * subdomain_spacing
        x_positions.extend(positions)
        labels.extend(subdf["subdomain"].to_list())

        dark_color, light_color = domain_colors(domain)
        bottom = np.zeros(n_sub)
        # Success = dark, Failure = light
        for status, color in zip(["success","failed"], [dark_color, light_color]):
            plt.bar(positions, subdf[status].to_numpy(), bottom=bottom, width=bar_width, color=color)
            bottom += subdf[status].to_numpy()

        # Add success % on top
        for x, rate, tot in zip(positions, subdf["success_rate"].to_numpy(), subdf["total"].to_numpy()):
            plt.text(x, tot + 0.5, f"{rate*100:.0f}%", ha="center", va="bottom", fontsize=9)

        current_pos = positions[-1] + cluster_spacing

    plt.xticks(x_positions, labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Pipeline Counts")
    plt.title("Pipeline Success vs Failed per Subdomain (Grouped by Domain)")

    # Legend: only domain base colors
    domain_handles = [plt.Rectangle((0,0),1,1,color=domain_color_map[d]) for d in unique_domains]
    plt.legend(domain_handles, unique_domains, bbox_to_anchor=(1.05,1), loc="upper left", title="Domain")

    plt.tight_layout()
    filename = f"reports/plots/{prefix}pipeline_success_stacked.png"
    plt.savefig(filename, dpi=400)
    print("Plot saved as reports/plots/pipeline_success_stacked.png")

def plot_avg_dadscore(df: pl.DataFrame, prefix: str = "") -> None:
    """Plot average DADscore per subdomain using Polars."""
    os.makedirs("reports/plots", exist_ok=True)

    # Aggregate: average per subdomain
    avg_scores = (
        df.group_by("subdomain")
          .agg(pl.col("DADscore").mean().alias("avg_dadscore"))
          .sort("avg_dadscore", descending=True)
    )

    # Extract as lists for plotting
    subdomains = avg_scores["subdomain"].to_list()
    scores = avg_scores["avg_dadscore"].to_list()

    plt.figure(figsize=(16, 8))
    sns.barplot(x=subdomains, y=scores, palette="coolwarm")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average DADscore")
    plt.title("Average DADscore per Subdomain")
    plt.tight_layout()

    filename = f"reports/plots/{prefix}avg_dadscore.png"
    plt.savefig(filename, dpi=400)
    print(f"Plot saved as {filename}")

def plot_last_refreshed(df: pl.DataFrame, prefix: str = "") -> None:
    """Clustered horizontal bar chart: days since last refresh per subdomain, grouped by domain, with fresh highlighting."""
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timezone

    os.makedirs("reports/plots", exist_ok=True)

    # Get last refreshed per (domain, subdomain)
    last_dates = (
        df.group_by(["domain", "subdomain"])
          .agg(pl.col("pipeline_updated_at").max().alias("last_refreshed"))
          .sort(["domain", "subdomain"])
    )

    now = datetime.now(timezone.utc)
    rows = []
    for domain, subdomain, last in zip(
        last_dates["domain"], last_dates["subdomain"], last_dates["last_refreshed"]
    ):
        if last is not None:
            if last.tzinfo is None:
                delta = (now.replace(tzinfo=None) - last).days
            else:
                delta = (now - last.astimezone(timezone.utc)).days
        else:
            delta = None
        rows.append((domain, subdomain, delta))

    # Unique domains + color mapping
    unique_domains = list(dict.fromkeys([r[0] for r in rows]))
    colors = sns.color_palette("tab10", len(unique_domains))
    domain_color_map = {d: colors[i] for i, d in enumerate(unique_domains)}

    # Build positions with extra spacing between domains
    y_positions = []
    bar_colors = []
    labels = []
    pos = 0
    cluster_spacing = 1.0
    bar_spacing = 0.4
    bar_height = 0.25  # thinner bars

    for domain in unique_domains:
        domain_rows = [r for r in rows if r[0] == domain]
        for subdomain, (_, _, days) in zip(
            [r[1] for r in domain_rows], domain_rows
        ):
            y_positions.append(pos)
            bar_colors.append(domain_color_map[domain])
            labels.append((subdomain, days, domain))
            pos += bar_spacing
        pos += cluster_spacing  # add space after each domain

    # Plot
    plt.figure(figsize=(14, 10))
    for (subdomain, days, domain), y, color in zip(labels, y_positions, bar_colors):
        plt.barh(y, days if days is not None else 0, height=bar_height, color=color)
        # Add text labels
        if days == 0:
            plt.text(days + 0.2, y, "FRESH", va="center", fontsize=9, color="green",
                     fontweight="bold",
                     bbox=dict(facecolor=color, alpha=0.3, edgecolor="none", boxstyle="round,pad=0.2"))
        elif days is not None:
            plt.text(days + 0.2, y, str(days), va="center", fontsize=8)

    # Set y-ticks
    plt.yticks(y_positions, [sub for sub, _, _ in labels])
    plt.xlabel("Days Since Last Refresh")
    plt.title("Pipeline Staleness per Subdomain (Grouped by Domain)")
    plt.gca().invert_yaxis()

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=domain_color_map[d]) for d in unique_domains]
    plt.legend(handles, unique_domains, title="Domain", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    filename = f"reports/plots/{prefix}last_refreshed.png"
    plt.savefig(filename, dpi=400)
    print(f"Plot saved as {filename}")


def main(csv_path: str) -> None:
    df, prefix = load_data(csv_path)
    df = clean_names(df)
    summarize_data(df)
    plot_branch_flow_violations(df, prefix=prefix)
    plot_pipeline_success_rate(df, prefix=prefix)
    plot_last_refreshed(df, prefix=prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polars-based GitLab pipeline analytics")
    parser.add_argument("csv_path", type=str, nargs="?", default=get_default_csv(),
                        help="Path to the pipeline data CSV")
    args = parser.parse_args()
    main(args.csv_path)