# GitLab CI/CD Analytics Tool

## Current version: v11.08.25

## Overview

`analytics.py` is a **comprehensive GitLab CI/CD analytics and reporting utility**.  
It collects repository-level CI/CD pipeline data, analyzes metrics such as:

- Pipeline success/failure rates
- Branch protection coverage
- Merge request approval rules
- Repository health scores
- Job duration statistics

It then generates **visual reports** (HTML dashboards and plots) and can optionally **upload results to Confluence**.

With the **`--publish`** parameter, the tool can **automatically run `confluence.py` after completing analytics**, enabling a seamless end-to-end workflow:  
1. Run analytics and generate reports.  
2. Immediately publish to Confluence.  

## 🔧 Key Features
- Parallel repository analysis with configurable workers
- Dynamic batch processing
- Interactive HTML dashboards
- Domain comparison reports
- Resource-aware throttling

## 📊 Repository Health Score

Each repository is scored (0–100) based on:

| Metric                | Weight |
|-----------------------|--------|
| Pipeline Success Rate | 0.4    |
| Branch Protection     | 0.2    |
| Approval Rules        | 0.2    |
| Test Coverage         | 0.1    |
| Avg Pipeline Duration | 0.1    |

## ⚙️ Configuration

- **GitLab URL**: `https://gitlab.com/`
- **Cert File**: `utils/cert.pem`
- **Authentication**: Token via `private_token_svc` in `gitlab_utils.py`

### Default Settings (Intel Core Ultra 5 8C/32GB)
```python
def main(
    max_workers: int = 10,     # 75% of logical cores
    batch_size: int = 100,      # ~15GB RAM usage
    days_back: int = 30,
    env: str = "prod"
):
```

### Runtime Configuration
Execute with custom parameters:

```bash
python3 analytics.py \
    --max_workers 12 \      # Adjust worker count (4-14)
    --batch_size 150 \      # Adjust batch size (50-200)
    --env nprod \           # Environment (prod/nprod)
    --days_back 60          # Analysis timeframe
Parameter Reference
Parameter	Default	Range	Description
--max_workers	10	4-14	Parallel threads (75-90% CPU)
--batch_size	100	50-200	Repos per batch
--env	prod	prod/nprod	Environment mode
--days_back	30	1-365	Days of history to analyze
```

## Resource Management
The tool includes:

- Automatic garbage collection between batches

- Optional CPU/RAM monitoring

- GitLab API rate limit handling

## 📂 Output

The below HTML file reports are generated for each env = [prod, nprod]

- reports/domain_comparison_[env].html : Domain-wise summary report
- reports/repository_health_[env].html : : Interactive dashboard (with boxplots)

## Best Practices
- Start with defaults, then increase workers/batches
- Monitor system resources during first runs

#### Analytics + Upload to Confluence
```bash
python analytics.py --env prod --upload_confluence
```

#### Upload Only (skip analysis, upload existing reports)
```bash
python analytics.py --upload_only --upload_confluence
```

#### Analytics + Automatic Publish via confluence.py
```bash
python analytics.py --env prod --publish

## 📁 Project Structure

```bash
project/
│
├── utils/
│   └── logger.py
│   └── cert.pem
├── common/
│   └── gitlab_utils.py
├── reports/
│   └── plots/
│         └── *.png
│   └── *.html
├── analytics.py
├── confluence.py
├── config.json
├── README.md
├── requirements.txt
```

---

## 🛠 Dependencies

```bash
pip install requests pandas numpy matplotlib seaborn plotly
```

## Complete Execution Parameters

### Core Analysis Parameters
| Parameter               | Default | Valid Range      | Description |
|-------------------------|---------|------------------|-------------|
| `--env`                 | `prod`  | `prod`/`nprod`   | Analysis environment |
| `--days_back`           | `30`    | `1-365`          | Days of historical data to analyze |
| `--max_pipelines_per_repo` | `10` | `1-100`       | Max pipelines analyzed per repository |

### Parallel Processing Control
| Parameter               | Default | Recommended Range | Description |
|-------------------------|---------|-------------------|-------------|
| `--max_workers`         | `10`    | `4-14`            | Thread count (75-90% CPU utilization) |
| `--batch_size`          | `100`   | `50-200`          | Repositories processed per batch |

### Domain Filtering
| Parameter               | Default          | Example Values       | Description |
|-------------------------|------------------|----------------------|-------------|
| `--prod_group_id`       | `"xxxxxxx"`       | GitLab group IDs     | Production group ID |
| `--exclude_domains`     | `['central-team-repository']` | Comma-separated list | Domains to exclude |

### Output Control
| Parameter               | Default          | Options             | Description |
|-------------------------|------------------|---------------------|-------------|
| `--output_dir`          | `"reports/"`     | Any valid path      | Report output directory |
| `--log_level`           | `"INFO"`         | `DEBUG/INFO/WARNING/ERROR` | Logging verbosity |

---

## 🛰️ Confluence Integration

The tool now supports **automated publishing to Confluence**, enabling seamless collaboration and visibility of CI/CD analytics.

### ✅ What It Does

- Converts HTML reports to Confluence-compatible storage format
- Uploads static PNG dashboards for environments (`prod` or `nprod`)
- Updates or creates Confluence pages using `page_id` from `config.json`
- Supports:
  - Domain comparison tables (HTML to Confluence)
  - Repository health dashboards (PNG images)

### 🔐 Authentication Support

- **PAT (Personal Access Token)** – Recommended for Atlassian Cloud
- **API Token + Email**
- **Basic Auth (Username + Password)**

Configure via `config.json`:
```json
{
  "confluence": {
    "base_url": "https://confluence.example.com",
    "space_key": "DEVOPS",
    "auth": {
      "type": "pat",
      "token": "your_confluence_token"
    },
    "reports": {
      "publish_page_ids": {
        "prod": {
          "domain_comparison": "123456",
          "repository_health": "234567"
        },
        "nprod": {
          "domain_comparison": "345678",
          "repository_health": "456789"
        }
      }
    }
  }
}
```

## 🖥️ New CLI Options (confluence.py)

```bash
python confluence.py [OPTIONS]
```

### 🔧 Options
Argument	Description
--env	Environment: prod or nprod
--run-analytics	Run analytics before publishing
--days-back	Days to look back (used with --run-analytics)
--max-pipelines	Max pipelines per repo (used with --run-analytics)
--reports-dir	Custom directory for reports (default: reports/)
--single-file	Publish one HTML file manually
--page-title	Title for single file upload (used with --single-file)
--config	Path to config.json for Confluence credentials

```bash
python3 analytics.py --env prod --days_back 60
```

### High-performance Non-Prod Scan
```bash
python3 analytics.py \
  --env nprod \
  --max_workers 14 \
  --batch_size 200 \
  --max_pipelines_per_repo 20 \
  --exclude_domains "test-domain,sandbox"
  ```

### Debug mode with Verbose Logging

```bash
python3 analytics.py \
  --log_level DEBUG \
  --api_delay 1.5 \
  --retry_count 5
```

## 🖼️ PNG Dashboard Charts
For Confluence compatibility, interactive Plotly dashboards are exported as static PNG images in:

```bash
reports/plots/
  ├── health_score_prod.png
  ├── success_rate_prod.png
  ├── ...
```
These are:

Attached to the corresponding Confluence page

Embedded in the page content using <ac:image> macros

---

## Release Notes
- Confluence Integration Added:

    - Publishes analytics reports directly to Confluence

    - Supports both domain comparison tables and dashboard charts

- Static PNG Dashboard Support:

    - Interactive charts converted to PNG for Confluence compatibility

- Automated Attachment Uploads:

    - PNG charts are attached and embedded in Confluence pages

- Flexible Auth Methods:

    - Supports PAT, API token + email, or basic auth for Confluence

- New CLI (confluence.py):

    - Allows publishing reports with or without re-running analytics

    - Supports single-file HTML upload to any Confluence page

- Config-Driven Publishing:

    - Uses config.json to define target Confluence pages for each environment
