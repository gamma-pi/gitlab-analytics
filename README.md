# GitLab CI/CD Analytics Suite
### Current version: v31.10.25

## Overview

A **comprehensive GitLab CI/CD analytics and reporting system** consisting of four integrated components:

1. **`analytics.py`** - Collects pipeline data from GitLab API with parallel processing
2. **`data_manipulator.py`** - Generates visualizations and analysis using Polars
3. **`confluence.py`** - Publishes reports and charts to Confluence

The suite analyzes repository-level CI/CD metrics including:
- Pipeline success/failure rates and status distribution
- Branch flow violations (dev → SIT → main)
- Pipeline staleness and last refresh times
- Domain and subdomain comparisons
- Job duration and performance statistics

Generated outputs include **PNG visualizations** and **CSV datasets** that can be automatically published to Confluence.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitLab CI/CD Analytics Suite                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
        ┌───────────────┐ ┌──────────────┐ ┌─────────────┐
        │ analytics.py  │ │data_manipu-  │ │confluence.py│
        │               │ │lator.py      │ │             │
        │ Data Collection│ │Visualization │ │Publishing   │
        └───────────────┘ └──────────────┘ └─────────────┘
                │                 │                 │
                ▼                 ▼                 ▼
        ┌───────────────┐ ┌──────────────┐ ┌─────────────┐
        │Pipeline CSV   │ │PNG Charts    │ │Confluence   │
        │Repository     │ │Analytics     │ │Pages        │
        │Metadata       │ │Reports       │ │             │
        └───────────────┘ └──────────────┘ └─────────────┘

```

---

## 📦 Component Details

### 1. analytics.py - Data Collection Engine

**Purpose**: Collects comprehensive CI/CD pipeline data from GitLab API with parallel processing and rate limiting.

**Key Features**:
- **Parallel Processing** - Configurable workers (default: 16) with batching (default: 80)
- **Rate Limiting** - Exponential backoff with jitter for API rate limits
- **Branch Flow Detection** - Validates dev → SIT → main/master patterns
- **Project ID Mapping** - Maps domains/subdomains to application IDs
- **Memory Management** - Batch processing with garbage collection
- **Retry Logic** - Handles 429, 502, 503, 504 errors with exponential backoff

**Data Collected**:
- Pipeline status, duration, timestamps
- Job details (name, status, duration, stage)
- Branch information and merge requests
- Branch flow violations
- Domain and subdomain metadata
- Project descriptions

**Excluded Jobs**:
```python
EXCLUDED_JOBS = []
```

**Output**: CSV file in `reports/pipeline_data_{env}_{days}days.csv`

---

### 2. data_manipulator.py - Visualization Engine

**Purpose**: Generates analytics visualizations and performs data analysis using Polars dataframes.

**Key Features**:
- **Polars-Based Processing** - Fast dataframe operations
- **Auto-Detection** - Finds most recent CSV in `reports/` directory
- **Environment Prefixing** - Adds `[prod]_` or `[nprod]_` to plot filenames
- **Data Cleaning** - Normalizes domain/subdomain names with corrections
- **Project ID Analysis** - Detailed breakdown of projectID coverage

**Domain Corrections Map**:
```python
CORRECTIONS_MAP = {}
```

**Generated Visualizations**:

1. **Branch Flow Violations** (`branch_flow_violations.png`)
   - Clustered bar chart grouped by domain
   - Shows violation counts per subdomain
   - Domain-based color coding with legend

2. **Pipeline Success Rate** (`pipeline_success_stacked.png`)
   - Stacked bar chart: success (dark) vs failed (light)
   - Success percentage displayed on top of bars
   - Domain-based color families


4. **Last Refreshed** (`last_refreshed.png`)
   - Horizontal bar chart showing pipeline staleness
   - Days since last refresh per subdomain
   - "FRESH" highlighting for 0-day staleness
   - Grouped by domain with cluster spacing

5. **Pipeline Status** (`pipeline_status_stacked.png`)
   - Stacked bars: running (green) vs not running (red)
   - Running percentage displayed
   - Background shading for domain clusters

**Output**: PNG files in `reports/plots/` directory

---

### 3. confluence.py - Publishing System

**Purpose**: Publishes analytics reports and visualizations to Confluence with multi-authentication support.

**Key Features**:
- **Multiple Auth Methods** - PAT, API Token, Basic Auth
- **Automated Publishing** - Bulk upload of reports and charts
- **Version Management** - Automatic page versioning
- **Attachment Handling** - Smart upload with duplicate detection
- **HTML Conversion** - Transforms HTML reports to Confluence storage format
- **Static Dashboard Generation** - Creates image-based dashboards from PNGs

**Authentication Types**:
```json
{
  "confluence": {
    "auth": {
      "type": "pat",           // Personal Access Token (recommended)
      "token": "..."
    }
  }
}
```

**Supported Report Types**:
- **Domain Comparison** - HTML report with cross-domain metrics
- **Repository Health** - Static dashboard with embedded PNG charts

**Published Content**:
- Converts HTML to Confluence storage format with XHTML preservation
- Uploads PNG attachments with version control
- Creates image macros referencing attached files
- Maintains page hierarchy and metadata

---

### 4. subdomain_projectID_map.json - Configuration

**Purpose**: Maps GitLab domains and subdomains to project application IDs for maturity score integration.

**Structure**:
```json
{
  "Domain Name": {
    "appID": "1234567",
    "subdomains": {
      "Subdomain Name": {"appID": "1234567"}
    }
  }
}
```

**Supported Domains**:
- Deliver (Shipment, Fulfillment Centers, Carriers, Trade, etc.)
- E2E Supply Chain (Item/Product/Site/Supplier Attribution)
- Engineering & Design (Configure, Design, Lifecycle, Quality, etc.)
- Make (Inventory, Order Management, Production)
- Plan (Build Plan, Demand Plan, S&OP, Supply Chain)
- Source (Commodity Management, Contracting, P2P, etc.)
- Reference Data (Calendar, Geography, Currency Exchange)
- External Purchased Data (3rd Party Analysis, Market Info, Weather)

**Matching Logic**:
- Case-insensitive matching with normalization
- Whitespace and underscore handling
- Fallback to domain-level appID if subdomain not found
- Logs unmatched domains/subdomains for troubleshooting

---

## ⚙️ Configuration

### GitLab Settings
```python
gitlab_url = 'https://gitlab.com/'
pem_file = "utils/cert.pem"
```

### Confluence Configuration (`config.json`)
```json
{
  "confluence": {
    "base_url": "https://confluence.example.com",
    "space_key": "DEVOPS",
    "auth": {
      "type": "pat",
      "token": "your_personal_access_token"
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

---

## 🚀 Usage Guide

### Workflow 1: Data Collection Only
```bash
# Collect production data for last 30 days
python3 analytics.py --env prod --days_back 30

# Collect with custom settings
python3 analytics.py \
    --env prod \
    --days_back 60 \
    --max_pipelines_per_repo 100 \
    --max_workers 16 \
    --batch_size 80
```

**Output**: `reports/pipeline_data_prod_30days.csv`

---

### Workflow 2: Data Collection + Visualization
```bash
# Step 1: Collect data
python3 analytics.py --env prod --days_back 30

# Step 2: Generate visualizations
python3 data_manipulator.py reports/pipeline_data_prod_30days.csv

# Or let it auto-detect the latest CSV
python3 data_manipulator.py
```

**Output**: 
- `reports/plots/[prod]_branch_flow_violations.png`
- `reports/plots/[prod]_pipeline_success_stacked.png`
- `reports/plots/[prod]_last_refreshed.png`
- `reports/plots/[prod]_pipeline_status_stacked.png`

---

### Workflow 3: Complete Pipeline with Publishing
```bash
# Step 1: Collect data
python3 analytics.py --env prod --days_back 30

# Step 2: Generate visualizations
python3 data_manipulator.py

# Step 3: Publish to Confluence
python3 confluence.py --env prod --reports-dir reports/
```

---

### Workflow 4: Automated End-to-End (Recommended)
```bash
# Run analytics, then publish
python3 confluence.py --run-analytics --env prod --days-back 30
```

---

## 📋 Command Reference

### analytics.py Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--env` | `prod` | `prod`, `nprod` | Target environment (production or non-production) |
| `--days_back` | `30` | `1-365` | Number of days to analyze historically |
| `--max_pipelines_per_repo` | `100` | `1-1000` | Maximum pipelines to fetch per repository |
| `--max_workers` | `16` | `4-32` | Number of parallel worker threads |
| `--batch_size` | `80` | `20-200` | Repositories processed per batch |
| `--csv_path` | `auto` | `path/to/file.csv` | Custom output CSV path |
| `--no_csv` | `False` | flag | Skip CSV file generation |

**Example - High Performance**:
```bash
python3 analytics.py \
    --env prod \
    --max_workers 24 \
    --batch_size 120 \
    --max_pipelines_per_repo 150
```

**Example - Quick Analysis**:
```bash
python3 analytics.py \
    --env prod \
    --days_back 7 \
    --max_pipelines_per_repo 20 \
```

---

### data_manipulator.py Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_path` | Most recent in `reports/` | Path to pipeline data CSV file |

**Example - Explicit CSV**:
```bash
python3 data_manipulator.py reports/pipeline_data_prod_30days.csv
```

**Example - Auto-detect Latest**:
```bash
python3 data_manipulator.py
```

**Console Output Includes**:
- Total records loaded
- Domain and repository counts
- Pipeline status distribution
- Branch flow violation summary
- Project ID analysis (null counts, unique IDs, coverage)
- Data quality warnings

---

### confluence.py Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `config.json` | Path to Confluence configuration file |
| `--env` | `prod` | Environment for report publishing |
| `--reports-dir` | `reports` | Directory containing reports to publish |
| `--run-analytics` | `False` | Execute analytics.py before publishing |
| `--days-back` | `30` | Days to analyze (with `--run-analytics`) |
| `--max-pipelines` | `10` | Pipeline limit (with `--run-analytics`) |
| `--single-file` | - | Publish a specific HTML file |
| `--page-title` | - | Title for single file upload (required with `--single-file`) |

**Example - Publish Existing Reports**:
```bash
python3 confluence.py --env prod
```

**Example - Run Analytics Then Publish**:
```bash
python3 confluence.py --run-analytics --env prod --days-back 60
```

**Example - Single File Upload**:
```bash
python3 confluence.py \
    --single-file reports/custom_report.html \
    --page-title "Custom CI/CD Report - Q4 2025"
```

---

## 📊 Output Structure

```
reports/
├── pipeline_data_prod_30days.csv           # Raw collected data
├── pipeline_data_nprod_30days.csv          # Non-prod data
├── plots/                                   # Generated visualizations
│   ├── [prod]_branch_flow_violations.png
│   ├── [prod]_pipeline_success_stacked.png
│   ├── [prod]_last_refreshed.png
│   ├── [prod]_pipeline_status_stacked.png
│   ├── [nprod]_branch_flow_violations.png
│   └── [nprod]_pipeline_success_stacked.png
├── domain_comparison_prod.html             # Domain summary report
├── repository_health_prod.html             # Interactive dashboard
└── logs/
    ├── cicd_analytics.log                  # Analytics execution logs
    └── confluence_publisher.log            # Publishing logs
```

---

## 📈 Data Schema

### Pipeline CSV Columns

**Domain & Repository**:
- `repo_name` - Repository name
- `repo_id` - GitLab repository ID
- `domain_project_description` - Domain description from GitLab
- `subdomain_project_description` - Subdomain description

**Pipeline Metadata**:
- `pipeline_id` - Unique pipeline identifier
- `pipeline_status` - Status (success, failed, running, canceled, skipped)
- `pipeline_created_at` - Pipeline creation timestamp
- `pipeline_updated_at` - Last update timestamp
- `pipeline_duration` - Total pipeline duration (seconds)
- `branch_name` - Git branch name
- `commit_sha` - Git commit SHA
- `branch_flow_violation` - 1 if flow violated, 0 otherwise

**Merge Request Data**:
- `merge_request_id` - MR ID if applicable
- `merge_request_source_branch` - Source branch
- `merge_request_target_branch` - Target branch
- `merge_request_state` - MR state (merged, opened, closed)

**Job Details**:
- `job_name` - CI/CD job name
- `job_status` - Job status
- `job_duration` - Reported job duration
- `job_actual_duration` - Calculated duration (finished - started)
- `job_stage` - Pipeline stage
- `job_created_at` - Job creation time
- `job_started_at` - Job start time
- `job_finished_at` - Job completion time

---

## 🔍 Branch Flow Validation

The system validates standard GitLab branch flow patterns:

### Valid Flows
```
Feature → dev → SIT → main/master
```

### Detection Logic

**SIT Branch**:
- ✅ Valid: `dev` → `SIT`
- ❌ Violation: Any other source → `SIT`

**Main/Master Branch**:
- ✅ Valid: `SIT` → `main/master`
- ❌ Violation: Any other source → `main/master`

**Dev Branch**:
- ✅ Valid: Any non-dev branch → `dev`
- ❌ Violation: `dev` → `dev` (self-merge)

---

## 🎨 Visualization Details

### 1. Branch Flow Violations
- **Type**: Clustered vertical bar chart
- **X-axis**: Subdomains (rotated labels)
- **Y-axis**: Total violation count
- **Colors**: Tab10 palette, one color per domain
- **Annotations**: Violation counts displayed on bars
- **Legend**: Domain color mapping

### 2. Pipeline Success Rate
- **Type**: Stacked bar chart
- **X-axis**: Subdomains grouped by domain
- **Y-axis**: Pipeline counts
- **Colors**: 
  - Dark shade = Success
  - Light shade = Failed
  - Same color family per domain
- **Annotations**: Success percentage on top of stacks

### 4. Last Refreshed (Staleness)
- **Type**: Horizontal bar chart with clustering
- **X-axis**: Days since last refresh
- **Y-axis**: Subdomains grouped by domain
- **Colors**: Domain-based color mapping
- **Special Indicators**: 
  - "FRESH" badge for 0-day staleness
  - Green bold text with colored background
- **Spacing**: Extra spacing between domain clusters

### 5. Pipeline Status
- **Type**: Stacked vertical bar chart
- **X-axis**: Subdomains grouped by domain
- **Y-axis**: Pipeline counts
- **Colors**:
  - Green (#247624) = Running
  - Red (#8e3333) = Not Running
- **Background**: Alternating gray shading per domain cluster
- **Annotations**: Running percentage on top

---

## 🛡️ Error Handling & Resilience

### Rate Limiting Strategy
```python
# Exponential backoff with jitter
base_delay = 1 second
max_delay = 300 seconds
jitter = random.uniform(0.75, 1.25)

delay = min(base_delay * (2 ** attempt), max_delay) * jitter
```

### Retry Logic
- **429 (Rate Limited)**: Respects `Retry-After` header or uses exponential backoff
- **502/503/504 (Server Errors)**: Automatic retry with exponential backoff
- **Timeout/Connection Errors**: Retry with backoff
- **404 (Not Found)**: No retry, logs and continues
- **401/403 (Auth Errors)**: No retry, logs error

### Memory Management
- **Batch Processing**: Processes repositories in batches (default: 80)
- **Garbage Collection**: Explicit `gc.collect()` between batches
- **Thread-Safe Data**: Locks protect shared data structures
- **Progress Tracking**: Thread-safe counter with logging every 10 repos

---

## 🔧 Dependencies

### Core Requirements
```bash
pip install requests pandas polars numpy matplotlib seaborn plotly
pip install beautifulsoup4 selenium webdriver-manager python-dateutil
```

### System Requirements
- **Python**: 3.8+
- **Memory**: 8GB minimum, 16GB+ recommended for large datasets
- **Network**: Stable connection to GitLab API
- **Certificate**: Corporate SSL certificate in `utils/cert.pem`

---

## 🚨 Troubleshooting

### Issue: "No CSV files found in reports/"
**Solution**:
```bash
# Run analytics first to generate CSV
python3 analytics.py --env prod
```

### Issue: "Certificate file not found"
**Solution**:
```bash
# Place certificate at utils/cert.pem
# Or temporarily disable SSL verification (not recommended for production)
```

### Issue: "Memory error during large data collection"
**Solution**:
```bash
# Reduce batch size and workers
python3 analytics.py \
    --env prod \
    --max_workers 8 \
    --batch_size 40 \
    --max_pipelines_per_repo 50
```

### Issue: "Confluence authentication failed"
**Solution**:
```bash
# Verify config.json authentication type and token
# Ensure base_url ends without trailing slash
# Check token permissions in Confluence
```

### Issue: "Unmatched domain/subdomain in projectID mapping"
**Solution**:
```bash
# Check logs for unmatched entries
# Update subdomain_projectID_map.json with correct mappings
# Ensure name normalization matches (strip, title case)
```

---

## 📁 Project Structure

```
project/
├── analytics.py                        # Data collection engine
├── data_manipulator.py                 # Visualization generator
├── confluence.py                       # Confluence publisher
├── config.json                         # Confluence API config
├── subdomain_projectID_map.json        # Project ID mappings
├── README.md                           # This documentation
├── requirements.txt                    # Python dependencies
├── utils/
│   ├── logger.py                       # Logging utilities
│   └── cert.pem                        # SSL certificate bundle
├── common/
│   └── gitlab_utils.py                 # GitLab API utilities
├── reports/                            # Generated outputs
│   ├── plots/                          # PNG visualizations
│   ├── *.csv                           # Pipeline datasets
│   └── *.html                          # HTML reports
└── logs/
    ├── cicd_analytics.log              # Analytics logs
    └── confluence_publisher.log        # Publishing logs
```

---

## 🎯 Performance Optimization

### Hardware-Based Recommendations

| System Specs | Recommended Settings |
|--------------|---------------------|
| **4-8 CPU cores, 8GB RAM** | `--max_workers 6`, `--batch_size 50` |
| **8-16 CPU cores, 16GB RAM** | `--max_workers 12`, `--batch_size 100` |
| **16+ CPU cores, 32GB+ RAM** | `--max_workers 24`, `--batch_size 150` |

### Network Optimization
- **High-Speed Network**: Increase workers and batch size
- **Rate-Limited GitLab**: Reduce workers (4-8), system handles backoff automatically
- **Corporate Proxy**: Ensure certificate bundle configured, consider SSL verification settings

### Data Size Optimization
- **Small Analysis (< 100 repos)**: Use defaults
- **Medium Analysis (100-500 repos)**: `--batch_size 100`, `--max_workers 12`
- **Large Analysis (500+ repos)**: `--batch_size 150`, `--max_workers 16-24`

---

## 📊 Sample Workflow: Complete Analysis

```bash
#!/bin/bash
# Complete CI/CD analytics workflow

echo "=== Step 1: Data Collection ==="
python3 analytics.py \
    --env prod \
    --days_back 30 \
    --max_pipelines_per_repo 100 \
    --max_workers 16 \
    --batch_size 80

echo ""
echo "=== Step 2: Data Visualization ==="
python3 data_manipulator.py

echo ""
echo "=== Step 3: Publish to Confluence ==="
python3 confluence.py --env prod

echo ""
echo "✅ Analysis complete! Check:"
echo "  - CSV: reports/pipeline_data_prod_30days.csv"
echo "  - Charts: reports/plots/"
echo "  - Logs: logs/cicd_analytics.log"
```

---

## 👥 Authors
**MAVSCM DevOps Team**

## 📅 Version History
- **30-Jul-2025** - Initial analytics.py release with parallel processing
- **06-Aug-2025** - Added confluence.py integration
- **11-Aug-2025** - Enhanced publishing workflow
- **31-Oct-2025** - Complete rewrite with data_manipulator.py (Polars), improved documentation

---

## 📋 Release Notes - v31.10.25

### New Features
- **Polars-Based Analytics** (`data_manipulator.py`)
  - 10-50x faster dataframe operations vs pandas
  - Memory-efficient processing of large datasets
  - Auto-detection of latest CSV files
  - Environment-aware plot naming

- **Enhanced Visualizations**
  - 5 comprehensive chart types
  - Domain clustering and color coding
  - "FRESH" highlighting for active pipelines
  - Stacked success/failure analysis
  - Pipeline staleness metrics

### Improvements
- **Better Error Handling**
  - Comprehensive retry logic for API failures
  - Detailed logging for troubleshooting

- **Performance Enhancements**
  - Configurable batch processing (default: 80)
  - Thread-safe progress tracking
  - Memory cleanup between batches
  - Optimized API request patterns

- **Documentation**
  - Complete workflow examples
  - Troubleshooting guide
  - Performance optimization guidelines
  - Architecture diagrams

### Bug Fixes
- Fixed domain/subdomain name normalization
- Resolved SSL certificate handling issues
- Corrected projectID mapping logic
- Fixed CSV auto-detection edge cases

---

## 🔒 Security Considerations

- **API Tokens**: Store in `config.json` (ensure `.gitignore` includes this file)
- **SSL Verification**: Enable in production, disable only for testing
- **Access Control**: Confluence tokens should have minimum required permissions
- **Logging**: Sensitive data is not logged (tokens, passwords)
- **Certificate Management**: Keep `utils/cert.pem` updated with corporate certificates

---

## 📖 Additional Resources

- **GitLab API Documentation**: https://docs.gitlab.com/ee/api/
- **Confluence REST API**: https://developer.atlassian.com/server/confluence/confluence-rest-api-examples/
- **Polars Documentation**: https://pola-rs.github.io/polars/
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/

---

## 💡 Tips & Best Practices

1. **Start Small**: Test with `--days_back 7` and `--max_pipelines_per_repo 10` first
3. **Monitor Logs**: Check `logs/` directory for detailed execution info
4. **Batch Size Tuning**: Adjust based on available memory and network speed
5. **Environment Separation**: Always use `--env` flag to separate prod/nprod data
6. **Confluence Page IDs**: Pre-create pages and note IDs in `config.json`
7. **Regular Updates**: Keep `subdomain_projectID_map.json` synchronized with organizational changes

---

**For questions or issues, contact the MAVSCM DevOps Team.**
