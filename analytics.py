import os
import sys
import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from dateutil import parser as date_parser
import random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

sys.path.append(os.getcwd())

from utils.logger import set_logg_handlers_to_console_and_file, log_function_data
from common.gitlab_utils import (
    get_subgroups, 
    get_repositories, 
    private_token_svc, 
    get_headers,
    get_protected_branches,
    get_branches_in_repo,
    get_approval_rules,
    process_subdomains
)

EXCLUDED_JOBS = [
   
]

logger = set_logg_handlers_to_console_and_file("logs/cicd_analytics.log")

gitlab_url = 'https://gitlab.dell.com/'
pem_file = "utils/cert.pem"

# -----------------------------
# Helper functions
# -----------------------------
def normalize_name(name: str) -> str:
    """Normalize GitLab domain/subdomain names to match JSON keys."""
    if not name:
        return ""
    return name.replace("_", " ").strip().title()

# -----------------------------
# DAD Score Fetching Functions
# -----------------------------

def init_browser(headless: bool = True):
    """Initialize Chrome browser for Selenium."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    return driver

def fetch_dad_score_for_project(project_id: str, config_path="config.json") -> float:
    """Fetch DAD score using API call instead of Selenium."""
    if not project_id:
        return np.nan

    with open(config_path, "r") as f:
        config = json.load(f)

    dad_api_config = config.get("confluence", {}).get("reports", {}).get("dad_api", {})
    base_url = dad_api_config.get("base_url")
    token = dad_api_config.get("access_token")

    if not base_url or not token:
        logger.error("DAD API configuration missing")
        return np.nan

    url = f"{base_url}/{project_id}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Adjust the key here based on API response JSON structure
            score = data.get("maturityScore") or data.get("score")
            return float(score) if score is not None else np.nan
        else:
            logger.error(f"DAD API request failed for {project_id}: {response.status_code}")
            return np.nan
    except Exception as e:
        logger.error(f"Error fetching DAD score for {project_id}: {e}")
        return np.nan

def add_dad_scores(df, config_path="config.json"):
    """Fetch DAD scores for all subdomains using API."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Flatten projectID column: use 'appID' if it's a dict
    df["projectID"] = df["projectID"].apply(lambda x: x['appID'] if isinstance(x, dict) and 'appID' in x else x)

    project_ids = [pid for pid in df["projectID"].to_list() if pid]

    if not project_ids:
        logger.warning("No valid project IDs found")
        df["DADscore"] = np.nan
        return df

    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_pid = {
            executor.submit(fetch_dad_score_for_project, pid, config_path): pid
            for pid in project_ids
        }

        for future in as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                score = future.result()
            except Exception as e:
                logger.error(f"Error fetching DADscore for project {pid}: {e}")
                score = np.nan
            results[pid] = score
            logger.info(f"DADscore fetched for project {pid}: {score}")

    # Map scores back to DataFrame
    df["DADscore"] = df["projectID"].map(lambda pid: results.get(pid, np.nan))
    return df

def get_group_description(group_id: int) -> Optional[str]:
    """Fetch GitLab group/subgroup description."""
    url = f"{gitlab_url}api/v4/groups/{group_id}"
    response = requests.get(url, headers=get_headers(), verify=pem_file, timeout=30)
    if response.status_code == 200:
        return response.json().get('description', None)
    else:
        logger.warning(f"Failed to fetch description for group {group_id}: {response.status_code}")
        return None
    
def apply_projectID_mapping(df: pd.DataFrame, mapping_path: str = "subdomain_projectID_map.json") -> pd.DataFrame:
    with open(mapping_path, "r") as f:
        project_map = json.load(f)

    unmatched = []

    def resolve_projectID(row):
        domain = normalize_name(row["domain"])
        subdomain = normalize_name(row["subdomain"]) if pd.notna(row["subdomain"]) else None
        domain_entry = next((d for d in project_map if normalize_name(d) == domain), None)
        if not domain_entry:
            unmatched.append((row["domain"], row["subdomain"]))
            return None
        domain_data = project_map[domain_entry]
        if subdomain and "subdomains" in domain_data:
            sub_entry = next((s for s in domain_data["subdomains"] if normalize_name(s) == subdomain), None)
            if sub_entry:
                return domain_data["subdomains"][sub_entry].get("appID") or domain_data.get("appID")
        return domain_data.get("appID")

    df["projectID"] = df.apply(resolve_projectID, axis=1)
    if unmatched:
        logger.warning(f"Unmatched domain/subdomains: {unmatched}")
    return df


class ThreadSafeCounter:
    """Thread-safe counter for tracking progress."""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

class GitLabDataFrameCollector:
    """
    GitLab CI/CD data collector that builds a comprehensive DataFrame.
    """
    
    def __init__(self, max_workers: int = 16, batch_size: int = 80):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._data_lock = threading.Lock()
        self._progress_counter = ThreadSafeCounter()
        self.pipeline_data = []
        self.max_retries = 5
        self.base_delay = 1
        self.max_delay = 300
        
    def _calculate_backoff_delay(self, attempt: int, base_delay: float = None, max_delay: float = None) -> float:
        if base_delay is None:
            base_delay = self.base_delay
        if max_delay is None:
            max_delay = self.max_delay
        exponential_delay = base_delay * (2 ** (attempt - 1))
        capped_delay = min(exponential_delay, max_delay)
        jitter = random.uniform(0.75, 1.25)
        return capped_delay * jitter

    def _make_gitlab_request(self, url: str, params: Optional[Dict] = None, max_retries: int = None) -> Optional[Dict]:
        if max_retries is None:
            max_retries = self.max_retries
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=get_headers(), params=params, verify=pem_file, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    server_retry_after = response.headers.get('Retry-After')
                    if server_retry_after:
                        try:
                            delay = min(int(server_retry_after), self.max_delay)
                        except ValueError:
                            delay = self._calculate_backoff_delay(attempt)
                    else:
                        delay = self._calculate_backoff_delay(attempt)
                    if attempt < max_retries:
                        logger.warning(f"Rate limited, retry {attempt}/{max_retries} in {delay:.1f}s: {url}")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Rate limited on final attempt {attempt}: {url}")
                        return None
                elif response.status_code in [502, 503, 504]:
                    if attempt < max_retries:
                        delay = self._calculate_backoff_delay(attempt)
                        logger.warning(f"Server error {response.status_code}, retry {attempt}/{max_retries} in {delay:.1f}s: {url}")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Server error {response.status_code} on final attempt: {url}")
                        return None
                elif response.status_code == 404:
                    logger.debug(f"Resource not found (404): {url}")
                    return None
                elif response.status_code in [401, 403]:
                    logger.error(f"Authentication error {response.status_code}: {url}")
                    return None
                else:
                    logger.warning(f"API request failed {response.status_code}: {url}")
                    return None
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Timeout retry {attempt}/{max_retries} in {delay:.1f}s: {url}")
                    time.sleep(delay)
                    last_exception = "Timeout"
                    continue
                else:
                    logger.error(f"Timeout on final attempt: {url}")
                    return None
            except requests.exceptions.ConnectionError:
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Connection error retry {attempt}/{max_retries} in {delay:.1f}s: {url}")
                    time.sleep(delay)
                    last_exception = "Connection error"
                    continue
                else:
                    logger.error(f"Connection error on final attempt: {url}")
                    return None
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Request exception retry {attempt}/{max_retries} in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    last_exception = str(e)
                    continue
                else:
                    logger.error(f"Request exception on final attempt: {e}")
                    return None
        logger.error(f"All attempts failed for {url}. Last: {last_exception}")
        return None

    def get_project_pipelines(self, project_id: int, days_back: int = 30, max_pipelines: int = 100) -> List[Dict]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        all_pipelines = []
        page = 1
        per_page = min(max_pipelines, 100)
        while len(all_pipelines) < max_pipelines:
            url = f"{gitlab_url}api/v4/projects/{project_id}/pipelines"
            params = {
                'updated_after': start_date.isoformat(),
                'updated_before': end_date.isoformat(),
                'per_page': per_page,
                'page': page,
                'order_by': 'updated_at',
                'sort': 'desc'
            }
            pipelines = self._make_gitlab_request(url, params)
            if not pipelines:
                break
            remaining_slots = max_pipelines - len(all_pipelines)
            all_pipelines.extend(pipelines[:remaining_slots])
            if len(all_pipelines) >= max_pipelines or len(pipelines) < per_page:
                break
            page += 1
        return all_pipelines

    def get_pipeline_jobs(self, project_id: int, pipeline_id: int) -> List[Dict]:
        url = f"{gitlab_url}api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs?per_page=100"
        jobs = self._make_gitlab_request(url) or []
        for job in jobs:
            if job.get('started_at') and job.get('finished_at'):
                started = date_parser.isoparse(job['started_at'])
                finished = date_parser.isoparse(job['finished_at'])
                job['actual_duration'] = (finished - started).total_seconds()
            else:
                job['actual_duration'] = job.get('duration', 0)
        return jobs

    def get_merge_request_for_pipeline(self, project_id: int, pipeline_sha: str) -> Optional[Dict]:
        url = f"{gitlab_url}api/v4/projects/{project_id}/merge_requests"
        params = {'state': 'all', 'order_by': 'updated_at', 'sort': 'desc', 'per_page': 50}
        merge_requests = self._make_gitlab_request(url, params) or []
        for mr in merge_requests:
            if mr.get('sha') == pipeline_sha or mr.get('merge_commit_sha') == pipeline_sha:
                return mr
        return None

    def detect_branch_flow_violation(self, branch_name: str, merge_request: Optional[Dict]) -> int:
        if not branch_name:
            return 0
        branch_lower = branch_name.lower()
        if 'sit' in branch_lower:
            if merge_request:
                source_branch = merge_request.get('source_branch', '').lower()
                target_branch = merge_request.get('target_branch', '').lower()
                if 'sit' in target_branch and 'dev' in source_branch:
                    return 0
                else:
                    return 1
            else:
                return 1
        elif 'main' in branch_lower or 'master' in branch_lower:
            if merge_request:
                source_branch = merge_request.get('source_branch', '').lower()
                target_branch = merge_request.get('target_branch', '').lower()
                if ('main' in target_branch or 'master' in target_branch) and 'sit' in source_branch:
                    return 0
                else:
                    return 1
            else:
                return 1
        elif 'dev' in branch_lower:
            if merge_request:
                source_branch = merge_request.get('source_branch', '').lower()
                target_branch = merge_request.get('target_branch', '').lower()
                if 'dev' in target_branch and 'dev' not in source_branch:
                    return 0
                else:
                    return 1
            else:
                return 0
        return 0

    def collect_repository_data(self, repo_data: Tuple[int, str, str, str, str, str, int, int]) -> List[Dict]:
        """
        repo_data: (repo_id, repo_name, subdomain_name, domain_name, domain_desc, subdomain_desc, days_back, max_pipelines)
        """
        repo_id, repo_name, subdomain_name, domain_name, domain_desc, subdomain_desc, days_back, max_pipelines = repo_data
        try:
            records = []
            pipelines = self.get_project_pipelines(repo_id, days_back, max_pipelines)
            for pipeline in pipelines:
                merge_request = None
                if pipeline.get('sha'):
                    merge_request = self.get_merge_request_for_pipeline(repo_id, pipeline['sha'])
                branch_flow_violation = self.detect_branch_flow_violation(pipeline.get('ref', ''), merge_request)
                jobs = self.get_pipeline_jobs(repo_id, pipeline['id'])
                base_record = {
                    'domain': domain_name,
                    'subdomain': subdomain_name,
                    'repo_name': repo_name,
                    'repo_id': repo_id,
                    'domain_project_description': domain_desc,
                    'subdomain_project_description': subdomain_desc,
                    'pipeline_id': pipeline['id'],
                    'pipeline_status': pipeline['status'],
                    'pipeline_created_at': pipeline.get('created_at'),
                    'pipeline_updated_at': pipeline.get('updated_at'),
                    'pipeline_duration': pipeline.get('duration'),
                    'branch_name': pipeline.get('ref'),
                    'commit_sha': pipeline.get('sha'),
                    'branch_flow_violation': branch_flow_violation,
                    'merge_request_id': merge_request.get('id') if merge_request else None,
                    'merge_request_source_branch': merge_request.get('source_branch') if merge_request else None,
                    'merge_request_target_branch': merge_request.get('target_branch') if merge_request else None,
                    'merge_request_state': merge_request.get('state') if merge_request else None,
                }
                if not jobs:
                    record = base_record.copy()
                    record.update({
                        'job_name': None, 'job_status': None, 'job_duration': None,
                        'job_stage': None, 'job_created_at': None, 'job_started_at': None,
                        'job_finished_at': None, 'job_actual_duration': None
                    })
                    records.append(record)
                else:
                    for job in jobs:
                        if any(excluded_job.lower() in job['name'].lower() for excluded_job in EXCLUDED_JOBS):
                            continue
                        record = base_record.copy()
                        record.update({
                            'job_name': job['name'],
                            'job_status': job['status'],
                            'job_duration': job.get('duration'),
                            'job_stage': job.get('stage'),
                            'job_created_at': job.get('created_at'),
                            'job_started_at': job.get('started_at'),
                            'job_finished_at': job.get('finished_at'),
                            'job_actual_duration': job.get('actual_duration')
                        })
                        records.append(record)
            count = self._progress_counter.increment()
            if count % 10 == 0:
                logger.info(f"Completed data collection for {count} repositories")
            return records
        except Exception as e:
            logger.error(f"Error collecting data for repository {repo_name} ({repo_id}): {e}")
            return []

    def collect_repositories_parallel(self, repo_tasks: List[Tuple], batch_size: Optional[int] = None) -> None:
        """
        Collect data from repositories in parallel batches.
        
        Args:
            repo_tasks: List of tuples for repository data collection
            batch_size: Size of each batch (uses instance default if None)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        total_repos = len(repo_tasks)
        logger.info(f"Starting parallel data collection from {total_repos} repositories in batches of {batch_size}")
        
        # Process repositories in batches
        for batch_start in range(0, total_repos, batch_size):
            batch_end = min(batch_start + batch_size, total_repos)
            batch = repo_tasks[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: repositories {batch_start+1}-{batch_end}")
            
            # Process current batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks in current batch
                future_to_repo = {
                    executor.submit(self.collect_repository_data, repo_data): repo_data[1]  # repo_name for tracking
                    for repo_data in batch
                }
                
                # Collect results as they complete
                batch_records = []
                for future in as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        result = future.result()
                        if result:
                            batch_records.extend(result)
                    except Exception as e:
                        logger.error(f"Repository data collection failed for {repo_name}: {e}")
                
                # Thread-safe addition to main data structure
                with self._data_lock:
                    self.pipeline_data.extend(batch_records)
                
            # Log batch completion and memory cleanup
            logger.info(f"Completed batch {batch_start//batch_size + 1}. Total records collected: {len(self.pipeline_data)}")
            
            # Force garbage collection between batches
            import gc
            gc.collect()

    def get_dataframe(self) -> pd.DataFrame:
        """Convert collected data to pandas DataFrame."""
        if not self.pipeline_data:
            logger.warning("No data collected yet")
            return pd.DataFrame()
            
        df = pd.DataFrame(self.pipeline_data)
        
        # Convert date columns to datetime
        date_columns = [
            'pipeline_created_at', 'pipeline_updated_at', 
            'job_created_at', 'job_started_at', 'job_finished_at'
        ]
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        logger.info(f"Created DataFrame with {len(df)} records and {len(df.columns)} columns")
        return df

    def collect_repository_tasks(self, domains: List[Dict], days_back: int, max_pipelines: int) -> List[Tuple]:
        """
        Collect all repository data collection tasks without executing them.
        Ensures that each subdomain is correctly associated with its parent domain and
        that domain/subdomain names are cleaned.
        
        Returns:
            List of tuples ready for parallel processing
        """
        repo_tasks = []

        def clean_name(name: Optional[str]) -> str:
            """Standardize names: strip whitespace and title-case. Returns empty string if None."""
            if not name:
                return ""
            return name.strip().title()

        for domain in domains:
            top_domain_name = clean_name(domain['name'])
            logger.info(f"Collecting repositories from domain: {top_domain_name}")

            top_domain_description = get_group_description(domain['id'])

            # Recursive subgroup fetching ensures we only get subgroups under this domain
            subgroups = get_subgroups(domain['id'], logger) or []
            if not subgroups:
                subgroups = [{'id': domain['id'], 'name': None}]
            for subdomain in subgroups:
                subdomain_name = clean_name(subdomain.get('name'))

                if not subdomain.get('name'):
                    logger.warning(f"Subdomain with ID {subdomain['id']} has no name")

                subdomain_description = get_group_description(subdomain['id'])

                repositories = get_repositories(subdomain['id'], logger)
                for repo in repositories:
                    repo_name = clean_name(repo['name'])

                    # Assign the correct parent domain
                    repo_tasks.append((
                        repo['id'],
                        repo_name,
                        subdomain_name,
                        top_domain_name,
                        top_domain_description,
                        subdomain_description,
                        days_back,
                        max_pipelines
                    ))

        logger.info(f"Collected {len(repo_tasks)} repositories for data collection")
        return repo_tasks

def collect_production_data(
    collector: GitLabDataFrameCollector,
    prod_group_id: Union[str, int],
    exclude_domains: List[str],
    days_back: int,
    max_pipelines: int
) -> pd.DataFrame:
    """Collect all production data using parallel processing."""
    logger.info(f"Running parallel production data collection for group ID: {prod_group_id}")
    
    domains = get_subgroups(prod_group_id, logger)
    filtered_domains = [d for d in domains if d['name'] not in exclude_domains]
    logger.info(f"Filtered out {len(domains) - len(filtered_domains)} excluded domains from production collection")
    
    repo_tasks = collector.collect_repository_tasks(filtered_domains, days_back, max_pipelines)
    collector.collect_repositories_parallel(repo_tasks)
    
    return collector.get_dataframe()

def collect_non_production_data(
    collector: GitLabDataFrameCollector,
    domain_group_id: str,
    exclude_domains: List[str],
    days_back: int,
    max_pipelines: int
) -> pd.DataFrame:
    """Collect all non-production data using parallel processing."""
    logger.info("Running parallel non-production data collection")
    
    domains = get_subgroups(domain_group_id, logger)
    filtered_domains = [d for d in domains if d['name'] not in exclude_domains]
    
    repo_tasks = collector.collect_repository_tasks(filtered_domains, days_back, max_pipelines)
    collector.collect_repositories_parallel(repo_tasks)
    
    return collector.get_dataframe()

def main(
    days_back: int = 30,
    exclude_domains: List[str] = None,
    max_pipelines_per_repo: int = 100,
    env: str = "prod",
    prod_group_id: Union[str, int] = "410115",
    max_workers: int = 16,
    batch_size: int = 80,
    save_csv: bool = True,
    csv_path: str = None,
    skip_dad: bool = True
) -> pd.DataFrame:
        """
        Main function to collect GitLab CI/CD data into a DataFrame and add projectID.
        
        Returns:
            pandas.DataFrame with comprehensive pipeline data
        """
        
        print(f"Starting GitLab CI/CD Data Collection with {max_workers} workers, batch size {batch_size}")
        start_time = time.time()

        if exclude_domains is None:
            exclude_domains = [
                'central-team-repository', 'Request-for-Change', 'External_Purchased_Data',
                'Reference_Data', 'Non_GO_Data', "Prod_Ops"
            ]
        
        try:
            os.makedirs("reports", exist_ok=True)
            
            collector = GitLabDataFrameCollector(max_workers=max_workers, batch_size=batch_size)
            
            logger.info(f"Starting parallel GitLab CI/CD data collection in {env} mode")
            
            # Collect data based on environment
            if env == "prod":
                df = collect_production_data(
                    collector=collector,
                    prod_group_id=prod_group_id,
                    exclude_domains=exclude_domains,
                    days_back=days_back,
                    max_pipelines=max_pipelines_per_repo
                )
            else:
                domain_group_id = "384233"
                df = collect_non_production_data(
                    collector=collector,
                    domain_group_id=domain_group_id,
                    exclude_domains=exclude_domains,
                    days_back=days_back,
                    max_pipelines=max_pipelines_per_repo
                )

            # Add projectID column
            if not df.empty:
                df = apply_projectID_mapping(df, mapping_path="subdomain_projectID_map.json")
                if not skip_dad:
                    df = add_dad_scores(df, config_path="config.json")


            # Save DataFrame to CSV if requested
            if save_csv and not df.empty:
                if csv_path is None:
                    csv_path = f"reports/pipeline_data_{env}_{days_back}days.csv"
                
                df.to_csv(csv_path, index=False)
                logger.info(f"DataFrame saved to {csv_path}")
                print(f"Data saved to: {csv_path}")
            
            # Display summary statistics
            if not df.empty:
                print(f"\nDataFrame Summary:")
                print(f"- Total records: {len(df):,}")
                print(f"- Date range: {df['pipeline_created_at'].min()} to {df['pipeline_created_at'].max()}")
                print(f"- Domains: {df['domain'].nunique()}")
                print(f"- Repositories: {df['repo_name'].nunique()}")
                print(f"- Unique pipelines: {df['pipeline_id'].nunique()}")
                print(f"- Branch flow violations: {df['branch_flow_violation'].sum():,} ({df['branch_flow_violation'].mean()*100:.1f}%)")
                
                print(f"\nPipeline Status Distribution:")
                status_counts = df['pipeline_status'].value_counts()
                for status, count in status_counts.head().items():
                    print(f"- {status}: {count:,} ({count/len(df)*100:.1f}%)")
            
            # Calculate and display runtime
            end_time = time.time()
            runtime_minutes = (end_time - start_time) / 60
            print(f"\nScript runtime: {runtime_minutes:.1f} minutes")
            logger.info(f"Script runtime: {runtime_minutes:.1f} minutes ({end_time - start_time:.1f} seconds)")
            
            return df
                
        except Exception as e:
            end_time = time.time()
            runtime_minutes = (end_time - start_time) / 60
            print(f"\nScript runtime before error: {runtime_minutes:.1f} minutes")
            logger.error(f"Error in main execution: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitLab CI/CD Data Collection")
    parser.add_argument("--env", type=str, default="prod", 
                        help="Environment to run (default: prod)", 
                        choices=["prod", "nprod"])
    parser.add_argument("--days_back", type=int, default=30,
                        help="Number of days to analyze (default: 30)")
    parser.add_argument("--max_pipelines_per_repo", type=int, default=100,
                        help="Max pipelines per repo (default: 100)")
    parser.add_argument("--max_workers", type=int, default=16, 
                          help="Number of parallel workers (default: 16)")
    parser.add_argument("--batch_size", type=int, default=80, 
                        help="Batch size for processing (default: 80)")
    parser.add_argument("--csv_path", type=str,
                         help="Custom path for CSV output file")
    parser.add_argument("--no_csv", action="store_true",
                        help="Skip saving CSV file")
    parser.add_argument("--skip_dad", action="store_true", 
                        help="Skip fetching DAD scores")
    
    args = parser.parse_args()

    df_result = main(
        env=args.env,
        days_back=args.days_back,
        max_pipelines_per_repo=args.max_pipelines_per_repo,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        save_csv=not args.no_csv,
        csv_path=args.csv_path, 
        skip_dad=args.skip_dad
    )
    
    print(f"\nData collection complete! DataFrame shape: {df_result.shape}")

