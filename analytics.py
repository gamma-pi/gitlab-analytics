import os
import sys
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
from collections import defaultdict
import textwrap
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
from io import BytesIO

import base64, mimetypes
from urllib.parse import urljoin
from pathlib import Path

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

logger = set_logg_handlers_to_console_and_file("logs/cicd_analytics.log")

# GitLab API details
gitlab_url = 'https://gitlab.com/'
pem_file = "utils/cert.pem"

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(__file__).parent / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

class ConfluenceClient:
    """
    Complete Confluence API client with all original methods plus enhanced file attachment handling.
    """
    
    def __init__(self, base_url: str, username: str, api_token: str):
        """
        Initialize the Confluence client with proper authentication.
        """
        # Standardize base URL
        base_url = base_url.rstrip('/')
        if not base_url.endswith('wiki'):
            base_url = f"{base_url}/wiki"
        base_url += '/'
        
        self.base_url = base_url
        self.api_token = api_token
        
        # Configure session
        self.session = requests.Session()
        
        # Use certificate if provided, otherwise disable verification
        if os.path.exists(pem_file):
            self.session.verify = pem_file
        else:
            logger.warning(f"Certificate file not found at {pem_file}, disabling SSL verification")
            self.session.verify = False
        
        # Updated headers with X-Atlassian-Token
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}',
            'X-Atlassian-Token': 'no-check'
        })
        
        # Rate limiting defaults
        self.rate_limit_remaining = 20
        self.rate_limit_reset = 60
        
        logger.debug(f"Confluence client initialized with base URL: {self.base_url}")

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None, files: Optional[Dict] = None) -> Optional[Dict]:
        """
        Enhanced request handler with proper authentication and error handling.
        """
        # Clean endpoint path
        endpoint = endpoint.lstrip('/')
        url = urljoin(self.base_url, endpoint)
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            request_kwargs = {
                'method': method,
                'url': url,
                'json': data,
                'params': params,
                'files': files,
                'timeout': 30,
                'verify': self.session.verify
            }
            
            # Remove None values
            request_kwargs = {k: v for k, v in request_kwargs.items() if v is not None}
            
            response = self.session.request(**request_kwargs)
            
            # Handle rate limiting
            self._update_rate_limits(response.headers)
            
            if response.status_code == 429:
                wait_time = self.rate_limit_reset + 1
                logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return self._make_request(method, endpoint, data, params, files)
                
            response.raise_for_status()
            return response.json() if response.content else None
            
        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
            return None

    def _update_rate_limits(self, headers: Dict):
        """Update rate limit tracking from headers"""
        if 'X-RateLimit-Remaining' in headers:
            self.rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            self.rate_limit_reset = int(headers['X-RateLimit-Reset'])

    def _handle_request_error(self, error: Exception):
        """Log detailed error information"""
        logger.error(f"Request failed: {error}")
        
        if hasattr(error, 'response') and error.response is not None:
            response = error.response
            try:
                error_details = response.json()
                logger.error(f"Error details: {error_details}")
            except ValueError:
                logger.error(f"Response content: {response.text}")
            
            logger.error(f"Status code: {response.status_code}")
            logger.error(f"Headers: {response.headers}")
            
            if response.status_code == 403:
                logger.error("""
                    Common 403 Fixes:
                    1. Verify your API token has correct permissions
                    2. Check if your Confluence instance requires special headers
                    3. Ensure your user account has access to the target page
                    4. Confirm your authentication method (Basic Auth is often disabled)
                """)
                
    def get_page(self, page_id: str) -> Optional[Dict]:
        """Get a page by ID."""
        return self._make_request('GET', f'rest/api/content/{page_id}?expand=body.storage,version')
        
    def update_page(self, page_id: str, title: str, content: str) -> Optional[Dict]:
        """Update an existing page."""
        page = self.get_page(page_id)
        if not page:
            return None
            
        new_version = page['version']['number'] + 1
        
        data = {
            'id': page_id,
            'type': 'page',
            'title': title,
            'body': {
                'storage': {
                    'value': content,
                    'representation': 'storage'
                }
            },
            'version': {
                'number': new_version,
                'message': f'Updated by CI/CD Analytics at {datetime.now().isoformat()}'
            }
        }
        
        return self._make_request('PUT', f'rest/api/content/{page_id}', data=data)
        
    def attach_file(self, page_id: str, file_path: str, comment: str = '', 
                   minor_edit: bool = True, allow_duplicates: bool = True) -> Optional[Dict]:
        """
        Enhanced file attachment with:
        - Better version control
        - Duplicate handling
        - Improved error handling
        
        Args:
            page_id: ID of the page to attach to
            file_path: Local path to file
            comment: Attachment comment
            minor_edit: Whether to mark as minor edit
            allow_duplicates: Whether to allow duplicate attachments
            
        Returns:
            Dictionary with attachment details or None if failed
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Attachment file not found: {file_path}")
                
            # Get file metadata
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            
            logger.info(f"Attaching {file_name} ({file_size} bytes) to page {page_id}")
            
            # Open file and prepare request
            with open(file_path, 'rb') as file_data:
                response = self._make_request(
                    'POST',
                    f'rest/api/content/{page_id}/child/attachment',
                    files={'file': (file_name, file_data, content_type)},
                    params={
                        'allowDuplicated': str(allow_duplicates).lower(),
                        'comment': comment,
                        'minorEdit': str(minor_edit).lower()
                    }
                )
                
            if not response:
                logger.error("No response received for attachment upload")
                return None
                
            # Handle different response formats
            if isinstance(response, list):
                return response[0] if response else None
            elif 'results' in response:
                return response['results'][0] if response['results'] else None
            return response
            
        except FileNotFoundError as e:
            logger.error(str(e))
            return None
        except Exception as e:
            logger.error(f"Unexpected error attaching file: {str(e)}")
            return None

    def upload_report_to_confluence(
        self,
        target_page_id: str,
        report_path: str,
        env: str,
        days_back: int
    ) -> bool:
        """
        Enhanced report upload with:
        - Better page version control
        - Improved error handling
        - More informative logging
        """
        try:
            if not os.path.exists(report_path):
                logger.error(f"Report file not found: {report_path}")
                return False
                
            logger.info(f"Starting Confluence upload for {report_path}")
            
            # Get existing page
            page = self.get_page(target_page_id)
            if not page:
                logger.error(f"Could not retrieve target page {target_page_id}")
                return False
                
            # Prepare new content
            env_title = "Production" if env == "prod" else "Non-Production"
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Create a new section with file preview
            new_section = f"""
            <h2>{env_title} CI/CD Analytics Report - {current_time}</h2>
            <p>Analysis period: Last {days_back} days</p>
            <p>Generated report attached below:</p>
            <p><ac:structured-macro ac:name="view-file" ac:schema-version="1">
                <ac:parameter ac:name="name">{os.path.basename(report_path)}</ac:parameter>
            </ac:structured-macro></p>
            <hr/>
            """
            
            # Update page content
            existing_content = page['body']['storage']['value']
            updated_content = new_section + existing_content
            
            # Update the page first
            update_result = self.update_page(
                target_page_id,
                page['title'],
                updated_content
            )
            
            if not update_result:
                logger.error("Failed to update page content")
                return False
                
            # Attach the file with retry logic
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    attachment_result = self.attach_file(
                        target_page_id,
                        report_path,
                        comment=f"{env_title} CI/CD Analytics Report generated on {current_time}",
                        minor_edit=True
                    )
                    
                    if attachment_result:
                        logger.info(f"Successfully uploaded report to Confluence page {target_page_id}")
                        return True
                        
                    logger.warning(f"Attachment attempt {attempt} failed")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        
                except Exception as e:
                    logger.error(f"Attachment attempt {attempt} failed with error: {str(e)}")
                    if attempt == max_retries:
                        raise
                    time.sleep(2 ** attempt)
            
            logger.error(f"Failed to attach file after {max_retries} attempts")
            return False
            
        except Exception as e:
            logger.error(f"Error during Confluence upload: {str(e)}", exc_info=True)
            return False

class ThreadSafeCounter:
    """Thread-safe counter for tracking progress."""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

class GitLabCICDAnalytics:
    """
    Comprehensive CI/CD analytics class with enhanced capabilities:
    - Pipeline success/failure analysis
    - Job duration and performance metrics
    - Branch protection analysis
    - Merge request approval metrics
    - Repository health scoring
    - Comparative analysis across domains/subdomains
    - Parallel processing for performance
    - Confluence integration for report publishing
    """
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        self.analytics_data.clear()
        self.pipeline_data.clear()

    def __init__(self, max_workers: int = 6, batch_size: int = 20):
        self.analytics_data = []
        self.pipeline_data = []
        self.job_data = []
        self.branch_data = []
        self.approval_data = []
        self.repository_health_scores = []
        
        # Parallelization settings
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._data_lock = threading.Lock()  # Protect shared data structures
        self._progress_counter = ThreadSafeCounter()
        
    def _make_gitlab_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Enhanced request handler with retry logic."""
        try:
            response = requests.get(
                url,
                headers=get_headers(),
                params=params,
                verify=pem_file,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_gitlab_request(url, params)
            else:
                logger.warning(f"API request failed: {url} - Status: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {e}")
            return None

    def get_project_pipelines(self, project_id: int, days_back: int = 30, max_pipelines: int = 10) -> List[Dict]:
        """Fetch pipelines with enhanced date filtering, pagination, and limits."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_pipelines = []
        page = 1
        per_page = min(max_pipelines, 100)  # Don't fetch more than needed per request
        
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
                
            # Add pipelines but don't exceed the limit
            remaining_slots = max_pipelines - len(all_pipelines)
            all_pipelines.extend(pipelines[:remaining_slots])
            
            # Break if we've reached our limit or if this was the last page
            if len(all_pipelines) >= max_pipelines or len(pipelines) < per_page:
                break
                
            page += 1
        
        return all_pipelines
    
    def get_pipeline_jobs(self, project_id: int, pipeline_id: int) -> List[Dict]:
        """Fetch jobs with additional performance metrics."""
        url = f"{gitlab_url}api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs?per_page=100"
        jobs = self._make_gitlab_request(url) or []
        
        # Add performance metrics
        for job in jobs:
            if job.get('started_at') and job.get('finished_at'):
                started = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
                finished = datetime.fromisoformat(job['finished_at'].replace('Z', '+00:00'))
                job['actual_duration'] = (finished - started).total_seconds()
            else:
                job['actual_duration'] = job.get('duration', 0)
                
        return jobs
    
    def get_pipeline_jobs(self, project_id: int, pipeline_id: int) -> List[Dict]:
        """Fetch jobs with additional performance metrics."""
        url = f"{gitlab_url}api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs?per_page=100"
        jobs = self._make_gitlab_request(url) or []
        
        # Add performance metrics
        for job in jobs:
            if job.get('started_at') and job.get('finished_at'):
                started = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
                finished = datetime.fromisoformat(job['finished_at'].replace('Z', '+00:00'))
                job['actual_duration'] = (finished - started).total_seconds()
            else:
                job['actual_duration'] = job.get('duration', 0)
                
        return jobs
    
    def analyze_branch_protection(self, project_id: int) -> Dict:
        """Analyze branch protection rules and compliance."""
        protected_branches = get_protected_branches(project_id, logger)
        all_branches = get_branches_in_repo(project_id, logger)
        
        protection_stats = {
            'main_protected': False,
            'dev_protected': False,
            'sit_protected': False,
            'protected_branch_count': len(protected_branches),
            'total_branch_count': len(all_branches),
            'protection_coverage': len(protected_branches) / len(all_branches) if all_branches else 0
        }
        
        for branch in protected_branches:
            name = branch['name']
            if name == 'main':
                protection_stats['main_protected'] = True
            elif name == 'dev':
                protection_stats['dev_protected'] = True
            elif name == 'sit':
                protection_stats['sit_protected'] = True
                
        return protection_stats
    
    def analyze_approval_rules(self, project_id: int) -> Dict:
        """Analyze merge request approval rules."""
        rules = get_approval_rules(project_id, logger)
        
        approval_stats = {
            'has_approval_rules': len(rules) > 0,
            'required_approvals': 0,
            'approval_rule_count': len(rules),
            'rules': []
        }
        
        if rules:
            approval_stats['required_approvals'] = max(r['approvals_required'] for r in rules)
            approval_stats['rules'] = [{
                'name': r['name'],
                'approvals_required': r['approvals_required'],
                'eligible_approvers': len(r['eligible_approvers'])
            } for r in rules]
            
        return approval_stats
    
    def calculate_repository_health(self, pipeline_stats: Dict, branch_stats: Dict, approval_stats: Dict) -> float:
        """Calculate a composite health score for the repository (0-100)."""
        weights = {
            'success_rate': 0.4,
            'main_protected': 0.2,
            'approval_rules': 0.2,
            'test_coverage': 0.1,
            'avg_duration': 0.1
        }
        
        # Normalize values
        success_score = pipeline_stats.get('success_rate', 0)
        protection_score = 100 if branch_stats.get('main_protected', False) else 0
        approval_score = 100 if approval_stats.get('has_approval_rules', False) else 0
        coverage_score = (pipeline_stats.get('test_coverage', 0) or 0) * 100
        duration_score = max(0, 100 - (pipeline_stats.get('avg_duration', 0) / 3600))  # Normalize hours to score
        
        # Calculate weighted score
        health_score = (
            weights['success_rate'] * success_score +
            weights['main_protected'] * protection_score +
            weights['approval_rules'] * approval_score +
            weights['test_coverage'] * coverage_score +
            weights['avg_duration'] * duration_score
        )
        
        return min(100, max(0, health_score))
    
    def analyze_repository_worker(self, repo_data: Tuple[int, str, str, str, int, int]) -> Optional[Dict]:
        """
        Worker function for parallel repository analysis.
        
        Args:
            repo_data: Tuple of (repo_id, repo_name, subdomain_name, domain_name, days_back, max_pipelines)
        
        Returns:
            Analytics dictionary or None if analysis failed
        """
        repo_id, repo_name, subdomain_name, domain_name, days_back, max_pipelines = repo_data
        
        try:
            # Pipeline analysis with limit
            pipelines = self.get_project_pipelines(repo_id, days_back, max_pipelines)
            pipeline_stats = self._analyze_pipelines(pipelines, repo_id)
            
            # Branch protection analysis
            branch_stats = self.analyze_branch_protection(repo_id)
            
            # Approval rules analysis
            approval_stats = self.analyze_approval_rules(repo_id)
            
            # Calculate repository health
            health_score = self.calculate_repository_health(pipeline_stats, branch_stats, approval_stats)
            
            analytics = {
                'repo_id': repo_id,
                'repo_name': repo_name,
                'subdomain': subdomain_name,
                'domain': domain_name,
                'health_score': health_score,
                'pipeline_stats': pipeline_stats,
                'branch_stats': branch_stats,
                'approval_stats': approval_stats,
                'last_updated': datetime.now().isoformat()
            }
            
            # Thread-safe progress tracking
            count = self._progress_counter.increment()
            if count % 10 == 0:  # Log every 10th completion
                logger.info(f"Completed analysis for {count} repositories")
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_name} (ID: {repo_id}): {e}")
            return None
    
    def analyze_repositories_parallel(self, repo_tasks: List[Tuple], batch_size: Optional[int] = None) -> None:
        """
        Analyze repositories in parallel batches to manage memory usage.
        
        Args:
            repo_tasks: List of tuples for repository analysis
            batch_size: Size of each batch (uses instance default if None)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        total_repos = len(repo_tasks)
        logger.info(f"Starting parallel analysis of {total_repos} repositories in batches of {batch_size}")
        
        # Process repositories in batches
        for batch_start in range(0, total_repos, batch_size):
            batch_end = min(batch_start + batch_size, total_repos)
            batch = repo_tasks[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: repositories {batch_start+1}-{batch_end}")
            
            # Process current batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks in current batch
                future_to_repo = {
                    executor.submit(self.analyze_repository_worker, repo_data): repo_data[1]  # repo_name for tracking
                    for repo_data in batch
                }
                
                # Collect results as they complete
                batch_results = []
                for future in as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Repository analysis failed for {repo_name}: {e}")
                
                # Thread-safe addition to main data structure
                with self._data_lock:
                    self.analytics_data.extend(batch_results)
                
            # Log batch completion and memory cleanup
            logger.info(f"Completed batch {batch_start//batch_size + 1}. Total repositories analyzed: {len(self.analytics_data)}")
            
            # Force garbage collection between batches to manage memory
            import gc
            gc.collect()
    
    def collect_repository_tasks(self, domains: List[Dict], days_back: int, max_pipelines: int) -> List[Tuple]:
        """
        Collect all repository analysis tasks without executing them.
        
        Returns:
            List of tuples ready for parallel processing
        """
        repo_tasks = []
        
        for domain in domains:
            logger.info(f"Collecting repositories from domain: {domain['name']}")
            
            subdomains = get_subgroups(domain['id'], logger)
            for subdomain in subdomains:
                repositories = get_repositories(subdomain['id'], logger)
                for repo in repositories:
                    repo_tasks.append((
                        repo['id'],
                        repo['name'],
                        subdomain['name'],
                        domain['name'],
                        days_back,
                        max_pipelines
                    ))
        
        logger.info(f"Collected {len(repo_tasks)} repositories for analysis")
        return repo_tasks
    

    
    def _analyze_pipelines(self, pipelines: List[Dict], project_id: int) -> Dict:
        """Analyze pipeline metrics and job performance."""
        if not pipelines:
            return {
                'total_pipelines': 0,
                'success_rate': 0,
                'avg_duration': 0,
                'status_breakdown': {},
                'test_coverage': None,
                'job_stats': {},
                'deployment_counts': {'success': 0, 'failed': 0}
            }
        
        # Add deployment counts tracking
        deployment_counts = {
            'success': len([p for p in pipelines if p['status'] == 'success']),
            'failed': len([p for p in pipelines if p['status'] in ('failed', 'canceled')])
        }
        
        # Basic metrics
        total_pipelines = len(pipelines)
        successful_pipelines = len([p for p in pipelines if p['status'] == 'success'])
        success_rate = (successful_pipelines / total_pipelines) * 100 if total_pipelines > 0 else 0
        
        # Duration analysis
        durations = [p['duration'] for p in pipelines if p.get('duration')]
        avg_duration = np.mean(durations) if durations else 0
        
        # Status breakdown
        status_breakdown = defaultdict(int)
        for pipeline in pipelines:
            status_breakdown[pipeline['status']] += 1
        
        # Job performance analysis
        job_stats = defaultdict(lambda: {
            'count': 0,
            'success_count': 0,
            'durations': [],
            'failure_rate': 0
        })
        
        # Sample recent pipelines for detailed job analysis (limit to 5 for performance)
        for pipeline in pipelines[:5]:
            jobs = self.get_pipeline_jobs(project_id, pipeline['id'])
            for job in jobs:
                job_name = job['name']
                job_stats[job_name]['count'] += 1
                if job['status'] == 'success':
                    job_stats[job_name]['success_count'] += 1
                if job.get('actual_duration'):
                    job_stats[job_name]['durations'].append(job['actual_duration'])
        
        # Calculate job-level metrics
        for job_name, stats in job_stats.items():
            if stats['count'] > 0:
                stats['success_rate'] = (stats['success_count'] / stats['count']) * 100
                stats['failure_rate'] = 100 - stats['success_rate']
                if stats['durations']:
                    stats['avg_duration'] = np.mean(stats['durations'])
                    stats['duration_stddev'] = np.std(stats['durations'])
                else:
                    stats['avg_duration'] = 0
                    stats['duration_stddev'] = 0
        
        return {
            'total_pipelines': total_pipelines,
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'status_breakdown': dict(status_breakdown),
            'job_stats': dict(job_stats),
            'test_coverage': self._get_test_coverage(project_id, pipelines),
            'deployment_counts': deployment_counts
        }
    
    def _get_test_coverage(self, project_id: int, pipelines: List[Dict]) -> Optional[float]:
        """Get test coverage from the most recent successful pipeline."""
        for pipeline in sorted(pipelines, key=lambda x: x.get('created_at', ''), reverse=True):
            if pipeline['status'] == 'success':
                test_report = self._make_gitlab_request(
                    f"{gitlab_url}api/v4/projects/{project_id}/pipelines/{pipeline['id']}/test_report"
                )
                if test_report and test_report.get('test_suites'):
                    return test_report['test_suites'][0].get('total_coverage')
        return None
    
    def _categorize_by_pattern(self, repo_name: str) -> Optional[str]:
        """Categorize repository by naming pattern."""
        repo_name = repo_name.lower()
        if 'raw' in repo_name or 'sdp_pattern_1' in repo_name:
            return 'Pattern 1'
        elif 'cdp' in repo_name or 'sdp_pattern_3' in repo_name:
            return 'Pattern 3'
        return None

    def generate_domain_comparison_report(self, env: str, save_path: str = None) -> None:
        """Generate comprehensive comparison report across domains."""
        if not self.analytics_data:
            logger.warning("No analytics data available for report generation")
            return
        
        # Set default filename based on environment if not provided
        if save_path is None:
            save_path = f"reports/domain_comparison_{env}.html"
        
        # Determine title based on environment
        env_title = "Production" if env=="prod" else "Non-Production"
        
        df = pd.DataFrame(self.analytics_data)
        
        # Extract nested values first
        df['success_rate'] = df['pipeline_stats'].apply(lambda x: x.get('success_rate', 0) if isinstance(x, dict) else 0)
        df['protection_coverage'] = df['branch_stats'].apply(lambda x: x.get('protection_coverage', 0) if isinstance(x, dict) else 0)
        df['has_approval_rules'] = df['approval_stats'].apply(lambda x: x.get('has_approval_rules', False) if isinstance(x, dict) else False)
        
        # Domain-level aggregates
        domain_stats = df.groupby('domain').agg({
            'health_score': ['mean', 'median', 'std'],
            'success_rate': 'mean',
            'protection_coverage': 'mean',
            'has_approval_rules': lambda x: (sum(x) / len(x)) * 100
        }).round(2)
        
        # Flatten multi-index columns
        domain_stats.columns = ['_'.join(col).strip() for col in domain_stats.columns.values]
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GitLab CI/CD {env_title} Domain Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; text-align: center; }}
                .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .good {{ color: #27ae60; font-weight: bold; }}
                .warning {{ color: #f39c12; font-weight: bold; }}
                .bad {{ color: #e74c3c; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GitLab CI/CD {env_title} Domain Comparison Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Domain Comparison</h2>
                <table>
                    <tr>
                        <th>Domain</th>
                        <th>Health Score</th>
                        <th>Success Rate</th>
                        <th>Branch Protection</th>
                        <th>Approval Rules</th>
                    </tr>
        """
        
        # Add domain rows with conditional formatting
        for domain, row in domain_stats.iterrows():
            health_class = "good" if row['health_score_mean'] > 75 else "warning" if row['health_score_mean'] > 50 else "bad"
            success_class = "good" if row['success_rate_mean'] > 80 else "warning" if row['success_rate_mean'] > 60 else "bad"
            protection_class = "good" if row['protection_coverage_mean'] > 0.8 else "warning" if row['protection_coverage_mean'] > 0.5 else "bad"
            approval_class = "good" if row['has_approval_rules_<lambda>'] > 80 else "warning" if row['has_approval_rules_<lambda>'] > 50 else "bad"
            
            html_content += f"""
                <tr>
                    <td>{domain}</td>
                    <td class="{health_class}">{row['health_score_mean']}</td>
                    <td class="{success_class}">{row['success_rate_mean']}%</td>
                    <td class="{protection_class}">{row['protection_coverage_mean']*100:.0f}%</td>
                    <td class="{approval_class}">{row['has_approval_rules_<lambda>']:.0f}%</td>
                </tr>
            """
        
        # Close HTML
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Domain comparison report saved to {save_path}")
    
    def generate_repository_health_dashboard(self, env: str, days_back: int, save_path: str = None) -> None:
        """Generate interactive repository health dashboard with pattern comparison."""
        if not self.analytics_data:
            logger.warning("No analytics data available for dashboard generation")
            return

        # Set default filename based on environment if not provided
        if save_path is None:
            save_path = f"reports/repository_health_{env}.html"

        # Determine environment
        env_title = "Production" if env == "prod" else "Non-Production"
        
        df = pd.DataFrame(self.analytics_data)
        
        # Prepare data for all charts
        # 1. Health Scores
        health_scores = df.groupby('domain')['health_score'].mean().round(2).to_dict()
        
        # 2. Success Rates
        success_rates = df.groupby('domain').apply(
            lambda x: x['pipeline_stats'].apply(
                lambda s: s.get('success_rate', 0) if isinstance(s, dict) else 0
            ).mean()
        ).round(2).to_dict()
        
        # 3. Protection Coverage
        protection_coverage = df.groupby('domain').apply(
            lambda x: x['branch_stats'].apply(
                lambda b: b.get('protection_coverage', 0) if isinstance(b, dict) else 0
            ).mean() * 100
        ).round(2).to_dict()
        
        # 4. Approval Adoption
        approval_adoption = df.groupby('domain').apply(
            lambda x: x['approval_stats'].apply(
                lambda a: a.get('has_approval_rules', False) if isinstance(a, dict) else False
            ).mean() * 100
        ).round(2).to_dict()
        
        # 5. Deployment Counts 
        deployment_data = defaultdict(lambda: {'success': 0, 'failed': 0})
        for entry in self.analytics_data:
            domain = entry['domain']
            counts = entry.get('pipeline_stats', {}).get('deployment_counts', {})
            deployment_data[domain]['success'] += counts.get('success', 0)
            deployment_data[domain]['failed'] += counts.get('failed', 0)
        
        # 6. Job Durations (Box Plot)
        box_data = []
        for entry in self.analytics_data:
            domain = entry['domain']
            pipeline_stats = entry.get('pipeline_stats', {})
            job_stats = pipeline_stats.get('job_stats', {})
            
            for job_name, stats in job_stats.items():
                if isinstance(stats, dict):
                    durations = stats.get('durations', [])
                    success_count = stats.get('success_count', 0)
                    total_count = stats.get('count', 0)
                    
                    for d in durations:
                        box_data.append({
                            'domain': domain,
                            'duration': d,
                            'status': 'success' if success_count > (total_count - success_count) else 'failed'
                        })

        # 7. Pattern Comparison Data
        pattern_data = []
        for entry in self.analytics_data:
            pattern = self._categorize_by_pattern(entry['repo_name'])
            if not pattern:
                continue
            
            counts = entry.get('pipeline_stats', {}).get('deployment_counts', {})
            pattern_data.append({
                'domain': entry['domain'],
                'pattern': pattern,
                'success': counts.get('success', 0),
                'failed': counts.get('failed', 0),
                'total': counts.get('success', 0) + counts.get('failed', 0)
            })

        # Generate HTML with all charts
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{env_title} Repository Health Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; text-align: center; border-radius: 5px; margin-bottom: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                .chart {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .full-width {{ grid-column: 1 / -1; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{env_title} Repository Health Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="dashboard">
                <div id="healthByDomain" class="chart"></div>
                <div id="successRateByDomain" class="chart"></div>
                <div id="protectionCoverageByDomain" class="chart"></div>
                <div id="approvalRulesByDomain" class="chart"></div>
                <div id="deploymentCountsByDomain" class="chart full-width"></div>
                <div id="patternComparison" class="chart full-width"></div>
                <div id="boxPlotDeployments" class="chart full-width"></div>
            </div>

            <script>
                // Data from Python
                var healthScores = {json.dumps(health_scores)};
                var successRates = {json.dumps(success_rates)};
                var protectionCoverage = {json.dumps(protection_coverage)};
                var approvalAdoption = {json.dumps(approval_adoption)};
                var deploymentData = {json.dumps(deployment_data)};
                var boxData = {json.dumps(box_data)};
                var patternData = {json.dumps(pattern_data)};

                // Helper function for bar charts
                function toChartData(dataDict, color = '#3498db') {{
                    return {{
                        x: Object.keys(dataDict),
                        y: Object.values(dataDict),
                        type: 'bar',
                        marker: {{ color: color }}
                    }};
                }}

                // Prepare deployment count data
                function prepareDeploymentData() {{
                    var domains = Object.keys(deploymentData);
                    var successData = [];
                    var failedData = [];
                    
                    domains.forEach(domain => {{
                        successData.push(deploymentData[domain].success || 0);
                        failedData.push(deploymentData[domain].failed || 0);
                    }});
                    
                    return [
                        {{
                            x: domains,
                            y: successData,
                            name: 'Successful',
                            type: 'bar',
                            marker: {{ color: '#27ae60' }}
                        }},
                        {{
                            x: domains,
                            y: failedData,
                            name: 'Failed',
                            type: 'bar',
                            marker: {{ color: '#e74c3c' }}
                        }}
                    ];
                }}

                // Prepare pattern comparison data
                function preparePatternData() {{
                    var pattern1Data = [];
                    var pattern3Data = [];
                    var domains = [...new Set(patternData.map(d => d.domain))];
                    
                    // Group data by domain and pattern
                    var groupedData = {{}};
                    domains.forEach(domain => {{
                        groupedData[domain] = {{
                            'Pattern 1': patternData.filter(d => d.domain === domain && d.pattern === 'Pattern 1').map(d => d.total),
                            'Pattern 3': patternData.filter(d => d.domain === domain && d.pattern === 'Pattern 3').map(d => d.total)
                        }};
                    }});
                    
                    // Create traces
                    var traces = [];
                    for (var domain in groupedData) {{
                        if (groupedData[domain]['Pattern 1'].length > 0) {{
                            traces.push({{
                                y: groupedData[domain]['Pattern 1'],
                                x: Array(groupedData[domain]['Pattern 1'].length).fill(domain),
                                name: 'Pattern 1',
                                type: 'box',
                                marker: {{ color: '#3498db' }},  // Blue for Pattern 1 
                                showlegend: traces.length === 0
                            }});
                        }}
                        
                        if (groupedData[domain]['Pattern 3'].length > 0) {{
                            traces.push({{
                                y: groupedData[domain]['Pattern 3'],
                                x: Array(groupedData[domain]['Pattern 3'].length).fill(domain),
                                name: 'Pattern 3',
                                type: 'box',
                                marker: {{ color: '#f39c12' }},  // Orange for Pattern 3
                                showlegend: traces.length === 1
                            }});
                        }}
                    }}
                    
                    return traces;
                }}

                // Prepare box plot data
                var successData = boxData.filter(d => d.status === "success");
                var failedData = boxData.filter(d => d.status === "failed");
                
                var traceSuccess = {{
                    y: successData.map(d => d.duration),
                    x: successData.map(d => d.domain),
                    name: 'Success',
                    type: 'box',
                    marker: {{ color: '#27ae60' }}
                }};

                var traceFailed = {{
                    y: failedData.map(d => d.duration),
                    x: failedData.map(d => d.domain),
                    name: 'Failed',
                    type: 'box',
                    marker: {{ color: '#e74c3c' }}
                }};

                // Create all plots
                Plotly.newPlot('healthByDomain', [toChartData(healthScores, '#2ecc71')], {{
                    title: 'Average Health Score by Domain',
                    yaxis: {{ title: 'Score (0-100)', range: [0, 100] }},
                    xaxis: {{ title: 'Domain' }}
                }});

                Plotly.newPlot('successRateByDomain', [toChartData(successRates, '#3498db')], {{
                    title: 'Pipeline Success Rate by Domain',
                    yaxis: {{ title: 'Success Rate (%)', range: [0, 100] }},
                    xaxis: {{ title: 'Domain' }}
                }});

                Plotly.newPlot('protectionCoverageByDomain', [toChartData(protectionCoverage, '#f39c12')], {{
                    title: 'Branch Protection Coverage by Domain',
                    yaxis: {{ title: 'Coverage (%)', range: [0, 100] }},
                    xaxis: {{ title: 'Domain' }}
                }});

                Plotly.newPlot('approvalRulesByDomain', [toChartData(approvalAdoption, '#9b59b6')], {{
                    title: 'Approval Rule Adoption by Domain',
                    yaxis: {{ title: 'Adoption Rate (%)', range: [0, 100] }},
                    xaxis: {{ title: 'Domain' }}
                }});

                Plotly.newPlot('deploymentCountsByDomain', prepareDeploymentData(), {{
                    title: 'Deployment Counts by Domain (Last {days_back} Days)',
                    yaxis: {{ title: 'Number of Deployments' }},
                    xaxis: {{ title: 'Domain' }},
                    barmode: 'stack'
                }});

                // Create pattern comparison plot if data exists
                if (patternData && patternData.length > 0) {{
                    Plotly.newPlot('patternComparison', preparePatternData(), {{
                        title: 'Deployment Count Comparison: Pattern 1 vs Pattern 3',
                        yaxis: {{ title: 'Number of Deployments' }},
                        xaxis: {{ title: 'Domain' }},
                        boxmode: 'group'
                    }});
                }} else {{
                    document.getElementById('patternComparison').innerHTML = 
                        '<p style="text-align: center; padding: 50px;">No pattern comparison data available</p>';
                }}

                // Only create box plot if we have data
                if (boxData.length > 0) {{
                    Plotly.newPlot('boxPlotDeployments', [traceSuccess, traceFailed], {{
                        title: 'Job Duration Distribution by Domain and Status',
                        yaxis: {{ title: 'Duration (seconds)' }},
                        xaxis: {{ title: 'Domain' }},
                        boxmode: 'group'
                    }});
                }} else {{
                    document.getElementById('boxPlotDeployments').innerHTML = 
                        '<p style="text-align: center; padding: 50px;">No duration data available for box plot</p>';
                }}
            </script>
        </body>
        </html>
        """
        
        # Save file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Repository health dashboard saved to {save_path}")

    def upload_report_to_confluence(
        self,
        confluence_client: ConfluenceClient,
        target_page_id: str,
        report_path: str,
        env: str,
        days_back: int
    ) -> bool:
        """
        Upload generated reports to Confluence with enhanced error handling.
        """
        try:
            if not os.path.exists(report_path):
                logger.error(f"Report file not found: {report_path}")
                return False
                
            logger.info(f"Starting Confluence upload for {report_path}")
            
            # Use the confluence client's upload method
            return confluence_client.upload_report_to_confluence(
                target_page_id=target_page_id,
                report_path=report_path,
                env=env,
                days_back=days_back
            )
        
        except Exception as e:
            logger.error(f"Error during Confluence upload: {str(e)}", exc_info=True)
            return False

    def generate_graph_images(self, env: str) -> Dict[str, str]:
        """Generate and save all graph images for reports in reports/plots directory.
        
        Args:
            env: Environment identifier ('prod' or 'nprod')
            
        Returns:
            Dictionary mapping graph names to their file paths
        """
        if not self.analytics_data:
            logger.warning("No analytics data available for graph generation")
            return {}

        plot_dir = os.path.join("reports", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        image_paths = {}
        df = pd.DataFrame(self.analytics_data)
        
        try:
            # Define common plot style
            plt.style.use('ggplot')
            plt.rcParams['figure.facecolor'] = 'white'
            
            # 1. Health Score by Domain
            plt.figure(figsize=(12, 8))
            health_by_domain = df.groupby('domain')['health_score'].mean().sort_values()
            health_path = os.path.join(plot_dir, f"health_score_{env}.png")
            health_by_domain.plot(kind='barh', color=plt.cm.Greens(np.linspace(0.4, 0.9, len(health_by_domain))))
            plt.title('Repository Health Score by Domain', pad=20)
            plt.xlabel('Health Score (0-100)')
            plt.xlim(0, 100)
            plt.grid(axis='x', alpha=0.3)
            plt.savefig(health_path, bbox_inches='tight', dpi=150)
            plt.close()
            image_paths['health_score'] = health_path

            # 2. Pipeline Success Rate
            plt.figure(figsize=(12, 8))
            df['success_rate'] = df['pipeline_stats'].apply(lambda x: x.get('success_rate', 0))
            success_path = os.path.join(plot_dir, f"success_rate_{env}.png")
            df.groupby('domain')['success_rate'].mean().sort_values().plot(kind='bar', color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(df['domain'].unique()))))
            plt.title('Pipeline Success Rate by Domain', pad=20)
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(success_path, bbox_inches='tight', dpi=150)
            plt.close()
            image_paths['success_rate'] = success_path

            # 3. Branch Protection Coverage
            plt.figure(figsize=(12, 8))
            df['protection'] = df['branch_stats'].apply(lambda x: x.get('protection_coverage', 0))
            protection_path = os.path.join(plot_dir, f"branch_protection_{env}.png")
            (df.groupby('domain')['protection'].mean()*100).sort_values().plot(kind='bar', color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df['domain'].unique()))))
            plt.title('Branch Protection Coverage by Domain', pad=20)
            plt.ylabel('Protected Branches (%)')
            plt.ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(protection_path, bbox_inches='tight', dpi=150)
            plt.close()
            image_paths['branch_protection'] = protection_path

            # 4. Approval Rules Adoption
            plt.figure(figsize=(12, 8))
            df['has_approvals'] = df['approval_stats'].apply(lambda x: x.get('has_approval_rules', False))
            approvals_path = os.path.join(plot_dir, f"approval_rules_{env}.png")
            (df.groupby('domain')['has_approvals'].mean()*100).sort_values().plot(kind='bar',color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df['domain'].unique()))))
            plt.title('Approval Rules Adoption by Domain', pad=20)
            plt.ylabel('Repositories with Approval Rules (%)')
            plt.ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(approvals_path, bbox_inches='tight', dpi=150)
            plt.close()
            image_paths['approval_rules'] = approvals_path

            # 5. Deployment Outcomes
            plt.figure(figsize=(12, 8))
            deployment_data = []
            for domain in df['domain'].unique():
                domain_df = df[df['domain'] == domain]
                success = sum(ps.get('deployment_counts', {}).get('success', 0) for ps in domain_df['pipeline_stats'])
                failed = sum(ps.get('deployment_counts', {}).get('failed', 0) for ps in domain_df['pipeline_stats'])
                deployment_data.append({'domain': domain, 'success': success, 'failed': failed})
            
            deployments_path = os.path.join(plot_dir, f"deployments_{env}.png")
            pd.DataFrame(deployment_data).set_index('domain').sort_values('success').plot(
                kind='barh', stacked=True, color=['#2ecc71', '#e74c3c'])
            plt.title('Deployment Outcomes by Domain', pad=20)
            plt.xlabel('Number of Deployments')
            plt.grid(axis='x', alpha=0.3)
            plt.legend(title='Status', loc='lower right')
            plt.savefig(deployments_path, bbox_inches='tight', dpi=150)
            plt.close()
            image_paths['deployments'] = deployments_path

            # 6. Job Duration Distribution
            job_data = []
            for repo in self.analytics_data:
                for job_name, stats in repo['pipeline_stats'].get('job_stats', {}).items():
                    if isinstance(stats, dict) and 'durations' in stats:
                        job_data.extend([(repo['domain'], job_name, d) for d in stats['durations']])
            
            if job_data:
                plt.figure(figsize=(14, 8))
                job_df = pd.DataFrame(job_data, columns=['domain', 'job', 'duration'])
                top_jobs = job_df.groupby('job')['duration'].median().nlargest(10).index
                durations_path = os.path.join(plot_dir, f"job_durations_{env}.png")
                sns.boxplot(data=job_df[job_df['job'].isin(top_jobs)], x='duration', y='job', hue='domain')
                plt.title('Top 10 Longest-Running Jobs by Duration', pad=20)
                plt.xlabel('Duration (seconds)')
                plt.ylabel('Job Name')
                plt.grid(axis='x', alpha=0.3)
                plt.savefig(durations_path, bbox_inches='tight', dpi=150)
                plt.close()
                image_paths['job_durations'] = durations_path

            # 7. Pipeline Status Distribution (Updated with custom colors)
            status_counts = defaultdict(int)
            for repo in self.analytics_data:
                for status, count in repo['pipeline_stats'].get('status_breakdown', {}).items():
                    status_counts[status] += count
            
            if status_counts:
                plt.figure(figsize=(10, 10))
                status_path = os.path.join(plot_dir, f"pipeline_status_{env}.png")
                
                # Define custom colors for different pipeline statuses
                status_color_map = {
                    'success': '#27ae60',      # Green
                    'failed': '#e74c3c',       # Red  
                    'skipped': '#95a5a6',      # Grey
                    'manual': '#f39c12',       # Orange
                    'canceled': "#063437",     # Grey (same as skipped)
                    'cancelled': "#0a494e",    # Grey (alternative spelling)
                    'running': '#3498db',      # Blue
                    'pending': '#f1c40f',      # Yellow
                    'created': '#9b59b6',      # Purple
                    'preparing': "#f9f9f9"     # Dark orange
                }
                
                # Get the statuses and their counts
                statuses = list(status_counts.keys())
                counts = list(status_counts.values())
                
                # Create color list based on status names
                colors = []
                for status in statuses:
                    # Use custom color if defined, otherwise use default
                    colors.append(status_color_map.get(status.lower(), '#34495e'))
                
                # Create the pie chart with custom colors
                plt.pie(counts, 
                        labels=statuses,
                        autopct='%1.1f%%',
                        colors=colors,
                        wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
                        startangle=90)
                
                plt.title('Overall Pipeline Status Distribution', pad=20, fontsize=14, fontweight='bold')
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
                # Add a legend with better positioning
                plt.legend(statuses, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                
                plt.savefig(status_path, bbox_inches='tight', dpi=150)
                plt.close()
                image_paths['pipeline_status'] = status_path

            logger.info(f"Generated {len(image_paths)} graphs in {plot_dir} for {env} environment")
            return image_paths

        except Exception as e:
            logger.error(f"Error generating graphs: {str(e)}", exc_info=True)
            # Clean up any partial files
            for path in image_paths.values():
                try:
                    os.remove(path)
                except:
                    pass
            return {}

    def generate_confluence_report_content(self, env: str, days_back: int, image_paths: Dict[str, str]) -> str:
        """Generate Confluence report content with images from reports/plots.
        
        Args:
            env: Environment identifier ('prod' or 'nprod')
            days_back: Number of days analyzed
            image_paths: Dictionary mapping graph names to file paths in reports/plots
            
        Returns:
            Confluence storage format HTML with embedded images
        """
        env_title = "Production" if env == "prod" else "Non-Production"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plot_dir = os.path.join("reports", "plots")
        
        # Default image paths in reports/plots
        default_images = {
            'health_score': os.path.join(plot_dir, f"health_score_{env}.png"),
            'success_rate': os.path.join(plot_dir, f"success_rate_{env}.png"),
            'branch_protection': os.path.join(plot_dir, f"branch_protection_{env}.png"),
            'approval_rules': os.path.join(plot_dir, f"approval_rules_{env}.png"),
            'deployments': os.path.join(plot_dir, f"deployments_{env}.png"),
            'job_durations': os.path.join(plot_dir, f"job_durations_{env}.png"),
            'pipeline_status': os.path.join(plot_dir, f"pipeline_status_{env}.png")
        }
        
        # Use provided paths or fall back to defaults
        effective_paths = {**default_images, **image_paths}
        
        # Verify images exist
        for name, path in effective_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Missing graph image: {path}")
                effective_paths[name] = None
        
        # Generate the report sections
        sections = [
            f"""<h2>{env_title} CI/CD Analytics Report</h2>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Analysis Period:</strong> Last {days_back} days</p>
            <hr/>"""
        ]
        
        # Add each graph section
        report_sections = [
            ('health_score', 'Repository Health Overview', 
            'Composite score evaluating pipeline success, branch protection, approval rules, and performance.'),
            ('success_rate', 'Pipeline Success Rates', 
            'Percentage of pipelines that completed successfully across domains.'),
            ('branch_protection', 'Branch Protection Status', 
            'Percentage of branches with protection rules enabled (main branch should be 100%).'),
            ('approval_rules', 'Approval Rules Adoption', 
            'Percentage of repositories requiring approvals for merge requests.'),
            ('deployments', 'Deployment Outcomes', 
            'Total deployment attempts and their success/failure status.'),
            ('job_durations', 'Job Performance', 
            'Duration distribution of the top 10 longest-running jobs across domains.'),
            ('pipeline_status', 'Pipeline Status Distribution', 
            'Overall percentage of pipelines by status (success, failed, canceled, etc.).')
        ]
        
        for graph_id, title, description in report_sections:
            sections.append(f"""
            <h3>{title}</h3>
            {self._generate_image_macro(effective_paths[graph_id], title)}
            <p>{description}</p>
            """)
        
        sections.append("""<hr/><p><em>Report generated automatically by CI/CD Analytics</em></p>""")
        
        return "\n".join(sections)

    def _generate_image_macro(self, image_path: Optional[str], alt_text: str) -> str:
        """Generate Confluence image macro for a given image file."""
        if image_path is None or not os.path.exists(image_path):
            return f'<p style="color: red;">[Missing graph: {alt_text}]</p>'
        
        filename = os.path.basename(image_path)
        return f"""
        <ac:structured-macro ac:name="image">
        <ac:parameter ac:name="alt">{alt_text}</ac:parameter>
        <ac:parameter ac:name="width">800</ac:parameter>
        <ac:rich-text-body>
            <ac:image ac:alt="{alt_text}">
            <ri:attachment ri:filename="{filename}"/>
            </ac:image>
        </ac:rich-text-body>
        </ac:structured-macro>
        """

    def upload_report_to_confluence(
                                        self,
                                        confluence_client: ConfluenceClient,
                                        target_page_id: str,
                                        env: str,
                                        days_back: int
                                    ) -> bool:
        """Upload generated graphs and report to Confluence."""
        try:
            # Generate all graphs
            image_paths = self.generate_graph_images(env)
            if not image_paths:
                logger.error("No graphs generated for Confluence upload")
                return False
                
            # Generate Confluence content
            report_content = self.generate_confluence_report_content(env, days_back, image_paths)
            
            # Update page with new content
            page = confluence_client.get_page(target_page_id)
            if not page:
                logger.error(f"Could not retrieve target page {target_page_id}")
                return False
                
            update_result = confluence_client.update_page(
                page_id=target_page_id,
                title=page['title'],
                content=report_content,
                version=page['version']['number']
            )
            
            if not update_result:
                logger.error("Failed to update page content")
                return False
                
            # Upload all images as attachments
            for img_path in image_paths.values():
                if not os.path.exists(img_path):
                    logger.warning(f"Graph image not found: {img_path}")
                    continue
                    
                attachment_result = confluence_client.attach_file(
                    target_page_id,
                    img_path,
                    comment=f"{env_title} CI/CD Analytics graph",
                    minor_edit=True
                )
                
                if not attachment_result:
                    logger.warning(f"Failed to attach image: {img_path}")
                    
            logger.info("Successfully updated Confluence page with graphs")
            return True
            
        except Exception as e:
            logger.error(f"Error during Confluence upload: {str(e)}", exc_info=True)
            return False

def _analyze_production_parallel(   
                                    analytics: GitLabCICDAnalytics,
                                    prod_group_id: Union[str, int],
                                    days_back: int,
                                    max_pipelines: int
                                ) -> GitLabCICDAnalytics:
    """Analyze all production repositories using parallel processing."""
    logger.info(f"Running parallel production analysis for group ID: {prod_group_id}")
    
    domains = get_subgroups(prod_group_id, logger)  # 1st level groups
    repo_tasks = analytics.collect_repository_tasks(domains, days_back, max_pipelines)
    
    # Process all repositories in parallel batches
    analytics.analyze_repositories_parallel(repo_tasks)
    
    _generate_reports(analytics, "prod", days_back)
    return analytics

def _analyze_non_production_parallel(
                                        analytics: GitLabCICDAnalytics,
                                        domain_group_id: str,
                                        exclude_domains: List[str],
                                        days_back: int,
                                        max_pipelines: int
                                    ) -> GitLabCICDAnalytics:
    """Analyze all non-production repositories using parallel processing."""
    logger.info("Running parallel non-production analysis")
    
    domains = get_subgroups(domain_group_id, logger)
    # Filter out excluded domains
    filtered_domains = [d for d in domains if d['name'] not in exclude_domains]
    
    repo_tasks = analytics.collect_repository_tasks(filtered_domains, days_back, max_pipelines)
    
    # Process all repositories in parallel batches
    analytics.analyze_repositories_parallel(repo_tasks)
    
    _generate_reports(analytics, "nprod", days_back)
    return analytics

def _generate_reports(analytics: GitLabCICDAnalytics, env: str, days_back: int) -> Tuple[str, str]:
    """Generate reports and return paths to both report files."""
    if not analytics.analytics_data:
        logger.warning("No repository data collected for analysis")
        return None, None
    
    # Generate both reports
    os.makedirs("reports", exist_ok=True)
    
    # Generate domain comparison report
    domain_report_path = f"reports/domain_comparison_{env}.html"
    analytics.generate_domain_comparison_report(env, domain_report_path)
    
    # Generate repository health dashboard
    health_report_path = f"reports/repository_health_{env}.html"
    analytics.generate_repository_health_dashboard(env, days_back, health_report_path)
    
    logger.info(f"Reports generated at:\n- {domain_report_path}\n- {health_report_path}")
    return domain_report_path, health_report_path

def main(
    days_back: int = 30,
    exclude_domains: List[str] = None,
    max_pipelines_per_repo: int = 10,
    env: str = "prod",
    prod_group_id: Union[str, int] = "410115",
    max_workers: int = 8,
    batch_size: int = 40,
    upload_to_confluence: bool = False,                 # ← Controls Confluence upload
    upload_only: bool = False                           # ← Upload existing reports only
) -> GitLabCICDAnalytics:
    # Initialize default excluded domains
    if exclude_domains is None:
        exclude_domains = ['central-team-repository', 'Request-for-Change']
    
    try:
        # Create reports directory structure
        os.makedirs(os.path.join("reports", "plots"), exist_ok=True)
        
        analytics = GitLabCICDAnalytics(max_workers=max_workers, batch_size=batch_size)
        
        if not upload_only:
            logger.info(f"Starting parallel GitLab CI/CD Analytics in {env} mode")
            
            domain_group_id = "384233"
            
            if env == "prod":
                analytics_result = _analyze_production_parallel(
                    analytics=analytics,
                    prod_group_id=prod_group_id,
                    days_back=days_back,
                    max_pipelines=max_pipelines_per_repo
                )
            else:
                analytics_result = _analyze_non_production_parallel(
                    analytics=analytics,
                    domain_group_id=domain_group_id,
                    exclude_domains=exclude_domains,
                    days_back=days_back,
                    max_pipelines=max_pipelines_per_repo
                )
            
            # Generate all graphs first
            logger.info("Generating visualization graphs...")
            image_paths = analytics.generate_graph_images(env)
            if not image_paths:
                logger.error("Failed to generate any graphs!")
                return analytics
            
            # Then generate reports that reference these graphs
            logger.info("Generating reports...")
            domain_report_path, health_report_path = _generate_reports(analytics, env, days_back)
            
        # Handle Confluence upload if requested
        if upload_to_confluence or upload_only:
            try:
                config = load_config()
                if not config.get('confluence'):
                    raise ValueError("Missing 'confluence' section in config.json")
                
                required_keys = ['page_id', 'base_url', 'auth']
                if not all(k in config['confluence'] for k in required_keys):
                    raise ValueError(f"Config missing required keys: {required_keys}")
                
                # Initialize Confluence client with proper auth
                auth_config = config['confluence']['auth']
                if auth_config['type'] == 'pat':
                    confluence_client = ConfluenceClient(
                        base_url=config['confluence']['base_url'],
                        username='api',  # Dummy value, not used with PAT
                        api_token=auth_config['token']
                    )
                else:
                    raise ValueError("Only PAT authentication is currently supported")
                
                # Upload reports and images
                if image_paths:
                    logger.info("Uploading graphs to Confluence...")
                    for img_path in image_paths.values():
                        if os.path.exists(img_path):
                            confluence_client.attach_file(
                                config['confluence']['page_id'],
                                img_path
                            )
                
                if domain_report_path and os.path.exists(domain_report_path):
                    confluence_client.upload_report_to_confluence(
                        target_page_id=config['confluence']['page_id'],
                        report_path=domain_report_path,
                        env=env,
                        days_back=days_back
                    )
                
                logger.info("Confluence upload completed successfully")
                
            except Exception as e:
                logger.error(f"Confluence upload failed: {str(e)}")
                logger.info("Reports were still generated locally")
        
        return analytics
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="prod", 
                      help="Environment to run (default: prod)", 
                      choices=["prod", "nprod"])
    parser.add_argument("--days_back", type=int, default=30,
                      help="Number of days to analyze (default: 30)")
    parser.add_argument("--max_pipelines_per_repo", type=int, default=10,
                      help="Max pipelines per repo (default: 10)")
    parser.add_argument("--max_workers", type=int, default=8, 
                      help="Number of parallel workers (default: 8)")
    parser.add_argument("--batch_size", type=int, default=40, 
                      help="Batch size for processing (default: 40)")
    parser.add_argument("--upload_only", action="store_true",
                      help="Upload existing reports without regeneration")
    parser.add_argument("--upload_confluence", action="store_true",
                          help="Upload reports to Confluence (disabled by default)")
    # parser.add_argument("--no_confluence", action="store_true",
    #                   help="Skip Confluence upload")
    parser.add_argument("--publish", action="store_true",
                      help="Run confluence.py after analytics completes")
    
    args = parser.parse_args()

    # Default behavior - production with upload unless explicitly disabled
    upload_to_confluence = args.upload_confluence  # Only upload if flag is provided
    
    # # If non-prod environment requested, require explicit upload flag
    # if args.env == "nprod":
    #     upload_to_confluence = False

    analytics_result = main(
        env=args.env,
        days_back=args.days_back,
        max_pipelines_per_repo=args.max_pipelines_per_repo,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        upload_to_confluence=upload_to_confluence,
        upload_only=args.upload_only
    )

    # Trigger confluence.py if --publish is set
    if args.publish:
        script_path = os.path.join(os.path.dirname(__file__), "confluence.py")
        logger.info(f"Publishing to Confluence by running: {script_path}")
        subprocess.run([sys.executable, script_path, "--env", args.env], check=True)
