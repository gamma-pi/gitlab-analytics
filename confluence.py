import os
import sys
import json
import requests
import base64
from datetime import datetime
from typing import Dict, Optional, List
import argparse
from pathlib import Path
import xml.sax.saxutils as saxutils

sys.path.append(os.getcwd())

from utils.logger import set_logg_handlers_to_console_and_file, log_function_data

logger = set_logg_handlers_to_console_and_file("logs/confluence_publisher.log")

class ConfluencePublisher:
    """
    Confluence API client for publishing GitLab CI/CD analytics reports.
    Supports both basic auth and API token authentication.
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.base_url = self.config['confluence']['base_url'].rstrip('/')
        self.space_key = self.config['confluence']['space_key']
        self.auth_headers = self._setup_auth()
        
        # Get absolute path to certificate
        cert_path = os.path.join(os.path.dirname(__file__), 'utils', 'cert.pem')
        logger.info(f"Using certificate bundle at: {cert_path}")
        
        # Verify certificate file exists
        if not os.path.exists(cert_path):
            logger.error(f"Certificate file not found at: {cert_path}")
            raise FileNotFoundError(f"Certificate file not found at: {cert_path}")
        
        self.session = requests.Session()
        self.session.headers.update(self.auth_headers)
        
        #self.session.verify = cert_path
        # TEMPORARY: Disable SSL verification (for testing only)
        self.session.verify = False
        
        # Add warning
        logger.warning("SSL verification is disabled - this is not recommended for production!")

        self._verify_connection()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _setup_auth(self) -> Dict[str, str]:
        """Setup authentication headers based on configuration."""
        auth_config = self.config['confluence']['auth']
        
        if auth_config['type'] == 'pat':
            # Personal Access Token authentication (recommended for Atlassian Cloud)
            return {
                'Authorization': f'Bearer {auth_config["token"]}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        elif auth_config['type'] == 'token':
            # API Token authentication (legacy, requires email)
            auth_string = f"{auth_config['email']}:{auth_config['token']}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            return {
                'Authorization': f'Basic {encoded_auth}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        elif auth_config['type'] == 'basic':
            # Basic authentication
            auth_string = f"{auth_config['username']}:{auth_config['password']}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            return {
                'Authorization': f'Basic {encoded_auth}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        else:
            raise ValueError(f"Unsupported authentication type: {auth_config['type']}")
    
    def upload_attachment(self, page_id: str, file_path: str) -> Optional[str]:
        """Handles both new uploads and updates of existing attachments"""
        if not os.path.exists(file_path):
            logger.error(f"Attachment not found: {file_path}")
            return None

        filename = os.path.basename(file_path)
        base_url = f"{self.base_url}/rest/api/content/{page_id}/child/attachment"

        # Required headers
        headers = {
            "X-Atlassian-Token": "no-check",
            "Authorization": self.auth_headers["Authorization"]
        }

        # Determine content type
        content_type = "image/png" if filename.lower().endswith(".png") else "application/octet-stream"

        try:
            # First check if attachment exists
            check_url = f"{base_url}?filename={filename}"
            check_response = requests.get(check_url, headers=headers, verify=False)
            
            with open(file_path, "rb") as file:
                files = {"file": (filename, file, content_type)}
                params = {"allowDuplicates": "false"}

                if check_response.status_code == 200 and check_response.json().get("results"):
                    # Update existing attachment
                    attachment_id = check_response.json()["results"][0]["id"]
                    update_url = f"{base_url}/{attachment_id}/data"
                    response = requests.post(
                        update_url,
                        headers=headers,
                        files=files,
                        params=params,
                        verify=False
                    )
                    action = "updated"
                else:
                    # New attachment
                    response = requests.post(
                        base_url,
                        headers=headers,
                        files=files,
                        params=params,
                        verify=False
                    )
                    action = "uploaded"

            if response.status_code in (200, 201):
                logger.info(f"Successfully {action} attachment: {filename}")
                return filename
            else:
                logger.error(f"Failed to {action} attachment (HTTP {response.status_code}): {response.text}")
                return None

        except Exception as e:
            logger.error(f"Attachment {action} error: {str(e)}")
            return None

    def _verify_connection(self) -> None:
        """Verify connection to Confluence API."""
        logger.debug(f"Using certificate bundle at: {self.session.verify}")

        try:
            response = self.session.get(f"{self.base_url}/rest/api/space/{self.space_key}")
            if response.status_code == 200:
                space_info = response.json()
                logger.info(f"Connected to Confluence space: {space_info['name']}")
            else:
                logger.error(f"Failed to connect to Confluence: {response.status_code} - {response.text}")
                raise ConnectionError("Failed to verify Confluence connection")
        except requests.exceptions.SSLError as ssl_error:
            logger.error("SSL verification failed. Possible solutions:")
            logger.error("1. Install your corporate root certificate")
            logger.error("2. Set REQUESTS_CA_BUNDLE environment variable")
            logger.error(f"Error details: {ssl_error}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error connecting to Confluence: {e}")
            raise
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling and logging."""
        try:
            response = self.session.request(method, url, **kwargs)
            logger.debug(f"{method} {url} - Status: {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url} - Error: {e}")
            raise
    
    def get_page_by_title(self, title: str) -> Optional[Dict]:
        """
        Get page by title in the configured space.
        
        Args:
            title: Page title to search for
            
        Returns:
            Page information if found, None otherwise
        """
        url = f"{self.base_url}/rest/api/content"
        params = {
            'title': title,
            'spaceKey': self.space_key,
            'expand': 'version,body.storage'
        }
        
        response = self._make_request('GET', url, params=params)
        
        if response.status_code == 200:
            results = response.json()
            if results['results']:
                page = results['results'][0]
                logger.info(f"Found existing page: {title} (ID: {page['id']})")
                return page
        
        logger.info(f"Page not found: {title}")
        return None
    
    def create_page(self, title: str, content: str, parent_id: Optional[str] = None) -> Dict:
        url = f"{self.base_url}/rest/api/content"

        # Escape XML special characters here:
        safe_content = saxutils.escape(content)

        page_data = {
            'type': 'page',
            'title': title,
            'space': {'key': self.space_key},
            'body': {
                'storage': {
                    'value': safe_content,
                    'representation': 'storage'
                }
            }
        }

        if parent_id:
            page_data['ancestors'] = [{'id': parent_id}]

        response = self._make_request('POST', url, json=page_data)

        if response.status_code == 200:
            page = response.json()
            logger.info(f"Created page: {title} (ID: {page['id']})")
            return page
        else:
            logger.error(f"Failed to create page: {response.status_code} - {response.text}")
            response.raise_for_status()

    def update_page(self, page_id: str, title: str, content: str, version: int) -> Dict:
        url = f"{self.base_url}/rest/api/content/{page_id}"

        page_data = {
            'version': {'number': version + 1},
            'title': title,
            'type': 'page',
            'body': {
                'storage': {
                    'value': content,  # REMOVED saxutils.escape()
                    'representation': 'storage'
                }
            }
        }

        response = self._make_request('PUT', url, json=page_data)
        if response.status_code == 200:
            return response.json()
        response.raise_for_status()
    
    def publish_or_update_page(self, title: str, content: str, parent_id: Optional[str] = None) -> Dict:
        """
        Publish content to Confluence. Creates new page or updates existing one.
        
        Args:
            title: Page title
            content: Page content in Confluence storage format
            parent_id: Optional parent page ID for new pages
            
        Returns:
            Page information (created or updated)
        """
        existing_page = self.get_page_by_title(title)
        
        if existing_page:
            # Update existing page
            return self.update_page(
                page_id=existing_page['id'],
                title=title,
                content=content,
                version=existing_page['version']['number']
            )
        else:
            # Create new page
            return self.create_page(title, content, parent_id)
    
    def html_to_confluence_storage(self, html_content: str) -> str:
        """
        Converts domain comparison HTML to proper Confluence Storage Format
        without escaping valid XHTML tags
        """
        from bs4 import BeautifulSoup

        CONFLUENCE_TEMPLATE = """<ac:structured-macro ac:name="panel">
    <ac:parameter ac:name="title">Domain Comparison Report</ac:parameter>
    <ac:rich-text-body>
    {content}
    </ac:rich-text-body>
    </ac:structured-macro>"""

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Add Confluence-compatible styling
            style_macro = """<ac:structured-macro ac:name="style">
    <ac:plain-text-body><![CDATA[
    .report-table { border-collapse: collapse; width: 100%; }
    .report-table th { background-color: #f5f5f5; font-weight: bold; }
    .report-table, .report-table th, .report-table td { 
        border: 1px solid #ddd; padding: 8px; text-align: left;
    }
    .good { background-color: #DFF2BF; color: #4F8A10 !important; }
    .bad { background-color: #FFBABA; color: #D8000C !important; }
    .warning { background-color: #FEEFB3; color: #9F6000 !important; }
    ]]></ac:plain-text-body>
    </ac:structured-macro>"""

            # Process tables
            for table in soup.find_all('table'):
                table['class'] = 'report-table'
                table['data-layout'] = 'default'

            # Extract main content
            main_content = ""
            if soup.body:
                main_content = str(soup.body)
            else:
                main_content = str(soup)

            # Combine elements
            full_content = f"""<p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    <hr/>
    {style_macro}
    {main_content}"""

            return CONFLUENCE_TEMPLATE.format(content=full_content)

        except Exception as e:
            logger.error(f"HTML conversion error: {str(e)}")
            return f"<p>Error processing report: {str(e)}</p>"
        
    def convert_repository_dashboard_to_static(self, env: str, plots_dir: Path) -> str:
        """
        Convert repository dashboard to static Confluence content with embedded PNG images.
        This replaces the interactive JavaScript charts with static images.
        
        Args:
            env: Environment identifier (prod/nprod)
            plots_dir: Path to the plots directory containing PNG files
            
        Returns:
            Confluence storage format content with embedded images
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            env_title = "Production" if env == "prod" else "Non-Production"
            
            # Define the chart mapping - these should match the filenames generated by your analytics
            chart_definitions = [
                {
                    'filename': f'health_score_{env}.png',
                    'title': 'Repository Health Overview',
                    'description': 'Composite score evaluating pipeline success, branch protection, approval rules, and performance.'
                },
                {
                    'filename': f'success_rate_{env}.png', 
                    'title': 'Pipeline Success Rates',
                    'description': 'Percentage of pipelines that completed successfully across domains.'
                },
                {
                    'filename': f'branch_protection_{env}.png',
                    'title': 'Branch Protection Status', 
                    'description': 'Percentage of branches with protection rules enabled (main branch should be 100%).'
                },
                {
                    'filename': f'approval_rules_{env}.png',
                    'title': 'Approval Rules Adoption',
                    'description': 'Percentage of repositories requiring approvals for merge requests.'
                },
                {
                    'filename': f'deployments_{env}.png',
                    'title': 'Deployment Outcomes',
                    'description': 'Total deployment attempts and their success/failure status.'
                },
                {
                    'filename': f'job_durations_{env}.png',
                    'title': 'Job Performance Analysis',
                    'description': 'Duration distribution of the top 10 longest-running jobs across domains.'
                },
                {
                    'filename': f'pipeline_status_{env}.png',
                    'title': 'Pipeline Status Distribution',
                    'description': 'Overall percentage of pipelines by status (success, failed, canceled, etc.).'
                }
            ]
            
            # Start building the Confluence content
            content_parts = [
                f'<h1>{env_title} Repository Health Dashboard</h1>',
                f'<p><em>Report generated on {timestamp}</em></p>',
                '<hr/>'
            ]
            
            # Add each chart section
            for chart in chart_definitions:
                image_path = plots_dir / chart['filename']
                
                # Check if the image file exists
                if image_path.exists():
                    # Create a section for this chart
                    content_parts.extend([
                        f'<h2>{chart["title"]}</h2>',
                        self._create_image_macro(chart['filename'], chart['title']),
                        f'<p>{chart["description"]}</p>',
                        '<br/>'
                    ])
                    logger.info(f"Added chart section: {chart['title']}")
                else:
                    # Add placeholder if image is missing
                    content_parts.extend([
                        f'<h2>{chart["title"]}</h2>',
                        f'<p style="color: red;"><strong>Missing chart:</strong> {chart["filename"]} not found in plots directory</p>',
                        f'<p>{chart["description"]}</p>',
                        '<br/>'
                    ])
                    logger.warning(f"Missing chart image: {chart['filename']}")
            
            # Add footer
            content_parts.extend([
                '<hr/>',
                '<p><em>This dashboard shows static visualizations of your CI/CD analytics. Charts are automatically generated from GitLab pipeline data.</em></p>'
            ])
            
            return '\n'.join(content_parts)
            
        except Exception as e:
            logger.error(f"Error converting dashboard to static format: {str(e)}")
            return f'<p>Error generating dashboard: {str(e)}</p>'
    
    def _create_image_macro(self, filename: str, alt_text: str) -> str:
        """
        Create a Confluence image macro that references an attached file.
        
        Args:
            filename: Name of the attached image file
            alt_text: Alt text for the image
            
        Returns:
            Confluence storage format image macro
        """
        return f"""<ac:image ac:align="center" ac:alt="{alt_text}" ac:width="600">
    <ri:attachment ri:filename="{filename}"/>
</ac:image>"""

    def publish_analytics_reports(self, reports_dir: str = "reports", env: str = "prod") -> List[Dict]:
        reports_path = Path(reports_dir)
        plots_path = reports_path / "plots"
        if not reports_path.exists():
            logger.error(f"Reports directory not found: {reports_dir}")
            return []

        published_pages = []
        page_ids = self.config['confluence']['reports']['publish_page_ids'].get(env, {})

        report_files = {
            'domain_comparison': {
                'filename': f'domain_comparison_{env}.html',
                'title': f"{env.capitalize()} Domain Comparison Report",
                'converter': self.html_to_confluence_storage,
                'upload_images': False  # Domain comparison doesn't use images
            },
            'repository_health': {
                'filename': f'repository_health_{env}.html', 
                'title': f"{env.capitalize()} Repository Health Dashboard",
                'converter': lambda x: self.convert_repository_dashboard_to_static(env, plots_path),
                'upload_images': True  # Repository health uses PNG charts
            }
        }

        for report_key, report_config in report_files.items():
            page_id = page_ids.get(report_key)
            if not page_id:
                logger.error(f"No page ID configured for {report_key}")
                continue

            try:
                # Get the existing page
                existing_page = self._get_page_by_id(page_id)
                if not existing_page:
                    logger.error(f"Target page not found: ID {page_id}")
                    continue

                # Handle repository health dashboard (static images)
                if report_key == "repository_health":
                    # First upload all the PNG images as attachments
                    if plots_path.exists() and report_config['upload_images']:
                        logger.info("Uploading PNG chart images from reports/plots...")
                        png_files = sorted(plots_path.glob(f"*_{env}.png"))
                        
                        for image_path in png_files:
                            uploaded_filename = self.upload_attachment(page_id, str(image_path))
                            if uploaded_filename:
                                logger.info(f"Uploaded chart: {image_path.name}")
                            else:
                                logger.warning(f"Failed to upload chart: {image_path.name}")
                    
                    # Generate static content that references the uploaded images
                    confluence_content = report_config['converter'](None)  # Pass None since we're not using HTML content
                
                # Handle domain comparison (HTML conversion)
                else:
                    file_path = reports_path / report_config['filename']
                    if not file_path.exists():
                        logger.warning(f"Report file not found: {file_path}")
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    confluence_content = report_config['converter'](html_content)

                # Update the page
                page_info = self.update_page(
                    page_id=page_id,
                    title=existing_page['title'],
                    content=confluence_content,
                    version=existing_page['version']['number']
                )

                published_pages.append(page_info)
                logger.info(f"Updated {report_config['title']} (ID: {page_id})")

            except Exception as e:
                logger.error(f"Failed to update {report_config['title']}: {str(e)}")
                continue

        return published_pages

    def _get_page_by_id(self, page_id: str) -> Optional[Dict]:
        """Get page by its ID with expanded version info."""
        url = f"{self.base_url}/rest/api/content/{page_id}"
        params = {'expand': 'version,body.storage'}
        response = self._make_request('GET', url, params=params)
        return response.json() if response.status_code == 200 else None

def run_analytics_and_publish(
    env: str = "prod",
    days_back: int = 30,
    max_pipelines_per_repo: int = 10,
    config_path: str = "config.json"
) -> None:
    """
    Run analytics and automatically publish reports to Confluence.
    
    Args:
        env: Environment (prod or nprod)
        days_back: Number of days to analyze
        max_pipelines_per_repo: Maximum pipelines per repository
        config_path: Path to Confluence configuration
    """
    try:
        # Import and run analytics
        from analytics import main as run_analytics
        
        logger.info(f"Running GitLab CI/CD analytics for {env} environment")
        analytics_result = run_analytics(
            env=env,
            days_back=days_back,
            max_pipelines_per_repo=max_pipelines_per_repo
        )
        
        # Publish reports to Confluence
        logger.info("Publishing reports to Confluence")
        publisher = ConfluencePublisher(config_path)
        published_pages = publisher.publish_analytics_reports("reports", env)
        
        if published_pages:
            logger.info("Successfully published reports:")
            for page in published_pages:
                logger.info(f"  - {page['title']}: {publisher.base_url}{page['_links']['webui']}")
        else:
            logger.warning("No reports were published")
    
    except Exception as e:
        logger.error(f"Failed to run analytics and publish: {e}")
        raise

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Publish GitLab CI/CD Analytics reports to Confluence')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Path to configuration file')
    parser.add_argument('--env', type=str, default='prod', choices=['prod', 'nprod'],
                       help='Environment (prod or nprod)')
    parser.add_argument('--reports-dir', type=str, default='reports',
                       help='Directory containing report files')
    parser.add_argument('--single-file', type=str,
                       help='Publish a single HTML file (provide file path)')
    parser.add_argument('--page-title', type=str,
                       help='Title for the page (required when using --single-file)')
    parser.add_argument('--run-analytics', action='store_true',
                       help='Run analytics before publishing')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Days back for analytics (when using --run-analytics)')
    parser.add_argument('--max-pipelines', type=int, default=10,
                       help='Max pipelines per repo (when using --run-analytics)')
    
    args = parser.parse_args()
    
    try:
        if args.run_analytics:
            # Run analytics and publish
            run_analytics_and_publish(
                env=args.env,
                days_back=args.days_back,
                max_pipelines_per_repo=args.max_pipelines,
                config_path=args.config
            )
        else:
            # Just publish existing reports
            publisher = ConfluencePublisher(args.config)
            
            if args.single_file:
                # Publish single file
                if not args.page_title:
                    logger.error("--page-title is required when using --single-file")
                    return
                
                page_info = publisher.publish_html_report(args.single_file, args.page_title)
                logger.info(f"Published page: {page_info['_links']['webui']}")
            else:
                # Publish all analytics reports
                published_pages = publisher.publish_analytics_reports(args.reports_dir, args.env)
                
                if published_pages:
                    logger.info("Published pages:")
                    for page in published_pages:
                        logger.info(f"  - {page['title']}: {publisher.base_url}{page['_links']['webui']}")
                else:
                    logger.warning("No reports were published")
    
    except Exception as e:
        logger.error(f"Failed to publish reports: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
