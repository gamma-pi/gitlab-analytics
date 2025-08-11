import requests
from typing import List, Dict

import requests
import json
import textwrap

import base64
from urllib.parse import quote

# GitLab API details
gitlab_url = 'https://gitlab.com/'
private_token_svc = 'xxxGIBBERISHxxx'


pem_file = "utils/cert.pem"

import requests

def get_approval_rules(project_id: int, logger):
    """
    Fetches approval rules for a given GitLab project.

    Args:
        project_id (int): The ID of the GitLab project.

    Returns:
        List[Dict]: A list of approval rules.
    """
    try:
        url = f"{gitlab_url}/api/v4/projects/{project_id}/approval_rules"
        response = requests.get(url, headers=get_headers(), verify=pem_file)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch approval rules for project {project_id}: {e}")
        return []


def delete_approval_rule(project_id: int, rule_id: int, logger):
    """
    Deletes an approval rule for a given GitLab project.

    Args:
        project_id (int): The ID of the GitLab project.
        rule_id (int): The ID of the approval rule.
    """
    try:
        url = f"{gitlab_url}/api/v4/projects/{project_id}/approval_rules/{rule_id}"
        response = requests.delete(url, headers=get_headers(), verify=pem_file)
        response.raise_for_status()
        logger.info(f"Deleted approval rule {rule_id} for project {project_id}")
    except Exception as e:
        logger.error(f"Failed to delete approval rule {rule_id} for project {project_id}: {e}")


def get_headers() -> Dict[str, str]:
    """
    Returns the headers required for GitLab API requests.

    Returns:
        dict: Dictionary containing the authentication token.
    """
    return {'PRIVATE-TOKEN': private_token_svc}

def get_branches_in_repo(project_id: int, logger) -> List[str]:
    """
    Fetches all branches in the given GitLab repository.

    Args:
        project_id (int): The ID of the GitLab repository.
        logger: Logger instance for logging messages.

    Returns:
        List[str]: A list of branch names.

    Raises:
        Exception: If the API request fails.
    """
    try:
        url = f'{gitlab_url}/api/v4/projects/{project_id}/repository/branches'
        
        response = requests.get(url, headers=get_headers(), verify=pem_file)
        response.raise_for_status()
        
        branches = response.json()
        
        logger.info(f"Fetched all branches for GitLab repository ID: {project_id}")
        
        return [branch['name'] for branch in branches]
    
    except Exception as e:
        logger.error(f"Error fetching branches: {e}")
        return []

def get_repositories(group_id: str, logger) -> List[Dict]:
    """
    Fetches all repositories in the given GitLab group.

    Args:
        group_id (str): The ID of the GitLab group.

    Returns:
        list: A list of dictionaries representing repositories.

    Raises:
        Exception: If the API request fails.
    """
    try:

        url = f'{gitlab_url}/api/v4/groups/{group_id}/projects'

        response = requests.get(url, headers=get_headers(), verify=pem_file)
        
        response.raise_for_status()

        logger.info(f"Fetched all repositories for GitLab group: {group_id}")

        return response.json()
    
    except Exception as e:

        logger.error(f"Error fetching repositories: {e}")

        return []


def get_subgroups(group_id: str, logger) -> List[Dict]:
    """
    Fetches all subgroups within a given GitLab group.

    Args:
        group_id (str): The ID of the GitLab group.

    Returns:
        list: A list of dictionaries representing subgroups.

    Raises:
        Exception: If the API request fails.
    """
    try:

        url = f'{gitlab_url}/api/v4/groups/{group_id}/subgroups'
        
        response = requests.get(url, headers=get_headers(), verify=pem_file)
        
        response.raise_for_status()

        logger.info(f"Fetched all subgroups within GitLab group: {group_id}")
        
        return response.json()
    
    except Exception as e:

        logger.error(f"Error fetching subgroups: {e}")

        return []


def get_protected_branches(project_id: int, logger) -> List[Dict]:
    """
    Fetches the list of protected branches for a given project.

    Args:
        project_id (int): The ID of the GitLab project.

    Returns:
        list: A list of dictionaries representing protected branches.

    Raises:
        Exception: If the API request fails.
    """
    try:

        url = f'{gitlab_url}/api/v4/projects/{project_id}/protected_branches'

        response = requests.get(url, headers=get_headers(), verify=pem_file)

        response.raise_for_status()

        logger.info(f"Fetched all protected branches for a given project: {project_id}")
        
        return response.json()
    
    except Exception as e:

        logger.error(f"Error fetching protected branches: {e}")

        return []
    

def create_gitlab_repo(subdomain_id: int, repo_name: str, logger) -> int:

    """
    Creates a new repository in the given subdomain.

    Args:
        subdomain_id (int): ID of the subdomain where the repo should be created.
        repo_name (str): Name of the new repository.
        logger: Logger instance to log actions.

    Returns:
        int: Project ID of the newly created repo if successful, else -1.
    """
    url = f"{gitlab_url}/api/v4/projects"
    
    data = {
        "name": repo_name,
        "path": repo_name.lower(),
        "namespace_id": subdomain_id,
        "visibility": "private",
        "initialize_with_readme": True
    }
    
    response = requests.post(url, headers=get_headers(), json=data, verify=pem_file)
    
    if response.status_code == 201:
        project_id = response.json()["id"]
        logger.info(f"Created repository '{repo_name}' with ID {project_id} in subdomain {subdomain_id}")
        return project_id
    else:
        logger.error(f"Failed to create repository '{repo_name}' in subdomain {subdomain_id}: {response.text}")
        return -1


def create_branch_in_repo(project_id: int, branch: str, logger) -> bool:
    """
    Creates a new branch in a GitLab repository.

    Args:
        project_id (int): GitLab project ID.
        branch (str): Branch name to create.
        logger: Logger instance to log actions.

    Returns:
        bool: True if the branch was created successfully, False otherwise.

    Raises:
        Exception: If the API request fails.
    """

    try:
        url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/branches"
        
        data = {
            "branch": branch,
            "ref": "main"
        }
        
        response = requests.post(url, headers=get_headers(), json=data, verify=pem_file)
        
        if response.status_code == 201:
            logger.info(f"Created branch '{branch}' in project {project_id}")
            return True
        else:
            logger.error(f"Failed to create branch '{branch}' in project {project_id}: {response.text}")
            return False
    except:
        logger.error(f"Failed to create branch '{branch}' in project {project_id}")
        return False


def process_subdomains(domain_name: str, domain_id: int, process_task: callable, logger, *args, **kwargs) -> None:
    """
    Processes subdomains within a given domain group.

    Args:
        domain_name (str): The name of the domain group.
        domain_id (int): The ID of the domain group.
        process_task (callable): A callable to process each subdomain.
        logger (logging.Logger): Logger instance to log actions.
        *args (list): Additional arguments to pass to the process_task.
        **kwargs (dict): Additional keyword arguments to pass to the process_task.

    Raises:
        Exception: If the API request fails while processing the subdomains.
    """
    
    subdomains = get_subgroups(domain_id, logger)
    
    for subdomain in subdomains:
        try:
            subdomain_id = subdomain['id']
            subdomain_name = subdomain['name']

            logger.info(f"Processing subdomain: {subdomain_name}")
            
            process_task(subdomain_id, subdomain_name, domain_id, domain_name, logger, *args, **kwargs)

            logger.info(f"Finished processing subdomain: {subdomain_name}")

        except Exception as e:
            logger.error(f"Error while processing subdomain {subdomain_id}: {e}")

def set_approval_rule(payload: dict, project_id: int) -> bool:
    """
    Sets an approval rule for a given project.

    Args:
        payload (dict): Payload with required parameters to create an approval rule.
        project_id (int): The ID of the GitLab project.

    Returns:
        bool: True if setting approval rule is successful, False otherwise.

    Raises:
        Exception: If the API request fails.
    """
    try:
        approval_url = f"{gitlab_url}/api/v4/projects/{project_id}/approval_rules"

        response = requests.post(approval_url, data=payload, headers=get_headers(), verify=pem_file)

        response.raise_for_status()

        return True
    
    except Exception as e:
        
        raise e
    
def get_repository_tree(project_id: int, branch: str) -> List[Dict]:
    """
    Fetches the repository tree for a given project and branch.

    Args:
        project_id (int): The ID of the GitLab project.
        branch (str): The name of the branch.

    Returns:
        list: A list of dictionaries representing the repository tree.

    Raises:
        Exception: If the API request fails.
    """
    url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/tree?ref={branch}"

    response = requests.get(url, headers=get_headers(), verify=pem_file)
    
    return response.json() if response.status_code == 200 else []


def delete_file(project_id: int, file_path: str, branch: str) -> None:
    """
    Deletes a file from a specified branch in a GitLab project repository.

    Args:
        project_id (int): The ID of the GitLab project.
        file_path (str): The path of the file to be deleted in the repository.
        branch (str): The name of the branch from which the file will be deleted.

    Raises:
        Exception: If the API request fails.
    """
    url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/files/{file_path}"

    data = {"branch": branch, "commit_message": "Removing old config"}

    requests.delete(url, headers=get_headers(), json=data, verify=pem_file)


def file_exists_in_gitlab(project_id: int, file_path: str, branch: str, logger) -> bool:
    """
    Checks if a file exists in a GitLab repository.

    Args:
        project_id (int): The GitLab project ID.
        file_path (str): The path of the file in the repo.
        branch (str): The branch to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/files/{file_path}?ref={branch}"
        response = requests.get(url, headers=get_headers(), verify=pem_file)

        logger.info(f"Checking if {file_path} exists in project {project_id}")

        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking if {file_path} exists in project {project_id}: {e}")
        return False
    


def delete_re_add_file_to_gitlab(project_id: int, branch: str, file_path: str, content: str, logger, is_binary: bool = False) -> bool:
    """
    Adds or updates a file in the specified branch of a GitLab project.
    Automatically deletes old version if it exists.

    Args:
        project_id (int): GitLab project ID.
        branch (str): Target branch.
        file_path (str): Path inside the repo.
        content (str): File content (base64-encoded if binary).
        is_binary (bool): Set True if the file is binary (e.g., .jar, .crt, .pem).

    Returns:
        bool: True if file successfully added.
    """
    encoded_path = quote(file_path, safe='')

    # Delete if exists
    if file_exists_in_gitlab(project_id, file_path, branch, logger):
        #logger.info(f"{file_path} already exists in {branch} branch in project {project_id}. Skipping...")
        #return
        delete_file_from_gitlab(project_id, branch, file_path, logger)

    url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/files/{encoded_path}"

    data = {
        "branch": branch,
        "content": content,
        "commit_message": f"Add {file_path} [skip ci]",
        "encoding": "base64" if is_binary else "text"
    }

    response = requests.post(url, headers=get_headers(), json=data, verify=pem_file)

    if response.status_code in [200, 201]:
        logger.info(f"Added: {file_path}")
        return True
    else:
        logger.error(f"Failed to add {file_path}: {response.status_code} - {response.text}")
        return False


def delete_file_from_gitlab(project_id: int, branch: str, file_path: str, logger) -> bool:
    """
    Deletes a file from a GitLab repository.

    Args:
        project_id (int): GitLab project ID.
        branch (str): Target branch.
        file_path (str): File path to delete.

    Returns:
        bool: True if deleted.
    """
    
    encoded_path = quote(file_path, safe='')
    url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/files/{encoded_path}"

    data = {
        "branch": branch,
        "commit_message": f"Delete {file_path} [skip ci]"
    }

    response = requests.delete(url, headers=get_headers(), json=data, verify=pem_file)

    if response.status_code == 204:
        logger.info(f"Deleted: {file_path}")
        return True
    else:
        logger.warning(f"Could not delete {file_path}: {response.status_code} - {response.text}")
        return False


def delete_branch_content(project_id: int, branch: str, logger) -> None:
    """
    Deletes all files in a specified branch of a GitLab project repository.

    Args:
        project_id (int): The ID of the GitLab project.
        branch (str): The name of the branch from which files will be deleted.
        logger: Logger instance to log the operation details.

    Raises:
        Exception: If the API request fails while fetching or deleting files.
    """

    files = get_repository_tree(project_id, branch)

    for file in files:
        delete_file(project_id, file["path"], branch)
        
    logger.info(f"Deleted all files in {branch} branch for project {project_id}")


def get_repo(subdomain_id: int, repo_name, logger) -> dict:
    """
    Fetches all repositories in a given subdomain and returns the one matching the given name.

    Args:
        subdomain_id (int): ID of the subdomain.
        repo_name (str): Name of the repository.
        logger: Logger instance to log the operation details.

    Returns:
        dict: Repository matching the given name.
    """
    repos = get_repositories(subdomain_id, logger)

    for repo in repos:
        if repo["name"].lower() == repo_name.lower():
            return repo

def init_repo(subdomain_id: int, subdomain_name: str, domain_id: int, domain_name: str, logger, *args, **kwargs) -> None:

    """
    Initializes a new GitLab repository in the given subdomain.

    Args:
        subdomain_id (int): The ID of the subdomain.
        subdomain_name (str): The name of the subdomain.
        domain_id (int): The ID of the domain group.
        domain_name (str): The name of the domain group.
        logger (logging.Logger): Logger instance to log the operation details.
        *args (list): Additional arguments to pass to the process_task.
        **kwargs (dict): Additional keyword arguments to pass to the process_task.

    Raises:
        ValueError: If the required 'repo_name' keyword argument is not provided.
    """
    repo_name = kwargs.get("repo_name", None)

    if repo_name is None:
        raise ValueError("repo_name is required")

    repos = get_repositories(subdomain_id, logger)

    existing_repos = [repo["name"].lower() for repo in repos]
    
    if repo_name.lower() in existing_repos:
        logger.info(f"Repo '{repo_name}' already exists in subdomain {subdomain_name}. Skipping creation.")
        repo = get_repo(subdomain_id, repo_name, logger)
        project_id = repo["id"]
    else:
        project_id = create_gitlab_repo(subdomain_id, repo_name, logger)
    

    # Fetch existing branches in the repository
    existing_branches = get_branches_in_repo(project_id, logger)  # Assuming this function returns a list of branch names

    for branch in ["dev", "sit"]:
        if branch not in existing_branches:
            create_branch_in_repo(project_id, branch, logger)
        else:
            logger.info(f"Branch '{branch}' already exists in repository '{repo_name}'. Skipping creation.")
