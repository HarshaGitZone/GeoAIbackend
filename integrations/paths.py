import os
from typing import Optional


def get_workspace_root() -> str:
	"""Return the workspace root path by traversing up from this file.

	Assumes the repo layout like: <workspace>/GeoAI/backend/integrations/paths.py
	"""
	current_dir = os.path.dirname(os.path.abspath(__file__))
	backend_dir = os.path.dirname(current_dir)
	geoai_dir = os.path.dirname(backend_dir)
	workspace_dir = os.path.dirname(geoai_dir)
	return workspace_dir


def get_project_path(project_name: str) -> Optional[str]:
	"""Return absolute path for a sibling top-level project if it exists."""
	root = get_workspace_root()
	candidate = os.path.join(root, project_name)
	return candidate if os.path.exists(candidate) else None


