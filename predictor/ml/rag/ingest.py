import json
from pathlib import Path
from typing import List, Dict


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"


def load_json_file(file_name: str) -> Dict:
    file_path = DATA_DIR / file_name
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gabes_knowledge_base() -> List[Dict]:
    """
    Charge les deux fichiers JSON et concatène leurs projets.
    """
    files = ["gabes_details.json", "gabes_bullet_points.json"]
    all_projects = []

    for file_name in files:
        data = load_json_file(file_name)
        projets = data.get("projets", [])
        for project in projets:
            project["_source_file"] = file_name
            all_projects.append(project)

    return all_projects