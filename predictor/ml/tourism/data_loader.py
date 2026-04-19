import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"


def load_json(file_name: str):
    file_path = DATA_DIR / file_name
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merged_tourism_data():
    data = load_json("gabes_tourism_knowledge_base_merged.json")
    return data.get("places", [])


def load_heritage_data():
    data = load_json("gabes_heritage_master.json")
    return data.get("places", [])


def load_services_data():
    data = load_json("gabes_services.json")
    return data.get("services", [])


def load_sources_registry():
    return load_json("sources_registry.json")


def load_all_tourism_data():
    return {
        "places": load_merged_tourism_data(),
        "services": load_services_data(),
        "sources": load_sources_registry(),
    }