from pathlib import Path


project_root = Path(__file__).resolve().parent

directories = [
    project_root / "data" / "raw",
    project_root / "data" / "processed",
    project_root / "models",
    project_root / "notebooks",
    project_root / "logs",
    project_root / "scripts",
    project_root / "src",
    project_root / "tests",
    project_root / "config",
    project_root / "reports",
]


for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Created: {directory}")

print("\nProject directory structure is ready.")
