import yaml
import requests
import os
from pathlib import Path

def main():
    # Paths
    base_dir = Path(__file__).parent
    yaml_path = base_dir / "target_web.yaml"
    downloads_dir = base_dir / "downloads"

    # Ensure downloads directory exists
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Read target_web.yaml
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    web_targets = config.get("web", {})

    # Iterate and download
    for name, url in web_targets.items():
        print(f"Downloading {name} from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save the file
            file_path = downloads_dir / f"{name}.html"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            print(f"Successfully saved to {file_path}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

if __name__ == "__main__":
    main()
