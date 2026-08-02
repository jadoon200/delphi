"""Fetch the CC-BY Azure Functions 2019 trace into ignored local storage."""

from pathlib import Path

from delphi.data.fetch import fetch_azure_functions, sha256_file


def main() -> None:
    destination = Path("data/raw/azure-functions-2019.tar.xz")
    if destination.exists():
        print(f"already present: {destination} ({sha256_file(destination)})")
        return
    fetched = fetch_azure_functions(destination)
    print(f"downloaded: {fetched} ({sha256_file(fetched)})")


if __name__ == "__main__":
    main()
