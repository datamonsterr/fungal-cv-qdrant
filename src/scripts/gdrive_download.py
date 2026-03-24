"""
Recursive Google Drive downloader.

Usage:
    python src/scripts/gdrive_download.py \
        --url <SHARED_DRIVE_URL> \
        --target <LOCAL_DIR> \
        [--quiet] [--fuzzy] [--force]
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional


def extract_drive_id(url: str) -> Optional[str]:
    """Extract the Google Drive file or folder ID from various URL formats.

    Supported formats:
    - https://drive.google.com/drive/folders/<ID>?usp=sharing
    - https://drive.google.com/file/d/<ID>/view
    - https://drive.google.com/open?id=<ID>
    - https://drive.google.com/uc?id=<ID>

    Args:
        url: Google Drive shared URL.

    Returns:
        The extracted Drive ID, or None if not found.
    """
    patterns = [
        r"/drive/folders/([a-zA-Z0-9_-]+)",
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_folder_url(url: str) -> bool:
    """Determine if the URL points to a Google Drive folder.

    Args:
        url: Google Drive shared URL.

    Returns:
        True if the URL points to a folder, False if it points to a file.
    """
    return "/drive/folders/" in url


def download_with_retry(
    drive_id: str,
    output_path: Path,
    is_folder: bool,
    quiet: bool,
    fuzzy: bool,
    force: bool,
    max_retries: int = 3,
) -> bool:
    """Download a file or folder from Google Drive with exponential backoff retry.

    Args:
        drive_id: The Google Drive file or folder ID.
        output_path: Local path to save downloaded content.
        is_folder: True if downloading a folder, False for a file.
        quiet: Suppress progress output if True.
        fuzzy: Use fuzzy ID matching if True.
        force: Re-download even if file already exists.
        max_retries: Maximum number of retry attempts.

    Returns:
        True if download succeeded, False otherwise.
    """
    import gdown  # type: ignore[import-untyped]

    attempt = 0
    delay = 5

    while attempt < max_retries:
        try:
            if is_folder:
                result = gdown.download_folder(
                    id=drive_id,
                    output=str(output_path),
                    quiet=quiet,
                    use_cookies=False,
                    remaining_ok=True,
                )
                return result is not None
            else:
                url = f"https://drive.google.com/uc?id={drive_id}"
                result = gdown.download(
                    url=url,
                    output=str(output_path),
                    quiet=quiet,
                    fuzzy=fuzzy,
                    resume=not force,
                )
                return result is not None
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "too many requests" in err_str:
                print(
                    "\nGoogle Drive download quota exceeded. "
                    "Please wait 24 hours before retrying.",
                    file=sys.stderr,
                )
                return False
            if "permission" in err_str or "access" in err_str or "forbidden" in err_str:
                print(
                    f"\nAccess denied: {exc}\n"
                    "Ensure the file/folder is shared publicly (Anyone with the link).",
                    file=sys.stderr,
                )
                return False
            attempt += 1
            if attempt < max_retries:
                print(
                    f"\nDownload error (attempt {attempt}/{max_retries}): {exc}",
                    file=sys.stderr,
                )
                print(f"Retrying in {delay}s...", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
            else:
                print(
                    f"\nDownload failed after {max_retries} attempts: {exc}",
                    file=sys.stderr,
                )
                return False

    return False


def count_files(directory: Path) -> int:
    """Recursively count files in a directory.

    Args:
        directory: Root directory to count from.

    Returns:
        Total number of files found.
    """
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file())


def run_download(
    url: str,
    target: str,
    force: bool = False,
    quiet: bool = False,
    fuzzy: bool = False,
) -> int:
    """Download a Google Drive file or folder to a local directory.

    Args:
        url: Shared Google Drive URL (file or folder).
        target: Local directory path to download into.
        force: If True, re-download files that already exist.
        quiet: If True, suppress progress bars and status output.
        fuzzy: If True, use fuzzy ID matching for file URLs.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    try:
        import gdown  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        print(
            "gdown is not installed. Run: uv add gdown",
            file=sys.stderr,
        )
        return 1

    # Parse the Drive ID from the URL
    drive_id = extract_drive_id(url)
    if not drive_id:
        print(
            f"Could not extract a Google Drive ID from URL: {url}\n"
            "Supported formats:\n"
            "  https://drive.google.com/drive/folders/<ID>?usp=sharing\n"
            "  https://drive.google.com/file/d/<ID>/view\n"
            "  https://drive.google.com/open?id=<ID>",
            file=sys.stderr,
        )
        return 1

    folder_download = is_folder_url(url)
    target_path = Path(target)

    if not quiet:
        print(f"Google Drive ID : {drive_id}")
        print(f"Type            : {'folder' if folder_download else 'file'}")
        print(f"Target          : {target_path.resolve()}")

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)

    # For files: check if already downloaded (resume-safe)
    if not folder_download and not force:
        existing = list(target_path.iterdir()) if target_path.exists() else []
        if existing and not quiet:
            print(
                f"Target directory already contains {len(existing)} item(s). "
                "Use --force to re-download."
            )

    # Count files before download (for skip reporting)
    files_before = count_files(target_path)

    if not quiet:
        print("\nStarting download...")

    success = download_with_retry(
        drive_id=drive_id,
        output_path=target_path,
        is_folder=folder_download,
        quiet=quiet,
        fuzzy=fuzzy,
        force=force,
    )

    if not success:
        print("\nDownload failed.", file=sys.stderr)
        return 1

    # Summary
    files_after = count_files(target_path)
    files_downloaded = files_after - files_before
    files_skipped = files_before if not force else 0

    if not quiet:
        print("\n--- Download Summary ---")
        print(f"Files downloaded : {max(files_downloaded, 0)}")
        print(f"Files skipped    : {files_skipped}")
        print(f"Total files now  : {files_after}")
        print(f"Location         : {target_path.resolve()}")
        print("Done.")

    return 0


def main() -> None:
    """Entry point for the gdrive_download CLI."""
    parser = argparse.ArgumentParser(
        description="Recursively download a Google Drive file or folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Shared Google Drive URL (file or folder)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Local directory to download into",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Use fuzzy ID matching for file URLs",
    )

    args = parser.parse_args()
    sys.exit(run_download(args.url, args.target, args.force, args.quiet, args.fuzzy))


if __name__ == "__main__":
    main()
