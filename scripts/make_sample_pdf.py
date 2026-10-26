"""Build data/sample_docs/atlas-sync-manual.pdf, a multi-page PDF used by the evaluation.

Atlas Sync is a fictional product. Its manual is full of error codes and config
keys, which is exactly the kind of text where keyword search beats vector search.

    python scripts/make_sample_pdf.py
"""

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "atlas-sync-manual.pdf"

PAGES = [
    (
        "Atlas Sync Manual - Overview and Installation",
        "Atlas Sync is a command-line tool that copies files between a laptop and a "
        "storage bucket. This manual describes a fictional product used to test the "
        "knowledge assistant.\n\n"
        "Atlas Sync requires Python 3.10 or newer. Install it with pip install atlas-sync. "
        "The installer creates a configuration folder at ~/.atlas. Run atlas init to "
        "create a default configuration file named atlas.toml.\n\n"
        "System requirements: at least 512 MB of free memory and 2 GB of free disk space "
        "for the local cache.",
    ),
    (
        "Configuration",
        "The configuration file atlas.toml contains the following keys.\n\n"
        "sync_interval sets how often Atlas checks for changes, in seconds. "
        "The default is 300 seconds.\n\n"
        "max_upload_mb limits the size of a single file upload. The default is 250 MB; "
        "larger files are skipped with a warning.\n\n"
        "retry_limit controls how many times a failed transfer is retried. "
        "The default is 5 attempts with exponential backoff.\n\n"
        "cache_dir sets the folder used for the local cache. The default is ~/.atlas/cache.\n\n"
        "encryption can be set to aes256 or none. When it is set to aes256, files are "
        "encrypted on the laptop before upload.\n\n"
        "bandwidth_limit caps the upload speed in kilobytes per second; 0 means unlimited.",
    ),
    (
        "Commands",
        "atlas push uploads changed local files to the bucket.\n\n"
        "atlas pull downloads files that changed in the bucket since the last sync.\n\n"
        "atlas status shows pending changes and the time of the last successful sync.\n\n"
        "atlas verify compares checksums of local and remote files and reports mismatches.\n\n"
        "atlas prune --older-than 30d deletes cached copies older than 30 days.\n\n"
        "atlas login opens a browser window to sign in. The access token is stored in the "
        "system keychain and expires after 12 hours.",
    ),
    (
        "Error Codes",
        "E1001: the configuration file atlas.toml could not be found. Run atlas init.\n\n"
        "E1007: the access token has expired. Run atlas login again.\n\n"
        "E1042: the local cache is full. Run atlas prune or increase the space available "
        "to cache_dir.\n\n"
        "E1043: a file exceeded max_upload_mb and was skipped.\n\n"
        "E1050: the network connection was lost during a transfer. Atlas retries up to "
        "retry_limit times.\n\n"
        "E2001: atlas verify found a checksum mismatch. The remote copy is uploaded again "
        "on the next push.\n\n"
        "E3000: the bucket does not exist or the account has no permission to write to it.",
    ),
    (
        "Troubleshooting",
        "Sync is slow: check that bandwidth_limit is not set too low.\n\n"
        "Atlas keeps asking me to sign in: access tokens are valid for 12 hours, so a new "
        "sign-in is needed twice a day.\n\n"
        "Files are missing after a pull: atlas pull never deletes local files. Deleted "
        "remote files are moved to ~/.atlas/trash instead.\n\n"
        "Running behind a proxy: set the HTTPS_PROXY environment variable before running "
        "Atlas.\n\n"
        "Reporting a bug: Atlas writes logs to ~/.atlas/logs/atlas.log. Include this file "
        "when reporting a bug.",
    ),
]


def main() -> None:
    doc = pymupdf.open()
    for title, body in PAGES:
        page = doc.new_page()  # A4 by default
        page.insert_text((72, 80), title, fontsize=16)
        overflow = page.insert_textbox(pymupdf.Rect(72, 110, 523, 770), body, fontsize=11)
        if overflow < 0:
            raise RuntimeError(f"Text did not fit on page '{title}'")
    doc.save(OUTPUT)
    doc.close()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
