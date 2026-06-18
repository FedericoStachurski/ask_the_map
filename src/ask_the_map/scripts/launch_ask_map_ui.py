from pathlib import Path
import subprocess
import sys


def main():
    ui_script = Path(__file__).with_name("ask_map_ui.py")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_script),
        "--",
        *sys.argv[1:],
    ]

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()