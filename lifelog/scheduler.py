"""Simple macOS-compatible scheduler for the lifelog pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime

from lifelog.pipeline import run_pipeline


DEFAULT_RUN_TIME = "23:30"


def cron_example(
    *,
    python_executable: str = sys.executable,
    run_time: str = DEFAULT_RUN_TIME,
) -> str:
    hour, minute = run_time.split(":", 1)
    return f"{int(minute)} {int(hour)} * * * cd {__import__('os').getcwd()} && {python_executable} -m lifelog.pipeline"


def run_schedule_loop(run_time: str = DEFAULT_RUN_TIME) -> None:
    last_run_date: str | None = None
    while True:
        now = datetime.now()
        if now.strftime("%H:%M") == run_time and last_run_date != now.date().isoformat():
            run_pipeline()
            last_run_date = now.date().isoformat()
        time.sleep(30)


def install_cron_preview() -> None:
    print("Add this line with `crontab -e`:")
    print(cron_example())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or print the lifelog scheduler.")
    parser.add_argument("--loop", action="store_true", help="Run a simple foreground schedule loop.")
    parser.add_argument("--cron", action="store_true", help="Print a cron entry for daily 23:30 runs.")
    parser.add_argument("--run-time", default=DEFAULT_RUN_TIME, help="Daily run time in HH:MM.")
    args = parser.parse_args()

    if args.loop:
        run_schedule_loop(args.run_time)
    elif args.cron:
        print(cron_example(run_time=args.run_time))
    else:
        subprocess.run([sys.executable, "-m", "lifelog.pipeline"], check=False)


if __name__ == "__main__":
    main()
