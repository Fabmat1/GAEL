#!/usr/bin/env python3
"""
Run GAEL fitting on multiple directories in parallel with live progress display.
Skips already processed folders by checking for fit_results/{folder_nr}/fit_parameters.csv

Now logs per-folder output and an aggregated errors.log if failures occur.
"""

import subprocess
import threading
import queue
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import argparse
from datetime import datetime, timedelta   # ← add timedelta
import shutil
import math
import textwrap

# ANSI color codes for terminal output
COLORS = {
    'HEADER': '\033[95m',
    'OKBLUE': '\033[94m',
    'OKCYAN': '\033[96m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'DIM': '\033[2m',
}

class GAELRunner:
    def __init__(self, start_idx, end_idx, max_workers=16,
                 lines_to_show=2, force_rerun=False):
        self.start_idx      = start_idx
        self.end_idx        = end_idx
        self.max_workers    = max_workers
        self.lines_to_show  = lines_to_show
        self.force_rerun    = force_rerun
        self.lock           = threading.Lock()
        self.completed      = 0
        self.skipped        = 0
        self.total          = end_idx - start_idx + 1
        self.folders_to_process = []
        self.worker_status  = {}
        self.worker_output  = {}
        self.worker_tail    = {}
        self.start_ts       = time.time()     # <-- NEW: needed for ETA

        # Logging ------------------------------------------
        self.log_dir        = "logs"
        self.error_log_path = os.path.join(self.log_dir, "errors.log")
        self._ensure_log_dir()

        # Decide what really must be done -------------------
        self._filter_folders_to_process()

    def _ensure_log_dir(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            # If we can't create logs dir, continue without crashing; logs just won't be written
            pass

    def _append_errors_log(self, text):
        try:
            with self.lock:
                with open(self.error_log_path, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
        except Exception:
            # Avoid crashing on logging failures
            pass

    def _log_failure_summary(self, folder_idx, cmd, return_code=None, exception=None, log_path=None, tail_lines=None, duration_s=None, started_at=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        folder_name = f"{folder_idx:04d}"
        header = f"[{ts}] Folder {folder_name} FAILED"
        if return_code is not None:
            header += f" (exit code {return_code})"
        if duration_s is not None:
            header += f" | duration {duration_s:.1f}s"
        if started_at is not None:
            header += f" | started {started_at}"

        lines = [header]
        lines.append(f"  Command: {' '.join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)}")
        if log_path:
            lines.append(f"  Log file: {os.path.abspath(log_path)}")
        if exception is not None:
            lines.append(f"  Exception: {repr(exception)}")

        if tail_lines:
            lines.append("  Last output lines:")
            for ln in tail_lines:
                # indent and make sure no stray ANSI codes in aggregate error log
                lines.append("    " + self.strip_ansi(ln))

        self._append_errors_log("\n".join(lines) + "\n" + ("-" * 80))

    def _filter_folders_to_process(self):
        """Filter out folders that have already been processed (unless force_rerun is True)."""
        all_folders = list(range(self.start_idx, self.end_idx + 1))

        if self.force_rerun:
            self.folders_to_process = all_folders
            print(f"{COLORS['WARNING']}Force rerun enabled - will process all {len(all_folders)} folders{COLORS['ENDC']}")
            return

        skipped_folders = []
        for folder_idx in all_folders:
            folder_name = f"{folder_idx:04d}"
            results_file = f"fit_results/{folder_name}/fit_parameters.csv"

            if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
                skipped_folders.append(folder_idx)
            else:
                self.folders_to_process.append(folder_idx)

        self.skipped = len(skipped_folders)

        print(f"{COLORS['OKGREEN']}Found {len(self.folders_to_process)} folders to process{COLORS['ENDC']}")
        print(f"{COLORS['OKCYAN']}Skipping {self.skipped} already processed folders{COLORS['ENDC']}")

        if skipped_folders and len(skipped_folders) <= 20:
            skipped_str = ', '.join(f'{idx:04d}' for idx in skipped_folders)
            print(f"{COLORS['DIM']}Skipped: {skipped_str}{COLORS['ENDC']}")
        elif skipped_folders:
            print(f"{COLORS['DIM']}Skipped folders: {min(skipped_folders):04d} - {max(skipped_folders):04d} (and others){COLORS['ENDC']}")

        if not self.folders_to_process:
            print(f"{COLORS['OKGREEN']}All folders already processed! Use --force to rerun all.{COLORS['ENDC']}")

    def is_already_processed(self, folder_idx):
        """Check if a folder has already been processed."""
        folder_name = f"{folder_idx:04d}"
        results_file = f"fit_results/{folder_name}/fit_parameters.csv"
        return os.path.exists(results_file) and os.path.getsize(results_file) > 0

    def run_single_gael(self, folder_idx):
        """Run GAEL for a single folder and capture output (and log errors if they happen)."""
        folder_name = f"{folder_idx:04d}"
        input_file = f"{folder_name}/input.json"
        log_path = os.path.join(self.log_dir, f"{folder_name}.log")

        # Double-check if already processed (in case of race conditions)
        if not self.force_rerun and self.is_already_processed(folder_idx):
            with self.lock:
                self.worker_status[folder_idx] = "✓ Skipped (already done)"
                self.completed += 1
            return folder_idx, True

        # Initialize worker status and tails
        with self.lock:
            self.worker_status[folder_idx] = "Starting"
            self.worker_output[folder_idx] = deque(maxlen=self.lines_to_show)
            self.worker_tail[folder_idx] = deque(maxlen=200)

        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()
        cmd = ["GAEL", "--fit", input_file, "--threads", "1", "--no-plots", "--no-pdf"]

        try:
            # Run GAEL command with unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # In case GAEL is a Python script

            # Open per-folder log and write header
            log_fh = None
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
                log_fh.write(f"# Folder: {folder_name}\n")
                log_fh.write(f"# Started: {started_at}\n")
                log_fh.write(f"# Command: {' '.join(cmd)}\n")
                log_fh.write("# Output:\n")
            except Exception as e_open:
                # If we cannot open log file, continue without per-folder logging
                log_fh = None

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # line-buffered
                env=env
            )

            # Update status
            with self.lock:
                self.worker_status[folder_idx] = "Running"

            # Read output line by line more responsively
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                line = line.rstrip("\n")
                if line:
                    with self.lock:
                        self.worker_output[folder_idx].append(line)
                        self.worker_tail[folder_idx].append(line)
                    if log_fh:
                        try:
                            log_fh.write(line + "\n")
                        except Exception:
                            pass

            # Wait for process to complete
            return_code = process.wait()
            t1 = time.time()
            duration = t1 - t0

            # Close log file handle
            if log_fh:
                try:
                    log_fh.write(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_fh.write(f"# Return code: {return_code}\n")
                    log_fh.close()
                except Exception:
                    pass

            with self.lock:
                if return_code == 0:
                    self.worker_status[folder_idx] = "✓ Complete"
                    self.completed += 1
                else:
                    self.worker_status[folder_idx] = f"✗ Failed (code {return_code})"

            if return_code != 0:
                # Log a failure summary with last lines of output
                tail_lines = list(self.worker_tail.get(folder_idx, []))[-25:]
                self._log_failure_summary(
                    folder_idx=folder_idx,
                    cmd=cmd,
                    return_code=return_code,
                    log_path=log_path if os.path.exists(self.log_dir) else None,
                    tail_lines=tail_lines,
                    duration_s=duration,
                    started_at=started_at
                )

            return folder_idx, return_code == 0

        except Exception as e:
            # Try to write exception to the per-folder log
            try:
                with open(log_path, "a", encoding="utf-8") as log_fh:
                    log_fh.write(f"\n# Exception while running folder {folder_name}: {repr(e)}\n")
            except Exception:
                pass

            with self.lock:
                self.worker_status[folder_idx] = f"✗ Error: {str(e)}"

            # Log to aggregated errors.log
            tail_lines = list(self.worker_tail.get(folder_idx, []))[-25:]
            self._log_failure_summary(
                folder_idx=folder_idx,
                cmd=cmd,
                exception=e,
                log_path=log_path if os.path.exists(self.log_dir) else None,
                tail_lines=tail_lines,
                duration_s=(time.time() - t0),
                started_at=started_at
            )
            return folder_idx, False

    def format_worker_box(self, folder_idx, box_width):
        """Format a single worker's status box."""
        folder_name = f"{folder_idx:04d}"
        status = self.worker_status.get(folder_idx, "Waiting")

        # Choose color based on status
        if "Complete" in status:
            color = COLORS['OKGREEN']
        elif "Skipped" in status:
            color = COLORS['OKBLUE']
        elif "Failed" in status or "Error" in status:
            color = COLORS['FAIL']
        elif status == "Running":
            color = COLORS['OKCYAN']
        else:
            color = COLORS['WARNING']

        # Create box lines
        lines = []

        # Header with folder name and status
        header = f"{color}[{folder_name}]{COLORS['ENDC']} {status}"
        if len(header) > box_width - 2:
            header = header[:box_width-5] + "..."
        lines.append(header)

        # Output lines (if running)
        if status == "Running" and folder_idx in self.worker_output:
            for line in self.worker_output[folder_idx]:
                if len(line) > box_width - 4:
                    line = line[:box_width-7] + "..."
                lines.append(f"{COLORS['DIM']}  {line}{COLORS['ENDC']}")

        # Pad to consistent height
        while len(lines) < self.lines_to_show + 1:
            lines.append("")

        return lines

    def display_progress(self):
        """Live progress display without flicker and with ETA."""
        total_to_process = len(self.folders_to_process)

        # hide the cursor while the live view is active
        sys.stdout.write('\x1b[?25l')
        sys.stdout.flush()

        try:
            while self.completed < total_to_process:
                frame = []                       # collect complete frame here
                term_w, _ = shutil.get_terminal_size()

                # ─── Header ────────────────────────────────────────────────
                bar = '=' * min(term_w, 100)
                frame.append('\x1b[H\x1b[J')    # cursor home + clear screen
                frame.append(f"{COLORS['HEADER']}{bar}{COLORS['ENDC']}")
                frame.append(
                    f"{COLORS['BOLD']}GAEL Parallel Fitting Progress"
                    f"{COLORS['ENDC']}".center(min(term_w, 100))
                )
                frame.append(f"{COLORS['HEADER']}{bar}{COLORS['ENDC']}")

                # ─── Progress Bar ─────────────────────────────────────────
                if total_to_process:
                    progress = self.completed / total_to_process
                    bar_len  = min(50, term_w - 40)
                    filled   = int(bar_len * progress)
                    bar_txt  = '█' * filled + '░' * (bar_len - filled)
                    frame.append(
                        f"\n{COLORS['OKGREEN']}Progress:{COLORS['ENDC']} "
                        f"[{bar_txt}] {progress*100:5.1f}% "
                        f"({self.completed}/{total_to_process})"
                    )
                else:
                    frame.append(
                        f"\n{COLORS['OKGREEN']}All folders already processed!"
                        f"{COLORS['ENDC']}"
                    )

                # ─── ETA ──────────────────────────────────────────────────
                elapsed = time.time() - self.start_ts
                if self.completed:
                    avg        = elapsed / self.completed
                    remain_s   = avg * (total_to_process - self.completed)
                    eta_abs_dt = datetime.now() + timedelta(seconds=remain_s)
                    eta_line = (
                        f"{COLORS['OKCYAN']}ETA:{COLORS['ENDC']} "
                        f"{eta_abs_dt.strftime('%H:%M:%S')}  "
                        f"({int(remain_s)//60:02d}:{int(remain_s)%60:02d} remaining)"
                    )
                else:
                    eta_line = f"{COLORS['OKCYAN']}ETA:{COLORS['ENDC']} --:--:--"
                frame.append(eta_line)

                # ─── Counters ─────────────────────────────────────────────
                with self.lock:
                    active  = sum(1 for s in self.worker_status.values()
                                  if s == "Running")
                    finished = sum(1 for s in self.worker_status.values()
                                   if "Complete" in s)
                    failed  = sum(1 for s in self.worker_status.values()
                                  if "Failed" in s or "Error" in s)
                    skipped = sum(1 for s in self.worker_status.values()
                                  if "Skipped" in s)
                frame.append(
                    f"{COLORS['OKCYAN']}Active:{COLORS['ENDC']} {active} | "
                    f"{COLORS['OKGREEN']}Complete:{COLORS['ENDC']} {finished} | "
                    f"{COLORS['FAIL']}Failed:{COLORS['ENDC']} {failed} | "
                    f"{COLORS['OKBLUE']}Skipped:{COLORS['ENDC']} {skipped}"
                )

                # ─── Worker grid (reuse existing helpers) ─────────────────
                n_cols     = 4
                box_width  = (term_w - (n_cols + 1) * 2) // n_cols
                box_width  = max(box_width, 30)

                frame.append("")                 # blank line before grid
                frame.append(f"{COLORS['BOLD']}Active Workers:{COLORS['ENDC']}")
                frame.append("─" * min(term_w, 100))

                with self.lock:
                    # build list of workers to show
                    running = [idx for idx, s in self.worker_status.items()
                               if s == "Running"]
                    others  = [idx for idx in self.worker_status.keys()
                               if idx not in running]
                    workers_to_show = running + others[:self.max_workers-len(running)]

                for row_start in range(0, len(workers_to_show), n_cols):
                    row = workers_to_show[row_start:row_start+n_cols]

                    boxes, max_lines = [], 0
                    for idx in row:
                        b = self.format_worker_box(idx, box_width)
                        boxes.append(b)
                        max_lines = max(max_lines, len(b))

                    for line_idx in range(max_lines):
                        row_line = []
                        for col, box in enumerate(boxes):
                            content = box[line_idx] if line_idx < len(box) else ""
                            content += " " * (box_width - len(self.strip_ansi(content)))
                            row_line.append(content)
                        frame.append("  " + "  ".join(row_line))

                    if row_start + n_cols < len(workers_to_show):
                        frame.append("")          # extra blank line between rows

                # ─── Flush frame to terminal ───────────────────────────────
                sys.stdout.write('\n'.join(frame) + '\n')
                sys.stdout.flush()

                time.sleep(0.1)   # refresh interval

        finally:
            # show the cursor again
            sys.stdout.write('\x1b[?25h')
            sys.stdout.flush()

    def strip_ansi(self, text):
        """Remove ANSI color codes from text for proper length calculation."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def run(self):
        """Run GAEL fitting in parallel."""
        if not self.folders_to_process:
            print(f"\n{COLORS['OKGREEN']}Nothing to do - all folders already processed!{COLORS['ENDC']}")
            return []

        # Start display thread
        display_thread = threading.Thread(target=self.display_progress, daemon=True)
        display_thread.start()

        # Run GAEL in parallel
        failed_folders = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks for folders that need processing
            futures = {
                executor.submit(self.run_single_gael, idx): idx
                for idx in self.folders_to_process
            }

            # Process completed tasks
            for future in as_completed(futures):
                folder_idx, success = future.result()
                if not success:
                    failed_folders.append(folder_idx)

        # Final display
        os.system('clear' if os.name == 'posix' else 'cls')
        terminal_width = shutil.get_terminal_size().columns

        print(f"{COLORS['HEADER']}{'='*min(terminal_width, 100)}{COLORS['ENDC']}")
        print(f"{COLORS['BOLD']}GAEL Fitting Complete!{COLORS['ENDC']}".center(min(terminal_width, 100)))
        print(f"{COLORS['HEADER']}{'='*min(terminal_width, 100)}{COLORS['ENDC']}")

        processed_count = len(self.folders_to_process) - len(failed_folders)
        print(f"\n{COLORS['OKGREEN']}✓ Successfully processed: {processed_count}/{len(self.folders_to_process)}{COLORS['ENDC']}")

        if self.skipped > 0:
            print(f"{COLORS['OKBLUE']}⚡ Skipped (already done): {self.skipped}{COLORS['ENDC']}")

        if failed_folders:
            print(f"{COLORS['FAIL']}✗ Failed folders: {len(failed_folders)}{COLORS['ENDC']}")
            failed_str = ', '.join(f'{idx:04d}' for idx in sorted(failed_folders))
            # Wrap long lists
            if len(failed_str) > terminal_width - 10:
                wrapped = textwrap.wrap(failed_str, width=terminal_width - 10)
                for line in wrapped:
                    print(f"   {line}")
            else:
                print(f"   {failed_str}")

        # Save failed folders to file if any
        if failed_folders:
            with open('failed_folders.txt', 'w') as f:
                for idx in sorted(failed_folders):
                    f.write(f"{idx:04d}\n")
            print(f"\n{COLORS['WARNING']}Failed folder list saved to 'failed_folders.txt'{COLORS['ENDC']}")
            print(f"{COLORS['WARNING']}See detailed logs in '{self.log_dir}/' and aggregated errors in '{self.error_log_path}'{COLORS['ENDC']}")

        print(f"\n{COLORS['DIM']}Total folders in range: {self.total} | Skipped: {self.skipped} | Processed: {processed_count} | Failed: {len(failed_folders)}{COLORS['ENDC']}")

        return failed_folders

def main():
    parser = argparse.ArgumentParser(description='Run GAEL fitting in parallel')
    parser.add_argument('--start', type=int, default=1, help='Starting folder index (default: 1)')
    parser.add_argument('--end', type=int, required=True, help='Ending folder index')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers (default: 16)')
    parser.add_argument('--lines', type=int, default=2, help='Number of output lines to show per worker (default: 2)')
    parser.add_argument('--force', action='store_true', help='Force rerun all folders, even if already processed')

    args = parser.parse_args()

    runner = GAELRunner(args.start, args.end, args.workers, args.lines, args.force)
    failed = runner.run()

    sys.exit(0 if not failed else 1)

if __name__ == "__main__":
    main()