#!/usr/bin/env python3
"""
Run GAEL fitting on multiple directories in parallel with live progress display.
Skips already processed folders by checking for fit_results/{folder_nr}/fit_parameters.csv
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
from datetime import datetime
import shutil
import math

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
    def __init__(self, start_idx, end_idx, max_workers=16, lines_to_show=2, force_rerun=False):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_workers = max_workers
        self.lines_to_show = lines_to_show
        self.force_rerun = force_rerun
        self.lock = threading.Lock()
        self.completed = 0
        self.skipped = 0
        self.total = end_idx - start_idx + 1
        self.folders_to_process = []
        self.worker_status = {}
        self.worker_output = {}
        
        # Filter folders that need processing
        self._filter_folders_to_process()
        
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
        """Run GAEL for a single folder and capture output."""
        folder_name = f"{folder_idx:04d}"
        input_file = f"{folder_name}/input.json"
        
        # Double-check if already processed (in case of race conditions)
        if not self.force_rerun and self.is_already_processed(folder_idx):
            with self.lock:
                self.worker_status[folder_idx] = "✓ Skipped (already done)"
                self.completed += 1
            return folder_idx, True
        
        # Initialize worker status
        with self.lock:
            self.worker_status[folder_idx] = "Starting"
            self.worker_output[folder_idx] = deque(maxlen=self.lines_to_show)
        
        try:
            # Run GAEL command with unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # In case GAEL is a Python script
            
            cmd = ["GAEL", "--fit", input_file, "--threads", "1"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=0,  # Unbuffered
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
                line = line.rstrip()
                if line:
                    with self.lock:
                        self.worker_output[folder_idx].append(line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            with self.lock:
                if return_code == 0:
                    self.worker_status[folder_idx] = "✓ Complete"
                    self.completed += 1
                else:
                    self.worker_status[folder_idx] = f"✗ Failed (code {return_code})"
                    
            return folder_idx, return_code == 0
            
        except Exception as e:
            with self.lock:
                self.worker_status[folder_idx] = f"✗ Error: {str(e)}"
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
        """Display live progress in terminal."""
        total_to_process = len(self.folders_to_process)
        
        while self.completed < total_to_process:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get terminal size
            terminal_width = shutil.get_terminal_size().columns
            terminal_height = shutil.get_terminal_size().lines
            
            # Header
            print(f"{COLORS['HEADER']}{'='*min(terminal_width, 100)}{COLORS['ENDC']}")
            print(f"{COLORS['BOLD']}GAEL Parallel Fitting Progress{COLORS['ENDC']}".center(min(terminal_width, 100)))
            print(f"{COLORS['HEADER']}{'='*min(terminal_width, 100)}{COLORS['ENDC']}")
            
            # Overall progress
            if total_to_process > 0:
                progress = self.completed / total_to_process
                bar_length = min(50, terminal_width - 40)
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\n{COLORS['OKGREEN']}Progress:{COLORS['ENDC']} [{bar}] {progress*100:.1f}% ({self.completed}/{total_to_process})")
            else:
                print(f"\n{COLORS['OKGREEN']}All folders already processed!{COLORS['ENDC']}")
            
            # Show skip information
            if self.skipped > 0:
                print(f"{COLORS['OKBLUE']}Skipped (already done):{COLORS['ENDC']} {self.skipped}")
            
            with self.lock:
                active_workers = [idx for idx, status in self.worker_status.items() if status == "Running"]
                completed_count = sum(1 for s in self.worker_status.values() if "Complete" in s)
                failed_count = sum(1 for s in self.worker_status.values() if "Failed" in s or "Error" in s)
                skipped_count = sum(1 for s in self.worker_status.values() if "Skipped" in s)
            
            print(f"{COLORS['OKCYAN']}Active:{COLORS['ENDC']} {len(active_workers)} | ", end="")
            print(f"{COLORS['OKGREEN']}Complete:{COLORS['ENDC']} {completed_count} | ", end="")
            print(f"{COLORS['FAIL']}Failed:{COLORS['ENDC']} {failed_count} | ", end="")
            if skipped_count > 0:
                print(f"{COLORS['OKBLUE']}Skipped:{COLORS['ENDC']} {skipped_count} | ", end="")
            print(f"{COLORS['DIM']}Time:{COLORS['ENDC']} {datetime.now().strftime('%H:%M:%S')}")
            
            # Calculate grid dimensions
            n_cols = 4  # Default number of columns
            box_width = (terminal_width - (n_cols + 1) * 2) // n_cols  # Account for spacing
            box_width = max(box_width, 30)  # Minimum box width
            
            # Worker grid
            print(f"\n{COLORS['BOLD']}Active Workers:{COLORS['ENDC']}")
            print("─" * min(terminal_width, 100))
            
            with self.lock:
                # Get all workers to display (prioritize active ones)
                workers_to_show = []
                
                # First add all running workers
                for idx in sorted(self.worker_status.keys()):
                    if self.worker_status[idx] == "Running":
                        workers_to_show.append(idx)
                
                # Then add recent completed/failed if space allows
                remaining_slots = self.max_workers - len(workers_to_show)
                if remaining_slots > 0:
                    other_workers = []
                    for idx in sorted(self.worker_status.keys(), reverse=True):
                        if self.worker_status[idx] != "Running":
                            other_workers.append(idx)
                    workers_to_show.extend(other_workers[:remaining_slots])
                
                # Display in grid
                for row_start in range(0, len(workers_to_show), n_cols):
                    row_workers = workers_to_show[row_start:row_start + n_cols]
                    
                    # Get all lines for this row of workers
                    worker_boxes = []
                    max_lines = 0
                    for idx in row_workers:
                        box_lines = self.format_worker_box(idx, box_width)
                        worker_boxes.append(box_lines)
                        max_lines = max(max_lines, len(box_lines))
                    
                    # Print each line of the row
                    for line_idx in range(max_lines):
                        for col_idx, box_lines in enumerate(worker_boxes):
                            if line_idx < len(box_lines):
                                line = box_lines[line_idx]
                                # Pad to box width
                                line_display = line + " " * (box_width - len(self.strip_ansi(line)))
                            else:
                                line_display = " " * box_width
                            
                            print(f"  {line_display}", end="")
                            if col_idx < len(worker_boxes) - 1:
                                print("  ", end="")  # Space between columns
                        print()  # New line
                    
                    # Add separator between rows
                    if row_start + n_cols < len(workers_to_show):
                        print()
            
            # Summary statistics at bottom
            if completed_count > 0 or failed_count > 0:
                print(f"\n{COLORS['DIM']}{'─' * min(terminal_width, 100)}{COLORS['ENDC']}")
                if failed_count > 0:
                    with self.lock:
                        failed_folders = [f"{idx:04d}" for idx, status in self.worker_status.items() 
                                        if "Failed" in status or "Error" in status]
                        failed_str = ", ".join(failed_folders[:10])
                        if len(failed_folders) > 10:
                            failed_str += f", ... ({len(failed_folders)-10} more)"
                    print(f"{COLORS['FAIL']}Failed folders:{COLORS['ENDC']} {failed_str}")
            
            time.sleep(0.1)  # Reduced from 0.5 to 0.1 for more responsive updates
    
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
                import textwrap
                wrapped = textwrap.wrap(failed_str, width=terminal_width - 10)
                for i, line in enumerate(wrapped):
                    if i == 0:
                        print(f"   {line}")
                    else:
                        print(f"   {line}")
            else:
                print(f"   {failed_str}")
        
        # Save failed folders to file if any
        if failed_folders:
            with open('failed_folders.txt', 'w') as f:
                for idx in sorted(failed_folders):
                    f.write(f"{idx:04d}\n")
            print(f"\n{COLORS['WARNING']}Failed folder list saved to 'failed_folders.txt'{COLORS['ENDC']}")
        
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