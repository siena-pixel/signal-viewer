#!/usr/bin/env python3
"""
Pretty test runner for Signal Viewer.

Usage:  python run_tests.py
        python run_tests.py tests/test_resampling.py   (single file)

Runs unittest discover, parses the verbose output, and prints a
colour-coded one-line-per-test report.

  OK      → green
  WARNING → yellow  (test passed but emitted app-level warnings)
  ERROR   → red
  FAIL    → red
"""

import os
import re
import subprocess
import sys

# ── ANSI codes ─────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def run_tests(extra_args=None):
    """Run tests and return the raw stderr output."""
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    if extra_args:
        cmd = [sys.executable, "-m", "unittest"] + extra_args + ["-v"]
    env = {**os.environ, "PYTHONPATH": "."}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stderr


def parse_results(raw):
    """
    Parse unittest verbose output into a list of (name, module_class, status).

    Handles multi-line entries where a docstring and/or warning logs appear
    between the test header and the "... ok" marker.
    """
    tests = []

    # Split into lines and walk them sequentially.
    # A test entry always starts with "test_xxx (module.Class)" and eventually
    # a line ending with "... ok" / "... FAIL" / "... ERROR" appears.
    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect the start of a test: "test_xxx (module.Class)..."
        start_m = re.match(r'^(test\S*)\s+\(([^)]+)\)', line)
        if not start_m:
            i += 1
            continue

        test_id = start_m.group(1)
        module_class = start_m.group(2)

        # Check if the result is on this same line
        end_m = re.search(r'\.\.\.\s+(ok|FAIL|ERROR|SKIP)\s*$', line)
        if end_m:
            status = end_m.group(1).upper()
            # Everything between "(Class) " and " ... ok" could be a docstring
            after_class = line[start_m.end():end_m.start()].strip()
            # strip leading newline-joined docstring text
            docstring = after_class.lstrip('. \n')
            has_warn = False
            i += 1
        else:
            # Multi-line: collect lines until we see "... ok/FAIL/ERROR"
            docstring = ""
            has_warn = False
            i += 1
            while i < len(lines):
                sub = lines[i]
                end_m = re.search(r'\.\.\.\s+(ok|FAIL|ERROR|SKIP)\s*$', sub)
                if end_m:
                    status = end_m.group(1).upper()
                    # Text before "..." on this line may be the docstring
                    pre = sub[:end_m.start()].strip()
                    if pre and not re.match(r'^(WARNING|ERROR|INFO):', pre):
                        docstring = pre
                    i += 1
                    break
                # Check for app warnings in between
                if re.match(r'^(WARNING|ERROR):signal_viewer', sub):
                    has_warn = True
                elif sub.strip() and not re.match(r'^(WARNING|ERROR|INFO):', sub):
                    # Likely a docstring line
                    docstring = sub.strip()
                i += 1
            else:
                status = "ERROR"
                i += 1

        # Choose display name
        if docstring:
            name = docstring.rstrip('.')
        else:
            name = test_id.replace('_', ' ')
            if name.startswith('test '):
                name = name[5:]

        if has_warn and status == "OK":
            status = "WARNING"

        tests.append((name, module_class, status))

    return tests


def print_report(tests, summary_line):
    total_ok = sum(1 for _, _, s in tests if s == "OK")
    total_warn = sum(1 for _, _, s in tests if s == "WARNING")
    total_err = sum(1 for _, _, s in tests if s in ("FAIL", "ERROR"))

    max_name = min(max((len(t[0]) for t in tests), default=40), 80)

    print()
    print(f"  {BOLD}Signal Viewer — Test Results{RESET}")
    print(f"  {DIM}{'─' * (max_name + 16)}{RESET}")
    print()
    print(f"  {GREEN}● {total_ok} passed{RESET}    "
          f"{YELLOW}● {total_warn} warnings{RESET}    "
          f"{RED}● {total_err} errors{RESET}")
    print()

    prev_class = ""
    for name, module_class, status in tests:
        cls = module_class.split('.')[-1] if '.' in module_class else module_class
        if cls != prev_class:
            if prev_class:
                print()
            label = cls.replace("Test", "").strip()
            label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)
            print(f"  {DIM}{label.upper()}{RESET}")
            prev_class = cls

        display = name[:max_name - 1] + "…" if len(name) > max_name else name
        dots = "·" * (max_name - len(display) + 4)

        if status == "OK":
            badge = f"{GREEN}OK{RESET}"
        elif status == "WARNING":
            badge = f"{YELLOW}WARNING{RESET}"
        else:
            badge = f"{RED}{status}{RESET}"

        print(f"  {display} {DIM}{dots}{RESET} {badge}")

    print()
    print(f"  {DIM}{'─' * (max_name + 16)}{RESET}")
    print(f"  {summary_line}")
    print()


def main():
    extra = sys.argv[1:] if len(sys.argv) > 1 else None
    raw = run_tests(extra)

    tests = parse_results(raw)

    sm = re.search(r'Ran \d+ tests? in .+', raw)
    summary = sm.group(0) if sm else f"Ran {len(tests)} tests"

    print_report(tests, summary)


if __name__ == "__main__":
    main()
