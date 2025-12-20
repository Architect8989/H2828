"""
shell.py - The first grounded probe of the Environment Mastery Engine (EME).

Prime Constraint:

> This module must never return raw strings or opaque outputs.
It must return a Delta or raise an exception.



This module executes deterministic shell commands and converts their results
into structured, verifiable Deltas. It only observes and reports change.

Epistemic Rule:

> A probe in EME must perform one experiment, report exactly what it observed,
and refuse to help if reality is unclear.



Quality Bar: This module should look like a hardware driver, a kernel probe, a truth sensor.
"""

import subprocess
import sys
import os
from typing import Dict, Tuple

Import only Delta - probes must not reason about Delta validity

from core.delta import Delta

def _run_command(command: str) -> Tuple[str, str, int]:
"""
Execute a shell command deterministically.

Args:  
    command: Shell command to execute  
      
Returns:  
    Tuple of (stdout, stderr, return_code)  
      
Raises:  
    subprocess.CalledProcessError: If command returns non-zero  
    FileNotFoundError: If command does not exist  
    OSError: If command execution fails  
"""  
# Use subprocess.run with explicit parameters for deterministic execution  
result = subprocess.run(  
    command,  
    shell=True,  
    capture_output=True,  
    text=True,  
    timeout=5,  # Prevent hanging  
    env={  # Minimal, controlled environment  
        'PATH': '/bin:/usr/bin:/sbin:/usr/sbin',  
        'LANG': 'C',  # Ensure consistent output format  
    }  
)  
  
return result.stdout.strip(), result.stderr.strip(), result.returncode

def _parse_uname_a(uname_output: str) -> Dict[str, str]:
"""
Parse uname -a output into structured facts.

Expected format (Linux):  
Linux hostname 5.15.0-generic #1 SMP ... x86_64 GNU/Linux  
  
Expected format (macOS):  
Darwin hostname 21.6.0 ... x86_64 Darwin  
  
Args:  
    uname_output: Output from `uname -a` command  
      
Returns:  
    Dictionary with parsed facts  
      
Raises:  
    ValueError: If output cannot be parsed  
"""  
if not uname_output:  
    raise ValueError("Empty output from uname -a")  
  
parts = uname_output.split()  
  
# Minimal parsing - extract known positions  
if len(parts) < 3:  
    raise ValueError(f"Invalid uname -a output format: {uname_output}")  
  
# Extract OS name from first token  
os_name_map = {  
    'Linux': 'Linux',  
    'Darwin': 'macOS',  
    'FreeBSD': 'FreeBSD',  
    'OpenBSD': 'OpenBSD',  
    'NetBSD': 'NetBSD',  
    'SunOS': 'Solaris',  
    'AIX': 'AIX',  
    'HP-UX': 'HP-UX'  
}  
  
raw_os = parts[0]  
os_name = os_name_map.get(raw_os, raw_os)  # Use mapping or raw if unknown  
  
# Kernel version is the third token  
kernel_version = parts[2]  
  
# Architecture is typically the last or second-to-last token  
# Look for common architecture patterns  
arch_keywords = {'x86_64', 'amd64', 'i386', 'i686', 'arm64', 'aarch64', 'arm', 'ppc', 'ppc64', 's390x'}  
  
arch = ""  
for part in parts:  
    if part in arch_keywords:  
        arch = part  
        break  
  
if not arch:  
    # If no match, use empty string to represent absence  
    # Delta will handle emptiness, not probe  
    arch = ""  
  
return {  
    "host.os": os_name,  
    "host.kernel": kernel_version,  
    "host.arch": arch  
}

def probe_system_identity() -> Delta:
"""
Probe system identity and return structured Delta.

Runs ONE deterministic command to discover:  
  - OS name  
  - Kernel version  
  - Architecture  
  
Returns:  
    Delta with structured changes like:  
    {  
      "host.os": "...",  
      "host.kernel": "...",  
      "host.arch": "..."  
    }  
      
Raises:  
    subprocess.CalledProcessError: If command fails  
    FileNotFoundError: If command does not exist  
    ValueError: If parsing yields ambiguous output  
    Any exception from Delta constructor if Delta is invalid  
      
Note:  
    Returns exactly what was observed. Emptiness is Delta's responsibility.  
    If parsing yields no new facts, returns an EMPTY Delta.  
    LifeLoop will enforce the crash on repeated empty deltas.  
"""  
# ONE action: uname -a  
# ONE observation: parse output exactly  
# ONE Delta: construct and return  
# OR crash: propagate any exception  
  
stdout, stderr, returncode = _run_command("uname -a")  
  
if returncode != 0:  
    raise subprocess.CalledProcessError(  
        returncode, "uname -a", stdout, stderr  
    )  
  
facts = _parse_uname_a(stdout)  
  
# DO NOT filter empty values - pass exact observation  
# Delta and LifeLoop handle emptiness, not probe  
return Delta(facts)

Minimal self-test when run directly

if name == "main":
"""
Self-test: Run probe and verify Delta structure.

Expected output:  
  - A valid Delta with host.os, host.kernel, host.arch (potentially empty)  
  - Or an exception if probe fails  
"""  
print("Testing shell.py probe...")  
  
try:  
    delta = probe_system_identity()  
      
    print(f"Probe succeeded. Delta: {delta}")  
    print(f"Delta is_valid(): {delta.is_valid()}")  
    print(f"Delta is_empty(): {delta.is_empty()}")  
      
    print("\nStructured facts (exactly as observed):")  
    for path, value in delta.changes.items():  
        print(f"  {path}: {repr(value)}")  
      
    if delta.is_valid():  
        print("\n✓ Probe successfully created structured Delta")  
        print("✓ Module obeys ONE action → ONE observation → ONE Delta rule")  
        print("✓ Module ready for LifeLoop integration")  
        sys.exit(0)  
    else:  
        print("\n✗ Probe created invalid Delta")  
        sys.exit(1)  
          
except Exception as e:  
    print(f"\n✗ Probe crashed with exception: {repr(e)}")  
    print("✓ Module correctly crashes on failure")  
    print("✓ Module refuses to help when reality is unclear")  
    sys.exit(1)
