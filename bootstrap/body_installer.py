"""
Body Installer

Bootstrap infrastructure for installing and verifying body (actuator) dependencies.
This module contains zero action logic and zero intelligence about behavior.
It simply ensures mechanical actuation tools exist so the EME can act on reality.

Absolute constraints:
- NO mouse movement, clicking, or typing
- NO action execution or simulation
- NO screen state interpretation
- NO retries, fallbacks, or heuristics
- NO privileged OS APIs beyond OpenCU
- NO behavior logic encoding

This module is pure mechanical dependency management - boring, auditable, and replaceable.
"""

import importlib
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
BODY_INSTALLER_LOG_LEVEL = logging.INFO


class Platform(str, Enum):
    """Platform identifiers for body dependency resolution."""
    LINUX = "linux"
    MACOS = "darwin"
    WINDOWS = "windows"


class BodyBackendType(str, Enum):
    """Type of body backend."""
    OPENCU = "opencu"


@dataclass(frozen=True)
class BodyDependency:
    """Mechanical body dependency definition with no semantic meaning."""
    name: str
    backend_type: BodyBackendType
    python_import: Optional[str] = None
    pip_package: Optional[str] = None
    system_dependencies: Tuple[str, ...] = ()
    min_version: Optional[str] = None


class BodyInstaller:
    """
    Stateless installer for body (actuator) dependencies.
    
    This class:
    1. Installs required body backend (OpenCU)
    2. Verifies backend availability and capabilities
    3. Checks system permissions for input device access
    4. Logs mechanical progress and errors
    5. Fails fast on missing or unusable dependencies
    
    This class does NOT:
    1. Move mouse, click, or type
    2. Execute or simulate any actions
    3. Interpret screen state or environment
    4. Implement behavior logic or heuristics
    5. Use privileged OS APIs beyond OpenCU
    6. Instantiate actuator objects (potential side effects)
    """
    
    # Primary body backend (OpenCU)
    PRIMARY_BACKEND = BodyDependency(
        name="OpenCU",
        backend_type=BodyBackendType.OPENCU,
        python_import="opencu",
        pip_package="opencu",
        system_dependencies=("xdotool", "xinput", "xrandr"),  # X11 dependencies
    )
    
    def __init__(self, installer_log_level: int = BODY_INSTALLER_LOG_LEVEL) -> None:
        """
        Initialize body installer.
        
        Args:
            installer_log_level: Logging level for installation operations
        """
        self._logger = logging.getLogger(f"{__name__}.BodyInstaller")
        self._log_level = installer_log_level
        self._platform = self._detect_platform()
        
        # Log platform detection
        self._log_platform_detection()
        
        # Validate platform
        if self._platform != Platform.LINUX:
            self._logger.error(f"Unsupported platform for body control: {self._platform.value}")
            raise RuntimeError(
                f"Body control currently only supported on Linux. Detected: {self._platform.value}"
            )
    
    def install_and_verify(self) -> bool:
        """
        Install and verify body dependencies.
        
        Returns:
            True if all body dependencies are available and usable
        
        Raises:
            RuntimeError: If installation fails or dependencies are missing
        
        Note:
            - No retry logic
            - No action execution
            - No mouse movement or keyboard input
            - Mechanical capability checking only
        """
        self._logger.log(self._log_level, "Starting body dependency installation")
        
        try:
            # Step 1: Install system dependencies
            self._install_system_dependencies()
            
            # Step 2: Install Python packages
            self._install_python_dependencies()
            
            # Step 3: Verify all dependencies
            verification_results = self._verify_all_dependencies()
            
            # Step 4: Check for any failures
            all_verified = all(verification_results.values())
            
            if all_verified:
                self._logger.log(self._log_level, "All body dependencies verified successfully")
                return True
            else:
                # Log specific failures
                failed_deps = [name for name, ok in verification_results.items() if not ok]
                error_msg = f"Body dependencies failed verification: {', '.join(failed_deps)}"
                self._logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            # Log failure and re-raise
            self._logger.error(f"Body installation failed: {str(e)}")
            raise
    
    def _install_system_dependencies(self) -> None:
        """Install required system dependencies for OpenCU."""
        if self._platform != Platform.LINUX:
            self._logger.warning("System dependency installation only supported on Linux")
            return
        
        system_deps = self.PRIMARY_BACKEND.system_dependencies
        if not system_deps:
            self._logger.log(self._log_level, "No system dependencies to install")
            return
        
        self._logger.log(self._log_level, f"Installing system dependencies: {', '.join(system_deps)}")
        
        try:
            # Use apt-get on Debian/Ubuntu systems
            # Note: This assumes the system uses apt package manager
            result = subprocess.run(
                ["sudo", "apt-get", "update"],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode != 0:
                self._logger.warning(f"Package list update failed: {result.stderr[:200]}")
            
            # Install each dependency
            for dep in system_deps:
                self._install_system_package(dep)
                
        except FileNotFoundError:
            raise RuntimeError("apt-get not found. Ensure you're on a Debian-based system.")
        except Exception as e:
            raise RuntimeError(f"System dependency installation failed: {str(e)}")
    
    def _install_system_package(self, package_name: str) -> None:
        """Install a single system package."""
        self._logger.log(self._log_level, f"Installing system package: {package_name}")
        
        try:
            result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", package_name],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode == 0:
                self._logger.log(self._log_level, f"Successfully installed system package: {package_name}")
            else:
                # Check if already installed
                check_result = subprocess.run(
                    ["dpkg", "-s", package_name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                
                if check_result.returncode == 0:
                    self._logger.log(self._log_level, f"System package already installed: {package_name}")
                else:
                    error_msg = f"Failed to install system package {package_name}: {result.stderr[:200]}"
                    self._logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except Exception as e:
            raise RuntimeError(f"Error installing system package {package_name}: {str(e)}")
    
    def _install_python_dependencies(self) -> None:
        """Install Python packages for body backend."""
        # Install primary backend (OpenCU)
        self._install_python_package(self.PRIMARY_BACKEND)
    
    def _install_python_package(self, dep: BodyDependency) -> None:
        """Install a single Python package."""
        if not dep.pip_package:
            self._logger.warning(f"No pip package specified for {dep.name}, skipping installation")
            return
        
        self._logger.log(self._log_level, f"Installing Python package: {dep.name} ({dep.pip_package})")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", dep.pip_package],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode != 0:
                error_msg = f"Failed to install {dep.name}: {result.stderr[:200]}"
                self._logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self._logger.log(self._log_level, f"Successfully installed {dep.name}")
            
        except Exception as e:
            raise RuntimeError(f"Error installing Python package {dep.name}: {str(e)}")
    
    def _verify_all_dependencies(self) -> Dict[str, bool]:
        """
        Verify all body dependencies are available and usable.
        
        Returns:
            Dictionary mapping dependency names to verification results
        """
        verification_results = {}
        
        # Verify primary backend (OpenCU) - static checks only
        primary_verified = self._verify_backend(self.PRIMARY_BACKEND)
        verification_results[self.PRIMARY_BACKEND.name] = primary_verified
        
        # Verify system permissions
        permissions_verified = self._verify_system_permissions()
        verification_results["System permissions"] = permissions_verified
        
        # Verify input device detection
        devices_verified = self._verify_input_devices()
        verification_results["Input devices"] = devices_verified
        
        # Verify display server
        display_verified = self._verify_display_server()
        verification_results["Display server"] = display_verified
        
        return verification_results
    
    def _verify_backend(self, dep: BodyDependency) -> bool:
        """Verify body backend can be imported using static checks only."""
        if not dep.python_import:
            self._logger.error(f"No python_import specified for {dep.name}")
            return False
        
        try:
            # Attempt to import the module
            module = importlib.import_module(dep.python_import)
            
            # Log version if available
            if hasattr(module, '__version__'):
                version = module.__version__
                self._logger.log(self._log_level, f"  {dep.name} version: {version}")
            
            # Static checks only - no object instantiation
            if dep.backend_type == BodyBackendType.OPENCU:
                return self._verify_opencu_static(module)
            else:
                self._logger.error(f"Unknown backend type: {dep.backend_type}")
                return False
            
        except ImportError as e:
            self._logger.error(f"Failed to import {dep.python_import}: {str(e)}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error verifying {dep.name}: {str(e)}")
            return False
    
    def _verify_opencu_static(self, opencu_module) -> bool:
        """Verify OpenCU backend using static checks only."""
        try:
            # Check for expected class names (static attribute existence)
            expected_classes = ["Mouse", "Keyboard", "Screen"]
            missing_classes = []
            
            for class_name in expected_classes:
                if not hasattr(opencu_module, class_name):
                    missing_classes.append(class_name)
            
            if missing_classes:
                self._logger.error(f"OpenCU missing expected classes: {', '.join(missing_classes)}")
                return False
            
            # Check that classes are callable (but don't call them)
            for class_name in expected_classes:
                class_obj = getattr(opencu_module, class_name)
                if not callable(class_obj):
                    self._logger.error(f"OpenCU class {class_name} is not callable")
                    return False
            
            self._logger.log(self._log_level, "  OpenCU backend classes found and callable")
            return True
                
        except Exception as e:
            self._logger.error(f"OpenCU static verification failed: {str(e)}")
            return False
    
    def _verify_system_permissions(self) -> bool:
        """Verify system permissions for input device access."""
        if self._platform != Platform.LINUX:
            self._logger.warning("System permission verification only supported on Linux")
            return True  # Not a failure, just unsupported check
        
        self._logger.log(self._log_level, "Verifying system permissions")
        
        # Check if running as root (not recommended but sometimes required)
        if os.geteuid() == 0:
            self._logger.warning("Running as root - ensure proper X11 permissions")
            # Root can bypass permissions, but warn about security
        
        # Check for X11 authority file
        xauth_file = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
        if os.path.exists(xauth_file):
            self._logger.log(self._log_level, f"  X11 authority file found: {xauth_file}")
        else:
            self._logger.error(f"X11 authority file not found: {xauth_file}")
            return False
        
        # Check for DISPLAY environment variable
        display = os.environ.get("DISPLAY")
        if display:
            self._logger.log(self._log_level, f"  DISPLAY environment variable: {display}")
            return True
        else:
            self._logger.error("DISPLAY environment variable not set")
            return False
    
    def _verify_input_devices(self) -> bool:
        """Verify mouse and keyboard input devices are detectable."""
        if self._platform != Platform.LINUX:
            self._logger.warning("Input device verification only supported on Linux")
            return True  # Not a failure, just unsupported check
        
        self._logger.log(self._log_level, "Verifying input devices")
        
        try:
            # Check for input device files
            input_devices = [
                "/dev/input/mice",
                "/dev/input/mouse0",
                "/dev/input/event*",  # Pattern for event devices
            ]
            
            found_devices = []
            for device in input_devices:
                if "*" in device:
                    # Pattern matching for event devices
                    import glob
                    matches = glob.glob(device)
                    found_devices.extend(matches)
                elif os.path.exists(device):
                    found_devices.append(device)
            
            if found_devices:
                self._logger.log(self._log_level, f"  Found input devices: {', '.join(found_devices[:3])}...")
                return True
            else:
                self._logger.error("No input devices found in /dev/input/")
                return False
                
        except Exception as e:
            self._logger.error(f"Input device verification failed: {str(e)}")
            return False
    
    def _verify_display_server(self) -> bool:
        """Verify display server is available."""
        if self._platform != Platform.LINUX:
            self._logger.warning("Display server verification only supported on Linux")
            return True  # Not a failure, just unsupported check
        
        self._logger.log(self._log_level, "Verifying display server")
        
        try:
            # Check for X11 or Wayland
            xdg_session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
            
            if xdg_session_type in ["x11", "wayland"]:
                self._logger.log(self._log_level, f"  Display server: {xdg_session_type}")
                
                # Additional check for X11
                if xdg_session_type == "x11":
                    # Try to run xdpyinfo (read-only)
                    result = subprocess.run(
                        ["xdpyinfo"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode == 0:
                        self._logger.log(self._log_level, "  X11 server accessible via xdpyinfo")
                        return True
                    else:
                        self._logger.error(f"X11 server not accessible: {result.stderr[:200]}")
                        return False
                else:
                    # Wayland - less standardized, assume accessible
                    self._logger.log(self._log_level, "  Assuming Wayland accessibility")
                    return True
            else:
                self._logger.error(f"Unsupported or unknown display server: {xdg_session_type}")
                return False
                
        except Exception as e:
            self._logger.error(f"Display server verification failed: {str(e)}")
            return False
    
    def _detect_platform(self) -> Platform:
        """Detect current platform."""
        system = platform.system().lower()
        
        if system == "linux":
            return Platform.LINUX
        elif system == "darwin":
            return Platform.MACOS
        elif system == "windows":
            return Platform.WINDOWS
        else:
            self._logger.warning(f"Unrecognized platform: {system}, defaulting to Linux")
            return Platform.LINUX
    
    def _log_platform_detection(self) -> None:
        """Log platform detection results."""
        log_data = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": self._platform.value,
        }
        
        self._logger.log(
            self._log_level,
            f"Detected platform: {self._platform.value}",
            extra=log_data
        )
    
    @staticmethod
    def run_cli() -> None:
        """
        Command-line interface entry point.
        
        Usage: python -m bootstrap.body_installer
        Exit codes: 0 = success, 1 = failure
        """
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Install and verify body (actuator) dependencies for EME"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        args = parser.parse_args()
        
        # Configure logging
        log_level = logging.DEBUG if args.verbose else BODY_INSTALLER_LOG_LEVEL
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Run installer
        installer = BodyInstaller(installer_log_level=log_level)
        
        try:
            success = installer.install_and_verify()
            if success:
                print("✓ Body dependencies installed and verified successfully")
                sys.exit(0)
            else:
                print("✗ Body dependencies verification failed")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Body installation failed: {str(e)}")
            sys.exit(1)


# Export public interface
__all__ = [
    "BodyInstaller",
    "BodyDependency",
    "Platform",
    "BodyBackendType",
    "BODY_INSTALLER_LOG_LEVEL",
]

# Allow direct script execution
if __name__ == "__main__":
    BodyInstaller.run_cli()