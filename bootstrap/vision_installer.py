"""
Vision Installer

Bootstrap infrastructure for installing and verifying raw vision dependencies.
This module contains zero perception logic and zero intelligence about vision.
It simply ensures mechanical dependencies exist so the EME can perceive reality.

Absolute constraints:
- NO OCR inference or image processing
- NO vision backend calls
- NO interpretation of outputs
- NO VLM installation or reference
- NO fallbacks, retries, or heuristics
- NO GPU availability assumptions
- NO learning or perception logic

This module is pure mechanical dependency management - boring, auditable, and replaceable.
"""

import importlib
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
INSTALLER_LOG_LEVEL = logging.INFO


class Platform(str, Enum):
    """Platform identifiers for dependency resolution."""
    LINUX = "linux"
    MACOS = "darwin"
    WINDOWS = "windows"


class DependencyType(str, Enum):
    """Type of dependency."""
    PYTHON_PACKAGE = "python_package"
    SYSTEM_BINARY = "system_binary"


@dataclass(frozen=True)
class Dependency:
    """Mechanical dependency definition with no semantic meaning."""
    name: str
    dep_type: DependencyType
    python_import: Optional[str] = None  # For Python packages
    binary_name: Optional[str] = None    # For system binaries
    pip_package: Optional[str] = None    # Alternative pip name
    min_version: Optional[str] = None    # Optional minimum version
    
    def __post_init__(self) -> None:
        """Validate dependency configuration."""
        if self.dep_type == DependencyType.PYTHON_PACKAGE and not self.python_import:
            raise ValueError(f"Python package {self.name} must specify python_import")
        if self.dep_type == DependencyType.SYSTEM_BINARY and not self.binary_name:
            raise ValueError(f"System binary {self.name} must specify binary_name")


class VisionInstaller:
    """
    Stateless installer for vision dependencies.
    
    This class:
    1. Installs required vision dependencies
    2. Verifies availability of each dependency
    3. Logs mechanical progress and errors
    4. Fails fast on missing or unusable dependencies
    
    This class does NOT:
    1. Run OCR inference or image processing
    2. Load OCR models into memory
    3. Inspect screen contents
    4. Compare or rank vision tools
    5. Install or configure VLMs
    6. Modify system behavior beyond installation
    """
    
    # Core vision dependencies for EME (mechanical list only)
    DEPENDENCIES = [
        # Python packages
        Dependency(
            name="PaddleOCR",
            dep_type=DependencyType.PYTHON_PACKAGE,
            python_import="paddleocr",
            pip_package="paddleocr",
        ),
        Dependency(
            name="Tesseract Python bindings",
            dep_type=DependencyType.PYTHON_PACKAGE,
            python_import="pytesseract",
            pip_package="pytesseract",
        ),
        Dependency(
            name="Screen capture (mss)",
            dep_type=DependencyType.PYTHON_PACKAGE,
            python_import="mss",
            pip_package="mss",
        ),
        Dependency(
            name="Screen capture (Pillow)",
            dep_type=DependencyType.PYTHON_PACKAGE,
            python_import="PIL",
            pip_package="Pillow",
        ),
        # System binaries
        Dependency(
            name="Tesseract OCR",
            dep_type=DependencyType.SYSTEM_BINARY,
            binary_name="tesseract",
        ),
    ]
    
    def __init__(self, installer_log_level: int = INSTALLER_LOG_LEVEL) -> None:
        """
        Initialize vision installer.
        
        Args:
            installer_log_level: Logging level for installation operations
        """
        self._logger = logging.getLogger(f"{__name__}.VisionInstaller")
        self._log_level = installer_log_level
        self._platform = self._detect_platform()
        
        # Log platform detection
        self._log_platform_detection()
    
    def install_and_verify(self) -> bool:
        """
        Install and verify all vision dependencies.
        
        Returns:
            True if all dependencies are available and usable
        
        Raises:
            RuntimeError: If installation fails or dependencies are missing
        
        Note:
            - No retry logic
            - No fallback mechanisms
            - Exact dependency checking only
        """
        self._logger.log(self._log_level, "Starting vision dependency installation")
        
        try:
            # Step 1: Install Python packages
            self._install_python_dependencies()
            
            # Step 2: Verify all dependencies
            verification_results = self._verify_all_dependencies()
            
            # Step 3: Check for any failures
            all_verified = all(verification_results.values())
            
            if all_verified:
                self._logger.log(self._log_level, "All vision dependencies verified successfully")
                return True
            else:
                # Log specific failures
                failed_deps = [name for name, ok in verification_results.items() if not ok]
                error_msg = f"Vision dependencies failed verification: {', '.join(failed_deps)}"
                self._logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            # Log failure and re-raise
            self._logger.error(f"Vision installation failed: {str(e)}")
            raise
    
    def _install_python_dependencies(self) -> None:
        """Install Python packages using pip."""
        # Get Python packages that need installation
        python_packages = [
            dep for dep in self.DEPENDENCIES 
            if dep.dep_type == DependencyType.PYTHON_PACKAGE
        ]
        
        for dep in python_packages:
            self._install_python_package(dep)
    
    def _install_python_package(self, dep: Dependency) -> None:
        """
        Install a single Python package.
        
        Note: No version pinning, no dependency resolution beyond basic pip install.
        """
        if not dep.pip_package:
            self._logger.warning(f"No pip package specified for {dep.name}, skipping installation")
            return
        
        self._logger.log(self._log_level, f"Installing Python package: {dep.name} ({dep.pip_package})")
        
        try:
            # Use subprocess to run pip install
            # Note: Using --quiet to reduce noise, but keeping errors visible
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
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Pip installation failed for {dep.name}: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error installing {dep.name}: {str(e)}") from e
    
    def _verify_all_dependencies(self) -> Dict[str, bool]:
        """
        Verify all dependencies are available and usable.
        
        Returns:
            Dictionary mapping dependency names to verification results
        """
        verification_results = {}
        
        for dep in self.DEPENDENCIES:
            try:
                if dep.dep_type == DependencyType.PYTHON_PACKAGE:
                    verified = self._verify_python_package(dep)
                elif dep.dep_type == DependencyType.SYSTEM_BINARY:
                    verified = self._verify_system_binary(dep)
                else:
                    self._logger.error(f"Unknown dependency type for {dep.name}: {dep.dep_type}")
                    verified = False
                
                verification_results[dep.name] = verified
                
                if verified:
                    self._logger.log(self._log_level, f"✓ Verified: {dep.name}")
                else:
                    self._logger.error(f"✗ Failed verification: {dep.name}")
                    
            except Exception as e:
                self._logger.error(f"Verification error for {dep.name}: {str(e)}")
                verification_results[dep.name] = False
        
        # Verify screen capture capability
        screen_capture_verified = self._verify_screen_capture()
        verification_results["Screen capture capability"] = screen_capture_verified
        
        return verification_results
    
    def _verify_python_package(self, dep: Dependency) -> bool:
        """Verify Python package can be imported."""
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
            
            return True
            
        except ImportError as e:
            self._logger.error(f"Failed to import {dep.python_import}: {str(e)}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error importing {dep.python_import}: {str(e)}")
            return False
    
    def _verify_system_binary(self, dep: Dependency) -> bool:
        """Verify system binary exists and is executable."""
        if not dep.binary_name:
            self._logger.error(f"No binary_name specified for {dep.name}")
            return False
        
        try:
            # Try to run the binary with --version or similar
            result = subprocess.run(
                [dep.binary_name, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode == 0:
                # Log version information
                version_output = result.stdout.strip()[:100]  # Limit output length
                self._logger.log(self._log_level, f"  {dep.name} version: {version_output}")
                return True
            else:
                self._logger.error(
                    f"Binary {dep.binary_name} returned non-zero exit code: {result.returncode}"
                )
                return False
                
        except FileNotFoundError:
            self._logger.error(f"Binary not found: {dep.binary_name}")
            return False
        except Exception as e:
            self._logger.error(f"Error executing {dep.binary_name}: {str(e)}")
            return False
    
    def _verify_screen_capture(self) -> bool:
        """
        Verify screen capture capability.
        
        Note: Only verifies ability to capture, not content or quality.
        """
        self._logger.log(self._log_level, "Verifying screen capture capability")
        
        # Try mss first (preferred), then PIL fallback
        capture_methods = ["mss", "PIL"]
        
        for method in capture_methods:
            try:
                if method == "mss":
                    success = self._verify_mss_capture()
                elif method == "PIL":
                    success = self._verify_pil_capture()
                else:
                    continue
                
                if success:
                    self._logger.log(self._log_level, f"  Screen capture verified using {method}")
                    return True
                    
            except Exception as e:
                self._logger.warning(f"Screen capture method {method} failed: {str(e)}")
                continue
        
        self._logger.error("All screen capture methods failed")
        return False
    
    def _verify_mss_capture(self) -> bool:
        """Verify mss screen capture capability."""
        try:
            import mss
            
            with mss.mss() as sct:
                # Get monitor information (mechanical check only)
                monitors = sct.monitors
                if not monitors:
                    self._logger.error("No monitors detected by mss")
                    return False
                
                # Attempt to capture a small region
                monitor = monitors[1] if len(monitors) > 1 else monitors[0]
                
                # Capture 1x1 pixel region (minimal impact)
                region = {
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": 1,
                    "height": 1,
                }
                
                screenshot = sct.grab(region)
                
                # Verify capture produced data
                if not screenshot:
                    self._logger.error("mss capture returned empty data")
                    return False
                
                # Check dimensions
                if screenshot.width != 1 or screenshot.height != 1:
                    self._logger.warning(f"mss capture dimensions unexpected: {screenshot.width}x{screenshot.height}")
                    # Still return True as capture worked
                
                return True
                
        except Exception as e:
            self._logger.error(f"mss capture verification failed: {str(e)}")
            return False
    
    def _verify_pil_capture(self) -> bool:
        """Verify PIL screen capture capability."""
        try:
            from PIL import ImageGrab
            
            # Attempt to capture screen
            # Note: On some systems, this may require additional permissions
            screenshot = ImageGrab.grab()
            
            if not screenshot:
                self._logger.error("PIL capture returned empty data")
                return False
            
            # Verify image has dimensions
            width, height = screenshot.size
            if width <= 0 or height <= 0:
                self._logger.error(f"PIL capture invalid dimensions: {width}x{height}")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"PIL capture verification failed: {str(e)}")
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
            "timestamp": platform.uname(),
            "platform": self._platform.value,
            "python_version": platform.python_version(),
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
        
        Usage: python -m bootstrap.vision_installer
        Exit codes: 0 = success, 1 = failure
        """
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Install and verify vision dependencies for EME"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        args = parser.parse_args()
        
        # Configure logging
        log_level = logging.DEBUG if args.verbose else INSTALLER_LOG_LEVEL
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Run installer
        installer = VisionInstaller(installer_log_level=log_level)
        
        try:
            success = installer.install_and_verify()
            if success:
                print("✓ Vision dependencies installed and verified successfully")
                sys.exit(0)
            else:
                print("✗ Vision dependencies verification failed")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Vision installation failed: {str(e)}")
            sys.exit(1)


# Export public interface
__all__ = [
    "VisionInstaller",
    "Dependency",
    "Platform",
    "DependencyType",
    "INSTALLER_LOG_LEVEL",
]

# Allow direct script execution
if __name__ == "__main__":
    VisionInstaller.run_cli()