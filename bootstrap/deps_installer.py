"""
Dependency Installer

Bootstrap infrastructure for installing and verifying networking and HTTP dependencies.
This module contains zero intelligence, zero reasoning, and zero LLM interaction.
It simply ensures mechanical connectivity tools exist so the brain can communicate externally.

Absolute constraints:
- NO LLM calls or inference
- NO prompt construction or sending
- NO response interpretation or parsing
- NO retries, backoff, or heuristics
- NO caching of any kind
- NO environment semantics inference
- NO intelligence or reasoning logic

This module is pure mechanical dependency management - boring, auditable, and replaceable.
"""

import asyncio
import importlib
import logging
import os
import platform
import socket
import ssl
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
DEPS_INSTALLER_LOG_LEVEL = logging.INFO


class Platform(str, Enum):
    """Platform identifiers for dependency resolution."""
    LINUX = "linux"
    MACOS = "darwin"
    WINDOWS = "windows"


class DependencyType(str, Enum):
    """Type of dependency."""
    PYTHON_PACKAGE = "python_package"
    SYSTEM_LIBRARY = "system_library"
    ENVIRONMENT_VARIABLE = "environment_variable"
    NETWORK_CONNECTIVITY = "network_connectivity"


@dataclass(frozen=True)
class Dependency:
    """Mechanical dependency definition with no semantic meaning."""
    name: str
    dep_type: DependencyType
    python_import: Optional[str] = None  # For Python packages
    pip_package: Optional[str] = None    # Alternative pip name
    system_package: Optional[str] = None # System package name
    env_var: Optional[str] = None        # Environment variable name
    url: Optional[str] = None           # URL for connectivity check
    min_version: Optional[str] = None   # Optional minimum version
    
    def __post_init__(self) -> None:
        """Validate dependency configuration."""
        if self.dep_type == DependencyType.PYTHON_PACKAGE and not self.python_import:
            raise ValueError(f"Python package {self.name} must specify python_import")
        if self.dep_type == DependencyType.SYSTEM_LIBRARY and not self.system_package:
            raise ValueError(f"System library {self.name} must specify system_package")
        if self.dep_type == DependencyType.ENVIRONMENT_VARIABLE and not self.env_var:
            raise ValueError(f"Environment variable {self.name} must specify env_var")
        if self.dep_type == DependencyType.NETWORK_CONNECTIVITY and not self.url:
            raise ValueError(f"Network connectivity {self.name} must specify url")


class DepsInstaller:
    """
    Stateless installer for networking and HTTP dependencies.
    
    This class:
    1. Installs required HTTP client and SSL libraries
    2. Verifies package imports and availability
    3. Checks network connectivity to required endpoints
    4. Verifies DeepSeek API environment variables
    5. Logs mechanical progress and errors
    
    This class does NOT:
    1. Call any LLM or perform inference
    2. Construct or send prompts
    3. Parse or interpret responses
    4. Implement retry logic or heuristics
    5. Cache or store any state
    6. Perform intelligence or reasoning
    7. Make assumptions about API key validity
    """
    
    # Core networking dependencies for brain connectivity
    DEPENDENCIES = [
        # Python packages (HTTP clients)
        Dependency(
            name="aiohttp",
            dep_type=DependencyType.PYTHON_PACKAGE,
            python_import="aiohttp",
            pip_package="aiohttp",
        ),
        Dependency(
            name="certifi",
            dep_type=DependencyType.PYTHON_PACKAGE,
            python_import="certifi",
            pip_package="certifi",
        ),
        # System libraries (SSL/certificates)
        Dependency(
            name="OpenSSL",
            dep_type=DependencyType.SYSTEM_LIBRARY,
            system_package="openssl",
        ),
        # Environment variables (DeepSeek API)
        Dependency(
            name="DeepSeek API Key",
            dep_type=DependencyType.ENVIRONMENT_VARIABLE,
            env_var="DEEPSEEK_API_KEY",
        ),
        # Network connectivity checks (only directly required endpoints)
        Dependency(
            name="DeepSeek API Endpoint",
            dep_type=DependencyType.NETWORK_CONNECTIVITY,
            url="https://api.deepseek.com",  # Directly required endpoint only
        ),
    ]
    
    def __init__(self, installer_log_level: int = DEPS_INSTALLER_LOG_LEVEL) -> None:
        """
        Initialize dependency installer.
        
        Args:
            installer_log_level: Logging level for installation operations
        """
        self._logger = logging.getLogger(f"{__name__}.DepsInstaller")
        self._log_level = installer_log_level
        self._platform = self._detect_platform()
        
        # Log platform detection
        self._log_platform_detection()
    
    def install_and_verify(self) -> bool:
        """
        Install and verify all networking dependencies.
        
        Returns:
            True if all dependencies are available and usable
        
        Raises:
            RuntimeError: If installation fails or dependencies are missing
        
        Note:
            - No retry logic
            - No LLM interaction
            - No prompt construction
            - No response interpretation
        """
        self._logger.log(self._log_level, "Starting networking dependency installation")
        
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
                self._logger.log(self._log_level, "All networking dependencies verified successfully")
                return True
            else:
                # Log specific failures
                failed_deps = [name for name, ok in verification_results.items() if not ok]
                error_msg = f"Networking dependencies failed verification: {', '.join(failed_deps)}"
                self._logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            # Log failure and re-raise
            self._logger.error(f"Networking installation failed: {str(e)}")
            raise
    
    def _install_system_dependencies(self) -> None:
        """Install required system libraries."""
        if self._platform != Platform.LINUX:
            self._logger.warning("System dependency installation only supported on Linux")
            return
        
        system_deps = [
            dep for dep in self.DEPENDENCIES 
            if dep.dep_type == DependencyType.SYSTEM_LIBRARY
        ]
        
        if not system_deps:
            self._logger.log(self._log_level, "No system dependencies to install")
            return
        
        for dep in system_deps:
            if dep.system_package:
                self._install_system_package(dep.name, dep.system_package)
    
    def _install_system_package(self, name: str, package: str) -> None:
        """Install a single system package."""
        self._logger.log(self._log_level, f"Installing system package: {name} ({package})")
        
        try:
            # Use apt-get on Debian/Ubuntu systems
            result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", package],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode == 0:
                self._logger.log(self._log_level, f"Successfully installed system package: {name}")
            else:
                # Check if already installed
                check_result = subprocess.run(
                    ["dpkg", "-s", package],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                
                if check_result.returncode == 0:
                    self._logger.log(self._log_level, f"System package already installed: {name}")
                else:
                    error_msg = f"Failed to install system package {name}: {result.stderr[:200]}"
                    self._logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except FileNotFoundError:
            raise RuntimeError(f"apt-get not found. Cannot install system package: {name}")
        except Exception as e:
            raise RuntimeError(f"Error installing system package {name}: {str(e)}")
    
    def _install_python_dependencies(self) -> None:
        """Install Python packages for networking."""
        python_deps = [
            dep for dep in self.DEPENDENCIES 
            if dep.dep_type == DependencyType.PYTHON_PACKAGE
        ]
        
        for dep in python_deps:
            self._install_python_package(dep)
    
    def _install_python_package(self, dep: Dependency) -> None:
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
        Verify all networking dependencies are available and usable.
        
        Returns:
            Dictionary mapping dependency names to verification results
        """
        verification_results = {}
        
        for dep in self.DEPENDENCIES:
            try:
                if dep.dep_type == DependencyType.PYTHON_PACKAGE:
                    verified = self._verify_python_package(dep)
                elif dep.dep_type == DependencyType.SYSTEM_LIBRARY:
                    verified = self._verify_system_library(dep)
                elif dep.dep_type == DependencyType.ENVIRONMENT_VARIABLE:
                    verified = self._verify_environment_variable(dep)
                elif dep.dep_type == DependencyType.NETWORK_CONNECTIVITY:
                    verified = self._verify_network_connectivity(dep)
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
            
            # Additional checks for specific packages
            if dep.python_import == "aiohttp":
                # Check that aiohttp has required attributes for HTTP client
                required_attrs = ["ClientSession", "ClientTimeout"]
                for attr in required_attrs:
                    if not hasattr(module, attr):
                        self._logger.error(f"aiohttp missing required attribute: {attr}")
                        return False
            
            return True
            
        except ImportError as e:
            self._logger.error(f"Failed to import {dep.python_import}: {str(e)}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error importing {dep.python_import}: {str(e)}")
            return False
    
    def _verify_system_library(self, dep: Dependency) -> bool:
        """Verify system library is available."""
        if not dep.system_package:
            self._logger.error(f"No system_package specified for {dep.name}")
            return False
        
        if self._platform != Platform.LINUX:
            self._logger.warning(f"System library verification only supported on Linux for {dep.name}")
            return True  # Not a failure, just unsupported check
        
        try:
            # Check if package is installed via dpkg
            result = subprocess.run(
                ["dpkg", "-s", dep.system_package],
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode == 0:
                # Extract version from dpkg output
                version_line = [line for line in result.stdout.split('\n') if line.startswith('Version:')]
                if version_line:
                    version = version_line[0].split(':')[1].strip()
                    self._logger.log(self._log_level, f"  {dep.name} version: {version}")
                return True
            else:
                self._logger.error(f"System library not found: {dep.system_package}")
                return False
                
        except Exception as e:
            self._logger.error(f"Error verifying system library {dep.name}: {str(e)}")
            return False
    
    def _verify_environment_variable(self, dep: Dependency) -> bool:
        """
        Verify environment variable exists and is non-empty.
        
        Note: No validation of content validity or format.
        Only mechanical existence and non-empty checks.
        """
        if not dep.env_var:
            self._logger.error(f"No env_var specified for {dep.name}")
            return False
        
        value = os.environ.get(dep.env_var)
        
        if value is None:
            self._logger.error(f"Environment variable not set: {dep.env_var}")
            return False
        
        if not value.strip():
            self._logger.error(f"Environment variable is empty: {dep.env_var}")
            return False
        
        # Log that variable exists (but not the value for security)
        # Note: No interpretation of whether the value is "valid" or "real"
        masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        self._logger.log(self._log_level, f"  {dep.name} is set (value: {masked_value})")
        
        return True
    
    def _verify_network_connectivity(self, dep: Dependency) -> bool:
        """
        Verify network connectivity to a URL.
        
        Note: Performs basic TCP/SSL connectivity check only.
        No HTTP requests that could be interpreted as API calls.
        """
        if not dep.url:
            self._logger.error(f"No URL specified for {dep.name}")
            return False
        
        self._logger.log(self._log_level, f"Verifying connectivity to: {dep.url}")
        
        try:
            # Parse URL
            parsed = urlparse(dep.url)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            
            if not hostname:
                self._logger.error(f"Could not parse hostname from URL: {dep.url}")
                return False
            
            # DNS resolution check
            try:
                ip_address = socket.gethostbyname(hostname)
                self._logger.log(self._log_level, f"  Resolved {hostname} to {ip_address}")
            except socket.gaierror as e:
                self._logger.error(f"DNS resolution failed for {hostname}: {str(e)}")
                return False
            
            # TCP connectivity check
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)  # 10 second timeout
                
                sock.connect((hostname, port))
                sock.close()
                
                self._logger.log(self._log_level, f"  TCP connection successful to {hostname}:{port}")
                
                # For HTTPS, also check SSL/TLS availability
                if parsed.scheme == "https":
                    # Create SSL context
                    context = ssl.create_default_context()
                    
                    # Try SSL handshake
                    ssl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    ssl_sock.settimeout(10)
                    
                    ssl_wrapped = context.wrap_socket(ssl_sock, server_hostname=hostname)
                    ssl_wrapped.connect((hostname, port))
                    
                    # Get SSL certificate info (mechanical verification only)
                    cert = ssl_wrapped.getpeercert()
                    if cert:
                        self._logger.log(self._log_level, f"  SSL certificate verified for {hostname}")
                        # Log basic certificate info without interpretation
                        cert_subject = dict(x[0] for x in cert.get('subject', []))
                        cert_issuer = dict(x[0] for x in cert.get('issuer', []))
                        self._logger.log(self._log_level, f"    Subject: {cert_subject}")
                        self._logger.log(self._log_level, f"    Issuer: {cert_issuer}")
                    
                    ssl_wrapped.close()
                
                return True
                
            except socket.timeout:
                self._logger.error(f"Connection timeout to {hostname}:{port}")
                return False
            except ConnectionRefusedError:
                self._logger.error(f"Connection refused to {hostname}:{port}")
                return False
            except ssl.SSLError as e:
                self._logger.error(f"SSL error connecting to {hostname}:{port}: {str(e)}")
                return False
            except Exception as e:
                self._logger.error(f"Connection error to {hostname}:{port}: {str(e)}")
                return False
                
        except Exception as e:
            self._logger.error(f"Network connectivity verification failed for {dep.url}: {str(e)}")
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
        
        Usage: python -m bootstrap.deps_installer
        Exit codes: 0 = success, 1 = failure
        """
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Install and verify networking dependencies for EME brain connectivity"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        args = parser.parse_args()
        
        # Configure logging
        log_level = logging.DEBUG if args.verbose else DEPS_INSTALLER_LOG_LEVEL
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Run installer
        installer = DepsInstaller(installer_log_level=log_level)
        
        try:
            success = installer.install_and_verify()
            if success:
                print("✓ Networking dependencies installed and verified successfully")
                sys.exit(0)
            else:
                print("✗ Networking dependencies verification failed")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Networking installation failed: {str(e)}")
            sys.exit(1)


# Export public interface
__all__ = [
    "DepsInstaller",
    "Dependency",
    "Platform",
    "DependencyType",
    "DEPS_INSTALLER_LOG_LEVEL",
]

# Allow direct script execution
if __name__ == "__main__":
    DepsInstaller.run_cli()