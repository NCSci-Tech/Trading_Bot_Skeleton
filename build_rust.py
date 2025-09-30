#!/usr/bin/env python3
"""
Build script for Rust components of the trading bot
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import argparse
import time

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message: str, color: str = Colors.OKGREEN):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def run_command(cmd, cwd=None, check=True, capture_output=False):
    """Run a command and handle output"""
    print_colored(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}", Colors.OKBLUE)
    
    try:
        if capture_output:
            result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
            return result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(cmd, cwd=cwd, check=check)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_colored(f"Command failed with return code {e.returncode}", Colors.FAIL)
        if hasattr(e, 'stderr') and e.stderr:
            print_colored(f"Error output: {e.stderr}", Colors.WARNING)
        if not check:
            return False
        raise
    except FileNotFoundError:
        print_colored(f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}", Colors.FAIL)
        if check:
            sys.exit(1)
        return False

def check_rust_installation():
    """Check if Rust is properly installed"""
    print_colored("🔍 Checking Rust installation...", Colors.HEADER)
    
    try:
        stdout, _ = run_command(["rustc", "--version"], capture_output=True)
        print_colored(f"✅ Rust compiler: {stdout}", Colors.OKGREEN)
    except:
        print_colored("❌ Rust compiler not found!", Colors.FAIL)
        print_colored("📝 Install Rust from: https://rustup.rs/", Colors.WARNING)
        return False
    
    try:
        stdout, _ = run_command(["cargo", "--version"], capture_output=True)
        print_colored(f"✅ Cargo: {stdout}", Colors.OKGREEN)
    except:
        print_colored("❌ Cargo not found!", Colors.FAIL)
        return False
    
    return True

def check_python_deps():
    """Check if required Python dependencies are installed"""
    print_colored("🐍 Checking Python dependencies...", Colors.HEADER)
    
    required_deps = ["maturin", "numpy"]
    missing_deps = []
    
    for dep in required_deps:
        try:
            __import__(dep)
            print_colored(f"✅ {dep} is installed", Colors.OKGREEN)
        except ImportError:
            missing_deps.append(dep)
            print_colored(f"❌ {dep} is missing", Colors.FAIL)
    
    if missing_deps:
        print_colored(f"📦 Installing missing dependencies: {', '.join(missing_deps)}", Colors.WARNING)
        run_command([sys.executable, "-m", "pip", "install"] + missing_deps)
    
    return len(missing_deps) == 0

def clean_build_artifacts():
    """Clean previous build artifacts"""
    print_colored("🧹 Cleaning build artifacts...", Colors.HEADER)
    
    rust_dir = Path("rust_modules")
    if not rust_dir.exists():
        print_colored("❌ rust_modules directory not found!", Colors.FAIL)
        return False
    
    # Clean each module
    modules = ["indicators", "orderbook", "backtesting"]
    
    for module in modules:
        module_dir = rust_dir / module
        if module_dir.exists():
            target_dir = module_dir / "target"
            if target_dir.exists():
                print_colored(f"🗑️  Cleaning {module}/target", Colors.WARNING)
                shutil.rmtree(target_dir)
            
            # Clean Python build artifacts
            build_dir = module_dir / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
            
            dist_dir = module_dir / "dist"
            if dist_dir.exists():
                shutil.rmtree(dist_dir)
    
    return True

def build_rust_module(module_name: str, release: bool = True, verbose: bool = False):
    """Build a specific Rust module"""
    print_colored(f"🔨 Building {module_name}...", Colors.HEADER)
    
    rust_dir = Path("rust_modules")
    module_dir = rust_dir / module_name
    
    if not module_dir.exists():
        print_colored(f"❌ Module {module_name} not found!", Colors.FAIL)
        return False
    
    # Build command
    cmd = ["maturin", "develop"]
    
    if release:
        cmd.append("--release")
    
    if verbose:
        cmd.extend(["--verbose"])
    
    # Add specific features if needed
    if module_name == "backtesting":
        cmd.extend(["--features", "parallel"])
    
    start_time = time.time()
    
    try:
        success = run_command(cmd, cwd=module_dir)
        build_time = time.time() - start_time
        
        if success:
            print_colored(f"✅ {module_name} built successfully in {build_time:.1f}s", Colors.OKGREEN)
        else:
            print_colored(f"❌ Failed to build {module_name}", Colors.FAIL)
        
        return success
    except Exception as e:
        print_colored(f"❌ Error building {module_name}: {e}", Colors.FAIL)
        return False

def test_rust_modules():
    """Test built Rust modules"""
    print_colored("🧪 Testing Rust modules...", Colors.HEADER)
    
    modules = ["indicators", "orderbook", "backtesting"]
    test_results = {}
    
    for module in modules:
        print_colored(f"Testing {module}...", Colors.OKBLUE)
        
        try:
            # Test import
            if module == "indicators":
                import indicators
                # Test a simple function
                import numpy as np
                test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                result = indicators.sma(test_data, 3)
                assert len(result) == 5
                test_results[module] = "✅ Pass"
            
            elif module == "orderbook":
                import orderbook
                # Test orderbook creation
                ob = orderbook.PyOrderBook("TEST")
                ob.update_bid(100.0, 1.0)
                ob.update_ask(101.0, 1.0)
                spread = ob.get_spread()
                assert spread == 1.0
                test_results[module] = "✅ Pass"
            
            elif module == "backtesting":
                import backtesting
                # Test backtest engine
                engine = backtesting.BacktestEngine(10000.0, 0.001, 0.0001)
                engine.execute_trade("2023-01-01T00:00:00Z", "TEST", "buy", 100.0, 1.0)
                test_results[module] = "✅ Pass"
            
        except Exception as e:
            test_results[module] = f"❌ Failed: {e}"
    
    # Print test results
    print_colored("\n📊 Test Results:", Colors.HEADER)
    for module, result in test_results.items():
        print_colored(f"  {module}: {result}", Colors.OKGREEN if "✅" in result else Colors.FAIL)
    
    return all("✅" in result for result in test_results.values())

def build_all_modules(release: bool = True, clean: bool = False, verbose: bool = False):
    """Build all Rust modules"""
    print_colored("🚀 Building all Rust modules...", Colors.HEADER)
    
    if clean:
        if not clean_build_artifacts():
            return False
    
    modules = ["indicators", "orderbook", "backtesting"]
    build_results = {}
    total_start_time = time.time()
    
    for module in modules:
        success = build_rust_module(module, release=release, verbose=verbose)
        build_results[module] = success
    
    total_time = time.time() - total_start_time
    
    # Print build summary
    successful_builds = sum(build_results.values())
    total_builds = len(build_results)
    
    print_colored(f"\n📋 Build Summary:", Colors.HEADER)
    print_colored(f"  Total time: {total_time:.1f}s", Colors.OKBLUE)
    print_colored(f"  Successful: {successful_builds}/{total_builds}", Colors.OKGREEN)
    
    for module, success in build_results.items():
        status = "✅ Success" if success else "❌ Failed"
        color = Colors.OKGREEN if success else Colors.FAIL
        print_colored(f"  {module}: {status}", color)
    
    return successful_builds == total_builds

def create_wheel_packages():
    """Create wheel packages for distribution"""
    print_colored("📦 Creating wheel packages...", Colors.HEADER)
    
    rust_dir = Path("rust_modules")
    modules = ["indicators", "orderbook", "backtesting"]
    
    for module in modules:
        module_dir = rust_dir / module
        if not module_dir.exists():
            continue
        
        print_colored(f"📦 Building wheel for {module}...", Colors.OKBLUE)
        
        cmd = ["maturin", "build", "--release", "--out", "../../wheels"]
        
        try:
            run_command(cmd, cwd=module_dir)
            print_colored(f"✅ Wheel created for {module}", Colors.OKGREEN)
        except:
            print_colored(f"❌ Failed to create wheel for {module}", Colors.FAIL)

def show_performance_info():
    """Show performance optimization information"""
    print_colored("\n⚡ Performance Optimization Tips:", Colors.HEADER)
    
    tips = [
        "🚀 Use --release flag for production builds (significant performance improvement)",
        "🔧 Set RUSTFLAGS='-C target-cpu=native' for CPU-specific optimizations",
        "📊 Enable LTO (Link Time Optimization) for smaller binaries",
        "⚡ Use rayon for parallel processing in compute-intensive operations",
        "🎯 Profile with 'cargo bench' to identify bottlenecks"
    ]
    
    for tip in tips:
        print_colored(f"  {tip}", Colors.OKBLUE)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Build Rust modules for trading bot")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts before building")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode (faster compilation, slower runtime)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--test", action="store_true", help="Run tests after building")
    parser.add_argument("--module", type=str, help="Build specific module only")
    parser.add_argument("--wheel", action="store_true", help="Create wheel packages")
    parser.add_argument("--performance-tips", action="store_true", help="Show performance optimization tips")
    
    args = parser.parse_args()
    
    if args.performance_tips:
        show_performance_info()
        return
    
    print_colored("🦀 Rust Trading Bot Module Builder", Colors.HEADER)
    print_colored("=" * 40, Colors.HEADER)
    
    # Check prerequisites
    if not check_rust_installation():
        sys.exit(1)
    
    if not check_python_deps():
        print_colored("⚠️  Some dependencies were missing but should be installed now", Colors.WARNING)
    
    # Set build mode
    release_mode = not args.debug
    mode_str = "release" if release_mode else "debug"
    print_colored(f"🏗️  Build mode: {mode_str}", Colors.OKBLUE)
    
    # Build modules
    if args.module:
        # Build specific module
        success = build_rust_module(args.module, release=release_mode, verbose=args.verbose)
        if not success:
            sys.exit(1)
    else:
        # Build all modules
        success = build_all_modules(release=release_mode, clean=args.clean, verbose=args.verbose)
        if not success:
            print_colored("❌ Some modules failed to build", Colors.FAIL)
            sys.exit(1)
    
    # Run tests
    if args.test:
        if not test_rust_modules():
            print_colored("❌ Some tests failed", Colors.FAIL)
            sys.exit(1)
    
    # Create wheels
    if args.wheel:
        create_wheel_packages()
    
    print_colored("\n🎉 Build completed successfully!", Colors.OKGREEN)
    print_colored("You can now import the Rust modules in Python:", Colors.OKBLUE)
    print_colored("  import indicators", Colors.OKCYAN)
    print_colored("  import orderbook", Colors.OKCYAN)
    print_colored("  import backtesting", Colors.OKCYAN)

if __name__ == "__main__":
    main()