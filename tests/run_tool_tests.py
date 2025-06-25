#!/usr/bin/env python3
"""
Test Runner for Wildfire Agent Tool Tests

This script runs all tool tests without invoking agents or LLMs.
It automatically detects which tools require LLM capabilities and skips them appropriately.
"""

import unittest
import sys
import os
import logging
from pathlib import Path
import importlib.util
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ToolTestRunner:
    """Manages and runs tool tests with dependency checking"""
    
    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path(__file__).parent / "tool_tests"
        self.results = {}
        
    def discover_test_modules(self) -> List[str]:
        """Discover all test modules in the tool_tests directory"""
        test_modules = []
        
        for test_file in self.test_dir.glob("test_*.py"):
            module_name = test_file.stem
            test_modules.append(module_name)
            
        logger.info(f"Discovered {len(test_modules)} test modules: {test_modules}")
        return test_modules
    
    def check_dependencies(self, module_name: str) -> Dict[str, Any]:
        """Check if dependencies for a test module are available"""
        dependency_status = {
            'available': True,
            'missing_deps': [],
            'skip_reason': None
        }
        
        # Define dependency requirements for different tools
        dependency_map = {
            'test_wildfire_yolo_toolkit': ['cv2', 'numpy', 'ultralytics', 'torch'],
            'test_image_analysis_toolkit': ['PIL', 'cv2'],
            'test_code_execution_toolkit': [],  # No special deps needed
            'test_search_toolkit': [],  # No special deps needed
            'test_browser_toolkit': ['selenium'],
            'test_video_analysis_toolkit': ['cv2', 'ffmpeg'],
        }
        
        required_deps = dependency_map.get(module_name, [])
        
        for dep in required_deps:
            try:
                if dep == 'ffmpeg':
                    # Special handling for ffmpeg (system dependency)
                    import subprocess
                    subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, check=True)
                else:
                    importlib.import_module(dep)
            except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
                dependency_status['available'] = False
                dependency_status['missing_deps'].append(dep)
        
        if not dependency_status['available']:
            dependency_status['skip_reason'] = f"Missing dependencies: {', '.join(dependency_status['missing_deps'])}"
            
        return dependency_status
    
    def is_llm_dependent_tool(self, module_name: str) -> bool:
        """Check if a tool requires LLM capabilities"""
        llm_dependent_tools = [
            'test_image_analysis_toolkit',  # Uses LLM for image understanding
            'test_video_analysis_toolkit',  # Uses LLM for video understanding  
            'test_document_processing_toolkit',  # Uses LLM for document analysis
        ]
        
        return module_name in llm_dependent_tools
    
    def run_single_test_module(self, module_name: str, skip_llm: bool = True) -> unittest.TestResult:
        """Run tests for a single module"""
        logger.info(f"Running tests for module: {module_name}")
        
        # Check if this is an LLM-dependent tool and we're skipping LLM tests
        if skip_llm and self.is_llm_dependent_tool(module_name):
            logger.info(f"Skipping {module_name} - LLM-dependent tool")
            # Return a mock successful result
            result = unittest.TestResult()
            result.testsRun = 0
            return result
        
        # Check dependencies
        dep_status = self.check_dependencies(module_name)
        if not dep_status['available']:
            logger.warning(f"Skipping {module_name} - {dep_status['skip_reason']}")
            result = unittest.TestResult()
            result.testsRun = 0
            result.skipped = [(None, dep_status['skip_reason'])]
            return result
        
        try:
            # Import the test module
            module_path = self.test_dir / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Load and run tests
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            runner = unittest.TextTestRunner(verbosity=2, buffer=True)
            result = runner.run(suite)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running tests for {module_name}: {e}")
            result = unittest.TestResult()
            result.errors = [(None, str(e))]
            return result
    
    def run_all_tests(self, skip_llm: bool = True, include_patterns: List[str] = None) -> Dict[str, unittest.TestResult]:
        """Run all discovered tool tests"""
        test_modules = self.discover_test_modules()
        
        # Filter by include patterns if specified
        if include_patterns:
            filtered_modules = []
            for module in test_modules:
                for pattern in include_patterns:
                    if pattern in module:
                        filtered_modules.append(module)
                        break
            test_modules = filtered_modules
        
        results = {}
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        
        logger.info(f"Running {len(test_modules)} test modules...")
        logger.info(f"LLM-dependent tests: {'SKIPPED' if skip_llm else 'INCLUDED'}")
        
        for module_name in test_modules:
            result = self.run_single_test_module(module_name, skip_llm)
            results[module_name] = result
            
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(getattr(result, 'skipped', []))
        
        # Print summary
        print("\n" + "="*60)
        print("TOOL TESTS SUMMARY")
        print("="*60)
        print(f"Total modules: {len(test_modules)}")
        print(f"Total tests run: {total_tests}")
        print(f"Failures: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        
        if total_failures > 0 or total_errors > 0:
            print("\nFAILED TESTS:")
            for module_name, result in results.items():
                if result.failures:
                    print(f"  {module_name}: {len(result.failures)} failures")
                if result.errors:
                    print(f"  {module_name}: {len(result.errors)} errors")
        
        success_rate = (total_tests - total_failures - total_errors) / max(total_tests, 1) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")
        print("="*60)
        
        return results


def main():
    """Main function to run tool tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Wildfire Agent Tool Tests")
    parser.add_argument("--include-llm", action="store_true", 
                       help="Include LLM-dependent tool tests")
    parser.add_argument("--filter", nargs="+", 
                       help="Filter tests by module name patterns")
    parser.add_argument("--list", action="store_true",
                       help="List available test modules")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = ToolTestRunner()
    
    if args.list:
        modules = runner.discover_test_modules()
        print("Available test modules:")
        for module in modules:
            llm_dep = runner.is_llm_dependent_tool(module)
            dep_status = runner.check_dependencies(module)
            status = "✓" if dep_status['available'] else "✗"
            llm_marker = " (LLM)" if llm_dep else ""
            print(f"  {status} {module}{llm_marker}")
        return
    
    # Run tests
    results = runner.run_all_tests(
        skip_llm=not args.include_llm,
        include_patterns=args.filter
    )
    
    # Exit with appropriate code
    has_failures = any(len(r.failures) + len(r.errors) > 0 for r in results.values())
    sys.exit(1 if has_failures else 0)


if __name__ == '__main__':
    main()