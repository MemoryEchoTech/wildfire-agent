#!/usr/bin/env python3
"""
Unit tests for WildfireYOLOToolkit

Tests the YOLO wildfire detection toolkit directly without involving agents or LLMs.
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from owl.utils.wildfire_yolo_toolkit import WildfireYOLOToolkit, YOLO_AVAILABLE
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the required dependencies installed:")
    print("conda install ultralytics opencv pytorch torchvision -c pytorch -y")
    sys.exit(1)


class TestWildfireYOLOToolkit(unittest.TestCase):
    """Test cases for WildfireYOLOToolkit"""

    @classmethod
    def setUpClass(cls):
        """Set up test class with common resources"""
        cls.test_data_dir = Path(__file__).parent.parent / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create a test image (simple colored rectangle)
        cls.test_image_path = cls.test_data_dir / "test_wildfire_image.jpg"
        cls._create_test_image(cls.test_image_path)
        
        # Use the actual Maui image if available
        cls.maui_image_path = project_root / "workspace" / "Maui Wildfires Image.jpg"
        if not cls.maui_image_path.exists():
            cls.maui_image_path = project_root / "Maui Wildfires Image.jpg"
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources"""
        if cls.test_data_dir.exists():
            shutil.rmtree(cls.test_data_dir)
    
    @staticmethod
    def _create_test_image(image_path: Path, width: int = 640, height: int = 480):
        """Create a simple test image for testing"""
        # Create a test image with some colored regions
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate objects
        cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), -1)  # Red rectangle (fire)
        cv2.rectangle(image, (200, 200), (300, 300), (128, 128, 128), -1)  # Gray rectangle (smoke)
        cv2.rectangle(image, (400, 100), (500, 200), (0, 255, 0), -1)  # Green rectangle (vegetation)
        
        cv2.imwrite(str(image_path), image)
    
    def setUp(self):
        """Set up for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up after each test"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(YOLO_AVAILABLE, "YOLO dependencies not available")
    def test_toolkit_initialization(self):
        """Test toolkit initialization with default parameters"""
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            output_dir=str(self.temp_dir)
        )
        
        self.assertIsInstance(toolkit, WildfireYOLOToolkit)
        self.assertEqual(str(toolkit.output_dir), str(self.temp_dir))
        self.assertEqual(toolkit.confidence_threshold, 0.5)  # Default value
    
    @unittest.skipUnless(YOLO_AVAILABLE, "YOLO dependencies not available")
    def test_toolkit_initialization_with_custom_params(self):
        """Test toolkit initialization with custom parameters"""
        custom_confidence = 0.3
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            confidence_threshold=custom_confidence,
            output_dir=str(self.temp_dir)
        )
        
        self.assertEqual(toolkit.confidence_threshold, custom_confidence)
    
    @unittest.skipUnless(YOLO_AVAILABLE, "YOLO dependencies not available")
    def test_get_tools(self):
        """Test that toolkit returns proper tools"""
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            output_dir=str(self.temp_dir)
        )
        
        tools = toolkit.get_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)
        
        # Check that tools have the expected names
        tool_names = [tool.__name__ if hasattr(tool, '__name__') else 
                     getattr(tool, 'func', tool).__name__ for tool in tools]
        expected_tools = ['detect_objects_in_image', 'analyze_wildfire_image']
        
        for expected_tool in expected_tools:
            self.assertIn(expected_tool, tool_names)
    
    @unittest.skipUnless(YOLO_AVAILABLE, "YOLO dependencies not available")
    def test_detect_objects_with_test_image(self):
        """Test object detection with a simple test image"""
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            output_dir=str(self.temp_dir)
        )
        
        # Test with our created test image
        result = toolkit.detect_objects_in_image(str(self.test_image_path))
        
        # Verify result structure
        self.assertIsInstance(result, str)
        
        # Result should be a JSON string, let's parse it
        try:
            result_data = json.loads(result)
            self.assertIn('detections', result_data)
            self.assertIn('summary', result_data)
            self.assertIn('image_path', result_data)
            self.assertIn('model_info', result_data)
        except json.JSONDecodeError:
            self.fail("Result should be valid JSON")
    
    @unittest.skipUnless(YOLO_AVAILABLE, "YOLO dependencies not available")
    def test_analyze_wildfire_image_with_test_image(self):
        """Test wildfire analysis with a test image"""
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            output_dir=str(self.temp_dir)
        )
        
        result = toolkit.analyze_wildfire_image(str(self.test_image_path))
        
        # Verify result structure
        self.assertIsInstance(result, str)
        
        try:
            result_data = json.loads(result)
            self.assertIn('risk_assessment', result_data)
            self.assertIn('recommendations', result_data)
            self.assertIn('image_path', result_data)
            self.assertIn('summary', result_data)
        except json.JSONDecodeError:
            self.fail("Analysis result should be valid JSON")
    
    @unittest.skipUnless(YOLO_AVAILABLE, "YOLO dependencies not available")
    @unittest.skipUnless(Path(project_root / "workspace" / "Maui Wildfires Image.jpg").exists() or 
                        Path(project_root / "Maui Wildfires Image.jpg").exists(), 
                        "Maui wildfire image not found")
    def test_analyze_real_maui_image(self):
        """Test analysis with the real Maui wildfire image"""
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            confidence_threshold=0.3,  # Lower threshold for real wildfire detection
            output_dir=str(self.temp_dir)
        )
        
        result = toolkit.analyze_wildfire_image(str(self.maui_image_path))
        
        # Verify result structure
        self.assertIsInstance(result, str)
        
        try:
            result_data = json.loads(result)
            self.assertIn('risk_assessment', result_data)
            self.assertIn('recommendations', result_data)
            self.assertIn('image_path', result_data)
            
            # Log the results for manual verification
            print(f"\n--- Maui Image Analysis Results ---")
            print(json.dumps(result_data, indent=2))
            
        except json.JSONDecodeError:
            self.fail("Analysis result should be valid JSON")
    
    def test_invalid_image_path(self):
        """Test behavior with invalid image path"""
        if not YOLO_AVAILABLE:
            self.skipTest("YOLO dependencies not available")
            
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            output_dir=str(self.temp_dir)
        )
        
        result = toolkit.detect_objects_in_image("/nonexistent/path/image.jpg")
        
        # Should handle gracefully and return error in JSON format
        try:
            result_data = json.loads(result)
            self.assertIn('error', result_data)
            self.assertIn('detections', result_data)
            self.assertIn('summary', result_data)
        except json.JSONDecodeError:
            self.fail("Error result should be valid JSON")
    
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist"""
        if not YOLO_AVAILABLE:
            self.skipTest("YOLO dependencies not available")
            
        nonexistent_dir = self.temp_dir / "new_output_dir"
        self.assertFalse(nonexistent_dir.exists())
        
        toolkit = WildfireYOLOToolkit(
            model_path="yolo11n.pt",
            output_dir=str(nonexistent_dir)
        )
        
        # Run a detection to trigger directory creation
        toolkit.detect_objects_in_image(str(self.test_image_path))
        
        # Directory should now exist
        self.assertTrue(nonexistent_dir.exists())
    
    @unittest.skipUnless(YOLO_AVAILABLE, "YOLO dependencies not available")
    def test_toolkit_without_yolo_model(self):
        """Test toolkit behavior when YOLO model file doesn't exist"""
        # Test by trying to create toolkit with invalid model path
        # This should raise an exception during initialization
        with self.assertRaises(Exception):
            toolkit = WildfireYOLOToolkit(
                model_path="definitely_nonexistent_model.pt",
                output_dir=str(self.temp_dir)
            )


class TestYOLOAvailability(unittest.TestCase):
    """Test YOLO availability and dependencies"""
    
    def test_yolo_availability_flag(self):
        """Test that YOLO_AVAILABLE flag is properly set"""
        self.assertIsInstance(YOLO_AVAILABLE, bool)
        
        if YOLO_AVAILABLE:
            # If available, we should be able to import required modules
            try:
                import cv2
                import numpy as np
                from ultralytics import YOLO
                import torch
            except ImportError:
                self.fail("YOLO_AVAILABLE is True but dependencies are missing")


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWildfireYOLOToolkit)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestYOLOAvailability))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)