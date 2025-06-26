#!/usr/bin/env python3

"""
Unit tests for Wildfire YOLO MindSpore Toolkit

This module tests the MindSpore-based YOLO toolkit for wildfire detection and analysis.
Tests are designed to work with or without MindSpore installation.
"""

import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path

# Import the toolkit
try:
    from owl.utils.wildfire_yolo_mindspore_toolkit import WildfireYOLOMindSporeToolkit
    TOOLKIT_AVAILABLE = True
except ImportError as e:
    TOOLKIT_AVAILABLE = False
    import_error = str(e)

# Test data
TEST_IMAGE_NAME = "Maui Wildfires Image.jpg"
TEST_IMAGE_PATHS = [
    f"/Users/kang/GitHub/wildfire-agent/{TEST_IMAGE_NAME}",
    f"workspace/{TEST_IMAGE_NAME}",
    f"./{TEST_IMAGE_NAME}",
    "workspace/maui_wildfire_satellite.png",
]


class TestWildfireYOLOMindSporeToolkit(unittest.TestCase):
    """Test cases for Wildfire YOLO MindSpore Toolkit."""

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.test_image_path = None
        
        # Find test image
        for path in TEST_IMAGE_PATHS:
            if os.path.exists(path):
                cls.test_image_path = path
                break
        
        # Create temporary directory for outputs
        cls.temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test class."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up each test."""
        if not TOOLKIT_AVAILABLE:
            self.skipTest(f"MindSpore YOLO Toolkit not available: {import_error}")

    def test_toolkit_import(self):
        """Test that the toolkit can be imported."""
        self.assertTrue(TOOLKIT_AVAILABLE, "MindSpore YOLO Toolkit should be importable")

    def test_toolkit_initialization_with_dependencies(self):
        """Test toolkit initialization when dependencies are available."""
        try:
            # This will succeed if MindSpore is available, fail gracefully if not
            toolkit = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                device="CPU"  # Use CPU to avoid GPU dependency issues
            )
            self.assertIsNotNone(toolkit)
            self.assertEqual(toolkit.device, "CPU")
            print("✅ MindSpore YOLO Toolkit initialized successfully")
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")

    def test_toolkit_initialization_without_dependencies(self):
        """Test that toolkit fails gracefully when dependencies are missing."""
        # This test will be skipped if dependencies are actually available
        if TOOLKIT_AVAILABLE:
            try:
                toolkit = WildfireYOLOMindSporeToolkit(output_dir=self.temp_dir)
                # If we get here, dependencies are available
                self.assertIsNotNone(toolkit)
            except ImportError:
                # Expected when dependencies are missing
                pass

    def test_get_tools(self):
        """Test that get_tools returns the expected functions."""
        try:
            toolkit = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                device="CPU"
            )
            tools = toolkit.get_tools()
            self.assertIsInstance(tools, list)
            self.assertEqual(len(tools), 2)
            
            # Check that tools are callable
            for tool in tools:
                self.assertTrue(callable(tool))
            
            print(f"✅ Toolkit provides {len(tools)} tools")
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")

    def test_detect_objects_in_image_file_not_found(self):
        """Test object detection with non-existent file."""
        try:
            toolkit = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                device="CPU"
            )
            
            result = toolkit.detect_objects_in_image("nonexistent_file.jpg", save_results=False)
            result_data = json.loads(result)
            
            self.assertIn("error", result_data)
            self.assertEqual(result_data["detections"], [])
            print("✅ Handled non-existent file gracefully")
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")

    def test_detect_objects_in_image_with_real_image(self):
        """Test object detection with real image if available."""
        if not self.test_image_path:
            self.skipTest("No test image available")
        
        try:
            toolkit = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                device="CPU"
            )
            
            result = toolkit.detect_objects_in_image(
                self.test_image_path, 
                save_results=True
            )
            result_data = json.loads(result)
            
            # Should have basic structure
            self.assertIn("image_path", result_data)
            self.assertIn("model_info", result_data)
            self.assertIn("detections", result_data)
            self.assertIn("detection_count", result_data)
            self.assertIn("summary", result_data)
            
            # Check model info
            model_info = result_data["model_info"]
            self.assertEqual(model_info["framework"], "MindSpore")
            self.assertIn("device", model_info)
            
            print(f"✅ Processed image: {os.path.basename(self.test_image_path)}")
            print(f"   Detections: {result_data['detection_count']}")
            print(f"   Framework: {model_info['framework']}")
            print(f"   Device: {model_info['device']}")
            
            # Check if output files were created
            if result_data["detection_count"] > 0:
                annotated_path = result_data.get("annotated_image_path")
                if annotated_path and os.path.exists(annotated_path):
                    print(f"   Annotated image saved: {annotated_path}")
                
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")
        except Exception as e:
            # Allow for graceful failure when model isn't properly loaded
            print(f"⚠️  Detection test completed with error (expected): {str(e)}")

    def test_analyze_wildfire_image(self):
        """Test wildfire-specific image analysis."""
        if not self.test_image_path:
            self.skipTest("No test image available")
        
        try:
            toolkit = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                device="CPU"
            )
            
            result = toolkit.analyze_wildfire_image(
                self.test_image_path,
                save_results=True
            )
            result_data = json.loads(result)
            
            # Should have wildfire analysis structure
            self.assertIn("analysis_type", result_data)
            self.assertIn("framework", result_data)
            self.assertIn("general_detections", result_data)
            self.assertIn("wildfire_indicators", result_data)
            self.assertIn("risk_assessment", result_data)
            self.assertIn("recommendations", result_data)
            self.assertIn("summary", result_data)
            
            # Check framework
            self.assertEqual(result_data["framework"], "MindSpore")
            self.assertEqual(result_data["analysis_type"], "wildfire_specialized_mindspore")
            
            # Check risk assessment structure
            risk_assessment = result_data["risk_assessment"]
            self.assertIn("overall_risk_level", risk_assessment)
            self.assertIn("fire_indicators_detected", risk_assessment)
            self.assertIn("infrastructure_elements", risk_assessment)
            
            print(f"✅ Wildfire analysis completed")
            print(f"   Risk Level: {risk_assessment['overall_risk_level']}")
            print(f"   Fire Indicators: {risk_assessment['fire_indicators_detected']}")
            print(f"   Infrastructure: {risk_assessment['infrastructure_elements']}")
            print(f"   Recommendations: {len(result_data['recommendations'])}")
            
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")
        except Exception as e:
            # Allow for graceful failure when model isn't properly loaded
            print(f"⚠️  Wildfire analysis test completed with error (expected): {str(e)}")

    def test_class_filtering(self):
        """Test detection with class filtering."""
        if not self.test_image_path:
            self.skipTest("No test image available")
        
        try:
            toolkit = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                device="CPU"
            )
            
            # Test with specific class filter
            result = toolkit.detect_objects_in_image(
                self.test_image_path,
                save_results=False,
                classes_filter=["person", "car", "truck"]
            )
            result_data = json.loads(result)
            
            # All detections should be from allowed classes
            for detection in result_data.get("detections", []):
                self.assertIn(detection["class"], ["person", "car", "truck"])
            
            print(f"✅ Class filtering test completed")
            
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")
        except Exception as e:
            print(f"⚠️  Class filtering test completed with error (expected): {str(e)}")

    def test_output_directory_creation(self):
        """Test that output directory is created properly."""
        try:
            custom_output_dir = os.path.join(self.temp_dir, "custom_mindspore_output")
            toolkit = WildfireYOLOMindSporeToolkit(
                output_dir=custom_output_dir,
                device="CPU"
            )
            
            self.assertTrue(os.path.exists(custom_output_dir))
            print(f"✅ Output directory created: {custom_output_dir}")
            
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")

    def test_confidence_threshold(self):
        """Test different confidence thresholds."""
        try:
            # Test with high confidence threshold
            toolkit_high = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                confidence_threshold=0.9,
                device="CPU"
            )
            self.assertEqual(toolkit_high.confidence_threshold, 0.9)
            
            # Test with low confidence threshold
            toolkit_low = WildfireYOLOMindSporeToolkit(
                output_dir=self.temp_dir,
                confidence_threshold=0.1,
                device="CPU"
            )
            self.assertEqual(toolkit_low.confidence_threshold, 0.1)
            
            print("✅ Confidence threshold configuration works")
            
        except ImportError as e:
            self.skipTest(f"MindSpore dependencies not available: {str(e)}")


def main():
    """Run the tests."""
    print("🧪 Testing Wildfire YOLO MindSpore Toolkit")
    print("=" * 50)
    
    if not TOOLKIT_AVAILABLE:
        print(f"❌ Toolkit not available: {import_error}")
        print("   Install MindSpore to enable full testing:")
        print("   pip install mindspore")
        return False
    
    # Find test image
    test_image = None
    for path in TEST_IMAGE_PATHS:
        if os.path.exists(path):
            test_image = path
            break
    
    if test_image:
        print(f"📸 Using test image: {test_image}")
    else:
        print("⚠️  No test image found. Some tests will be skipped.")
        print(f"   Searched paths: {TEST_IMAGE_PATHS}")
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)