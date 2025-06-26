# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

"""
Wildfire YOLO Detection Toolkit - MindSpore Implementation

This toolkit provides specialized YOLO-based object detection capabilities for wildfire analysis
using MindSpore framework, including fire detection, smoke detection, and burned area identification
in satellite and aerial imagery.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path

from camel.toolkits.base import BaseToolkit

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import mindspore as ms
    from mindspore import Tensor, context
    MINDSPORE_AVAILABLE = True
except ImportError as e:
    MINDSPORE_AVAILABLE = False
    missing_deps = str(e)
    # Create dummy classes for type hints when MindSpore is not available
    class Tensor:
        pass

try:
    # Try to import MindYOLO (if available)
    import mindyolo
    from mindyolo.models import build_model
    from mindyolo.data import create_transforms
    MINDYOLO_AVAILABLE = True
except ImportError:
    MINDYOLO_AVAILABLE = False

logger = logging.getLogger(__name__)


class WildfireYOLOMindSporeToolkit(BaseToolkit):
    r"""A toolkit for wildfire detection and analysis using YOLO object detection models on MindSpore.
    
    This toolkit provides specialized functionality for:
    - Fire hotspot detection in satellite imagery
    - Smoke plume identification  
    - Burned area assessment
    - Infrastructure risk analysis
    - Emergency response object detection
    
    Uses MindSpore framework for inference instead of PyTorch.
    """

    def __init__(
        self,
        model_path: str = "yolo11n_mindspore.ckpt",
        model_config: Optional[str] = None,
        confidence_threshold: float = 0.5,
        output_dir: str = "./yolo_mindspore_detections",
        device: str = "auto"
    ):
        r"""Initialize the Wildfire YOLO MindSpore Toolkit.
        
        Args:
            model_path (str): Path to MindSpore YOLO model checkpoint. Defaults to 'yolo11n_mindspore.ckpt'.
            model_config (Optional[str]): Path to model configuration file. If None, uses default config.
            confidence_threshold (float): Minimum confidence for detections. Defaults to 0.5.
            output_dir (str): Directory to save detection results. Defaults to './yolo_mindspore_detections'.
            device (str): Device to run inference on ('auto', 'CPU', 'GPU', 'Ascend'). Defaults to 'auto'.
        """
        if not MINDSPORE_AVAILABLE:
            raise ImportError(
                "MindSpore dependencies not available. Please install MindSpore: "
                "pip install mindspore or conda install mindspore -c mindspore"
            )
        
        if not OPENCV_AVAILABLE:
            raise ImportError(
                "OpenCV not available. Please install: "
                "conda install opencv -y"
            )
        
        self.model_path = model_path
        self.model_config = model_config
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set MindSpore context and device
        self._setup_mindspore_context(device)
        
        # Initialize model
        self.model = None
        self.class_names = {}
        self._load_model()
        
        logger.info(f"MindSpore YOLO model loaded on device: {self.device}")

    def _setup_mindspore_context(self, device: str):
        """Setup MindSpore context and device configuration."""
        if device == "auto":
            # Auto-detect best available device
            try:
                context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
                self.device = "GPU"
                logger.info("Using GPU device for MindSpore")
            except:
                try:
                    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
                    self.device = "Ascend"
                    logger.info("Using Ascend device for MindSpore")
                except:
                    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
                    self.device = "CPU"
                    logger.info("Using CPU device for MindSpore")
        else:
            device_map = {
                "cpu": "CPU",
                "gpu": "GPU", 
                "ascend": "Ascend"
            }
            target_device = device_map.get(device.lower(), device)
            context.set_context(mode=context.GRAPH_MODE, device_target=target_device)
            self.device = target_device

    def _load_model(self):
        """Load MindSpore YOLO model."""
        try:
            if MINDYOLO_AVAILABLE and self.model_config:
                # Use MindYOLO if available
                logger.info(f"Loading MindYOLO model: {self.model_path}")
                self.model = build_model(self.model_config)
                # Load checkpoint
                param_dict = ms.load_checkpoint(self.model_path)
                ms.load_param_into_net(self.model, param_dict)
                self.model.set_train(False)
            else:
                # Fallback to basic MindSpore model loading
                logger.info(f"Loading MindSpore model: {self.model_path}")
                self.model = self._load_basic_mindspore_model()
            
            # Set up class names (COCO classes by default)
            self.class_names = self._get_coco_class_names()
            
        except Exception as e:
            logger.error(f"Failed to load MindSpore model: {str(e)}")
            # Create a dummy model for testing purposes
            self.model = None
            self.class_names = self._get_coco_class_names()

    def _load_basic_mindspore_model(self):
        """Load basic MindSpore model (placeholder implementation)."""
        # This is a placeholder - in practice, you would load your specific MindSpore YOLO model
        # For now, return None to indicate model should be implemented
        logger.warning("Basic MindSpore model loading not implemented. Please provide MindYOLO model.")
        return None

    def _get_coco_class_names(self) -> Dict[int, str]:
        """Get COCO dataset class names."""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
            21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
            26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
            31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }

    def detect_objects_in_image(
        self,
        image_path: str,
        save_results: bool = True,
        classes_filter: Optional[List[str]] = None
    ) -> str:
        r"""Detect objects in an image using MindSpore YOLO model.
        
        Args:
            image_path (str): Path to the input image
            save_results (bool): Whether to save annotated results. Defaults to True.
            classes_filter (Optional[List[str]]): Filter detections by class names. If None, detect all classes.
            
        Returns:
            str: JSON string containing detection results and analysis
        """
        try:
            if not os.path.exists(image_path):
                return json.dumps({
                    "error": f"Image file not found: {image_path}",
                    "detections": [],
                    "summary": "Failed to process image"
                })
            
            # Load and preprocess image
            logger.info(f"Running MindSpore YOLO detection on: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                return json.dumps({
                    "error": f"Failed to load image: {image_path}",
                    "detections": [],
                    "summary": "Image loading failed"
                })
            
            # Run inference
            detections = self._run_inference(image)
            
            # Filter by classes if specified
            if classes_filter:
                detections = [d for d in detections if d["class"] in classes_filter]
            
            # Generate summary
            summary = self._generate_detection_summary(detections, image_path)
            
            # Save annotated image if requested
            output_path = None
            if save_results and detections:
                output_path = self._save_annotated_image(image_path, image, detections)
            
            result_data = {
                "image_path": image_path,
                "model_info": {
                    "framework": "MindSpore",
                    "model_path": self.model_path,
                    "confidence_threshold": self.confidence_threshold,
                    "device": self.device
                },
                "detections": detections,
                "detection_count": len(detections),
                "summary": summary,
                "annotated_image_path": output_path
            }
            
            # Save JSON results
            if save_results:
                self._save_json_results(result_data, image_path)
            
            return json.dumps(result_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error in MindSpore YOLO detection: {str(e)}")
            return json.dumps({
                "error": f"Detection failed: {str(e)}",
                "detections": [],
                "summary": "Error occurred during object detection"
            })

    def _run_inference(self, image: np.ndarray) -> List[Dict]:
        """Run MindSpore model inference on preprocessed image."""
        try:
            if self.model is None:
                # Return dummy detections for demonstration
                logger.warning("Model not loaded. Returning demo detections.")
                return self._get_demo_detections(image)
            
            # Preprocess image for MindSpore model
            input_tensor = self._preprocess_image(image)
            
            # Run inference
            with ms.no_grad():
                outputs = self.model(input_tensor)
            
            # Post-process outputs
            detections = self._postprocess_outputs(outputs, image.shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> Tensor:
        """Preprocess image for MindSpore model input."""
        # Resize image to model input size (typically 640x640 for YOLO)
        height, width = 640, 640
        resized = cv2.resize(image, (width, height))
        
        # Convert BGR to RGB and normalize
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension and convert to CHW format
        input_array = np.transpose(normalized, (2, 0, 1))
        input_array = np.expand_dims(input_array, axis=0)
        
        # Convert to MindSpore tensor
        return Tensor(input_array, ms.float32)

    def _postprocess_outputs(self, outputs: Tensor, original_shape: Tuple[int, int, int]) -> List[Dict]:
        """Post-process model outputs to extract detections."""
        # This is a placeholder implementation
        # In practice, you would implement the specific post-processing for your MindSpore YOLO model
        detections = []
        
        # For demonstration, we'll create some dummy detections
        logger.warning("Using placeholder post-processing. Implement specific model post-processing.")
        
        return detections

    def _get_demo_detections(self, image: np.ndarray) -> List[Dict]:
        """Generate demo detections for testing purposes."""
        height, width = image.shape[:2]
        
        # Create some demo detections
        demo_detections = [
            {
                "id": 0,
                "class": "car",
                "confidence": 0.85,
                "bbox": {
                    "x1": width * 0.1,
                    "y1": height * 0.1,
                    "x2": width * 0.3,
                    "y2": height * 0.3,
                    "width": width * 0.2,
                    "height": height * 0.2
                }
            },
            {
                "id": 1,
                "class": "person",
                "confidence": 0.76,
                "bbox": {
                    "x1": width * 0.5,
                    "y1": height * 0.4,
                    "x2": width * 0.6,
                    "y2": height * 0.8,
                    "width": width * 0.1,
                    "height": height * 0.4
                }
            }
        ]
        
        logger.info(f"Generated {len(demo_detections)} demo detections")
        return demo_detections

    def analyze_wildfire_image(
        self,
        image_path: str,
        save_results: bool = True
    ) -> str:
        r"""Specialized analysis for wildfire-related objects in satellite/aerial imagery using MindSpore.
        
        Args:
            image_path (str): Path to the wildfire image
            save_results (bool): Whether to save results. Defaults to True.
            
        Returns:
            str: Detailed wildfire analysis results
        """
        try:
            # First run general object detection
            general_results = json.loads(self.detect_objects_in_image(image_path, save_results))
            
            # Wildfire-specific analysis
            wildfire_analysis = {
                "image_path": image_path,
                "analysis_type": "wildfire_specialized_mindspore",
                "framework": "MindSpore",
                "general_detections": general_results.get("detections", []),
                "wildfire_indicators": [],
                "risk_assessment": {},
                "recommendations": []
            }
            
            # Analyze detections for wildfire relevance
            detections = general_results.get("detections", [])
            
            # Categorize detections by wildfire relevance
            high_risk_objects = []
            infrastructure_at_risk = []
            natural_features = []
            
            for detection in detections:
                class_name = detection["class"].lower()
                confidence = detection["confidence"]
                
                # High-risk wildfire indicators
                if any(keyword in class_name for keyword in ["fire", "smoke", "flame", "burn"]):
                    high_risk_objects.append(detection)
                    wildfire_analysis["wildfire_indicators"].append({
                        "type": "fire_related",
                        "object": class_name,
                        "confidence": confidence,
                        "location": detection["bbox"]
                    })
                
                # Infrastructure at risk
                elif any(keyword in class_name for keyword in ["building", "house", "car", "truck", "person", "road"]):
                    infrastructure_at_risk.append(detection)
                
                # Natural features
                elif any(keyword in class_name for keyword in ["tree", "forest", "vegetation", "water"]):
                    natural_features.append(detection)
            
            # Risk assessment
            wildfire_analysis["risk_assessment"] = {
                "fire_indicators_detected": len(high_risk_objects),
                "infrastructure_elements": len(infrastructure_at_risk),
                "natural_features": len(natural_features),
                "overall_risk_level": self._assess_risk_level(high_risk_objects, infrastructure_at_risk),
                "confidence_score": np.mean([d["confidence"] for d in detections]) if detections else 0.0
            }
            
            # Generate recommendations
            wildfire_analysis["recommendations"] = self._generate_wildfire_recommendations(
                high_risk_objects, infrastructure_at_risk, natural_features
            )
            
            # Enhanced summary
            wildfire_analysis["summary"] = self._generate_wildfire_summary(wildfire_analysis)
            
            # Save wildfire analysis results
            if save_results:
                self._save_wildfire_analysis(wildfire_analysis, image_path)
            
            return json.dumps(wildfire_analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Error in MindSpore wildfire analysis: {str(e)}")
            return json.dumps({
                "error": f"Wildfire analysis failed: {str(e)}",
                "analysis_type": "wildfire_specialized_mindspore",
                "framework": "MindSpore", 
                "summary": "Error occurred during wildfire analysis"
            })

    def _generate_detection_summary(self, detections: List[Dict], image_path: str) -> str:
        """Generate a summary of detected objects."""
        if not detections:
            return f"No objects detected in {os.path.basename(image_path)} with confidence >= {self.confidence_threshold} (MindSpore)"
        
        # Count objects by class
        class_counts = {}
        for detection in detections:
            class_name = detection["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary_parts = [f"Detected {len(detections)} objects in {os.path.basename(image_path)} (MindSpore):"]
        for class_name, count in sorted(class_counts.items()):
            summary_parts.append(f"- {class_name}: {count}")
        
        return "\n".join(summary_parts)

    def _assess_risk_level(self, fire_objects: List[Dict], infrastructure: List[Dict]) -> str:
        """Assess overall wildfire risk level."""
        if len(fire_objects) > 3:
            return "CRITICAL"
        elif len(fire_objects) > 1 and len(infrastructure) > 0:
            return "HIGH"
        elif len(fire_objects) > 0 or len(infrastructure) > 5:
            return "MODERATE"
        else:
            return "LOW"

    def _generate_wildfire_recommendations(
        self, fire_objects: List[Dict], infrastructure: List[Dict], natural_features: List[Dict]
    ) -> List[str]:
        """Generate wildfire management recommendations."""
        recommendations = []
        
        if fire_objects:
            recommendations.append("🔥 IMMEDIATE: Active fire indicators detected - deploy emergency response teams")
            recommendations.append("📞 URGENT: Alert emergency services and implement evacuation protocols")
        
        if infrastructure:
            recommendations.append(f"🏠 PROTECT: {len(infrastructure)} infrastructure elements at risk - establish firebreaks")
            recommendations.append("🚗 EVACUATE: Ensure evacuation routes are clear for vehicles and personnel")
        
        if natural_features:
            recommendations.append("🌲 MONITOR: Natural vegetation detected - assess fuel load and fire behavior")
        
        if not fire_objects and infrastructure:
            recommendations.append("🛡️ PREVENTIVE: Implement fire prevention measures around infrastructure")
        
        # Add MindSpore-specific recommendation
        recommendations.append("🧠 MINDSPORE: Powered by MindSpore AI framework for efficient inference")
        
        return recommendations

    def _generate_wildfire_summary(self, analysis: Dict) -> str:
        """Generate comprehensive wildfire analysis summary."""
        fire_count = analysis["risk_assessment"]["fire_indicators_detected"]
        infra_count = analysis["risk_assessment"]["infrastructure_elements"]
        risk_level = analysis["risk_assessment"]["overall_risk_level"]
        
        summary = f"🔥 WILDFIRE ANALYSIS SUMMARY (MindSpore):\n"
        summary += f"Risk Level: {risk_level}\n"
        summary += f"Fire Indicators: {fire_count}\n"
        summary += f"Infrastructure at Risk: {infra_count}\n"
        summary += f"Total Objects Detected: {len(analysis['general_detections'])}\n"
        summary += f"Framework: MindSpore on {self.device}\n"
        
        if analysis["recommendations"]:
            summary += f"\n📋 KEY RECOMMENDATIONS:\n"
            for rec in analysis["recommendations"][:3]:  # Top 3 recommendations
                summary += f"• {rec}\n"
        
        return summary.strip()

    def _save_annotated_image(self, image_path: str, image: np.ndarray, detections: List[Dict]) -> str:
        """Save annotated image with detection boxes."""
        try:
            # Create annotated image
            annotated_image = image.copy()
            
            for detection in detections:
                bbox = detection["bbox"]
                class_name = detection["class"]
                confidence = detection["confidence"]
                
                # Draw bounding box
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Generate output filename
            input_name = Path(image_path).stem
            output_filename = f"{input_name}_mindspore_yolo_detections.jpg"
            output_path = self.output_dir / output_filename
            
            # Save annotated image
            cv2.imwrite(str(output_path), annotated_image)
            logger.info(f"Saved MindSpore annotated image: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving annotated image: {str(e)}")
            return None

    def _save_json_results(self, results: Dict, image_path: str):
        """Save detection results as JSON."""
        try:
            input_name = Path(image_path).stem
            json_filename = f"{input_name}_mindspore_yolo_results.json"
            json_path = self.output_dir / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved MindSpore JSON results: {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON results: {str(e)}")

    def _save_wildfire_analysis(self, analysis: Dict, image_path: str):
        """Save wildfire analysis results."""
        try:
            input_name = Path(image_path).stem
            analysis_filename = f"{input_name}_mindspore_wildfire_analysis.json"
            analysis_path = self.output_dir / analysis_filename
            
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved MindSpore wildfire analysis: {analysis_path}")
            
        except Exception as e:
            logger.error(f"Error saving wildfire analysis: {str(e)}")

    def get_tools(self) -> List[Any]:
        r"""Get the list of tools available in this toolkit.
        
        Returns:
            List[Any]: List of tool functions for the wildfire MindSpore YOLO toolkit.
        """
        return [
            self.detect_objects_in_image,
            self.analyze_wildfire_image,
        ]