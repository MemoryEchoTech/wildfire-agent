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
Wildfire YOLO Detection Toolkit

This toolkit provides specialized YOLO-based object detection capabilities for wildfire analysis,
including fire detection, smoke detection, and burned area identification in satellite and aerial imagery.
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
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)


class WildfireYOLOToolkit(BaseToolkit):
    r"""A toolkit for wildfire detection and analysis using YOLO object detection models.
    
    This toolkit provides specialized functionality for:
    - Fire hotspot detection in satellite imagery
    - Smoke plume identification
    - Burned area assessment
    - Infrastructure risk analysis
    - Emergency response object detection
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        output_dir: str = "./yolo_detections",
        device: str = "auto"
    ):
        r"""Initialize the Wildfire YOLO Toolkit.
        
        Args:
            model_path (str): Path to YOLO model weights. Defaults to 'yolo11n.pt' (nano model).
            confidence_threshold (float): Minimum confidence for detections. Defaults to 0.5.
            output_dir (str): Directory to save detection results. Defaults to './yolo_detections'.
            device (str): Device to run inference on ('auto', 'cpu', 'cuda'). Defaults to 'auto'.
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLO dependencies not available. Please install: "
                "conda install ultralytics opencv pytorch torchvision -y"
            )
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO model
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"YOLO model loaded on device: {self.device}")

    def detect_objects_in_image(
        self,
        image_path: str,
        save_results: bool = True,
        classes_filter: Optional[List[str]] = None
    ) -> str:
        r"""Detect objects in an image using YOLO model.
        
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
            
            # Run YOLO inference
            logger.info(f"Running YOLO detection on: {image_path}")
            results = self.model(image_path, conf=self.confidence_threshold, save=False)
            
            # Process results
            detections = []
            result = results[0]  # Get first result
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    class_name = self.model.names[int(cls_id)]
                    
                    # Apply class filter if specified
                    if classes_filter and class_name not in classes_filter:
                        continue
                    
                    x1, y1, x2, y2 = box
                    detection = {
                        "id": i,
                        "class": class_name,
                        "confidence": float(conf),
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "width": float(x2 - x1),
                            "height": float(y2 - y1)
                        }
                    }
                    detections.append(detection)
            
            # Generate summary
            summary = self._generate_detection_summary(detections, image_path)
            
            # Save annotated image if requested
            output_path = None
            if save_results and detections:
                output_path = self._save_annotated_image(image_path, result)
            
            result_data = {
                "image_path": image_path,
                "model_info": {
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
            logger.error(f"Error in YOLO detection: {str(e)}")
            return json.dumps({
                "error": f"Detection failed: {str(e)}",
                "detections": [],
                "summary": "Error occurred during object detection"
            })

    def analyze_wildfire_image(
        self,
        image_path: str,
        save_results: bool = True
    ) -> str:
        r"""Specialized analysis for wildfire-related objects in satellite/aerial imagery.
        
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
                "analysis_type": "wildfire_specialized",
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
            logger.error(f"Error in wildfire analysis: {str(e)}")
            return json.dumps({
                "error": f"Wildfire analysis failed: {str(e)}",
                "analysis_type": "wildfire_specialized",
                "summary": "Error occurred during wildfire analysis"
            })

    def _generate_detection_summary(self, detections: List[Dict], image_path: str) -> str:
        """Generate a summary of detected objects."""
        if not detections:
            return f"No objects detected in {os.path.basename(image_path)} with confidence >= {self.confidence_threshold}"
        
        # Count objects by class
        class_counts = {}
        for detection in detections:
            class_name = detection["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary_parts = [f"Detected {len(detections)} objects in {os.path.basename(image_path)}:"]
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
        
        return recommendations

    def _generate_wildfire_summary(self, analysis: Dict) -> str:
        """Generate comprehensive wildfire analysis summary."""
        fire_count = analysis["risk_assessment"]["fire_indicators_detected"]
        infra_count = analysis["risk_assessment"]["infrastructure_elements"]
        risk_level = analysis["risk_assessment"]["overall_risk_level"]
        
        summary = f"🔥 WILDFIRE ANALYSIS SUMMARY:\n"
        summary += f"Risk Level: {risk_level}\n"
        summary += f"Fire Indicators: {fire_count}\n"
        summary += f"Infrastructure at Risk: {infra_count}\n"
        summary += f"Total Objects Detected: {len(analysis['general_detections'])}\n"
        
        if analysis["recommendations"]:
            summary += f"\n📋 KEY RECOMMENDATIONS:\n"
            for rec in analysis["recommendations"][:3]:  # Top 3 recommendations
                summary += f"• {rec}\n"
        
        return summary.strip()

    def _save_annotated_image(self, image_path: str, result) -> str:
        """Save annotated image with detection boxes."""
        try:
            # Get annotated image
            annotated_image = result.plot()
            
            # Generate output filename
            input_name = Path(image_path).stem
            output_filename = f"{input_name}_yolo_detections.jpg"
            output_path = self.output_dir / output_filename
            
            # Save annotated image
            cv2.imwrite(str(output_path), annotated_image)
            logger.info(f"Saved annotated image: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving annotated image: {str(e)}")
            return None

    def _save_json_results(self, results: Dict, image_path: str):
        """Save detection results as JSON."""
        try:
            input_name = Path(image_path).stem
            json_filename = f"{input_name}_yolo_results.json"
            json_path = self.output_dir / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved JSON results: {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON results: {str(e)}")

    def _save_wildfire_analysis(self, analysis: Dict, image_path: str):
        """Save wildfire analysis results."""
        try:
            input_name = Path(image_path).stem
            analysis_filename = f"{input_name}_wildfire_analysis.json"
            analysis_path = self.output_dir / analysis_filename
            
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved wildfire analysis: {analysis_path}")
            
        except Exception as e:
            logger.error(f"Error saving wildfire analysis: {str(e)}")

    def get_tools(self) -> List[Any]:
        r"""Get the list of tools available in this toolkit.
        
        Returns:
            List[Any]: List of tool functions for the wildfire YOLO toolkit.
        """
        return [
            self.detect_objects_in_image,
            self.analyze_wildfire_image,
        ]