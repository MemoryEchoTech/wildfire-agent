<h1 align="center">
	🔥 Wildfire Agent: AI-Powered Wildfire Management and Analysis System
</h1>

<div align="center">

[![Documentation][docs-image]][docs-url]
[![Package License][package-license-image]][package-license-url]
[![Star][star-image]][star-url]

</div>

<hr>

<div align="center">

🔥 **Wildfire Agent** is an advanced AI system for wildfire disaster management, emergency response, and geospatial analysis. Built on the proven [OWL framework](https://github.com/camel-ai/owl) and [CAMEL-AI](https://github.com/camel-ai/camel) architecture, it provides intelligent assistance for wildfire monitoring, evacuation planning, risk assessment, and post-disaster recovery operations.

</div>

![Wildfire Agent Architecture](./assets/owl_architecture.png)

# 📋 Table of Contents

- [🔥 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [📁 Workspace Management](#-workspace-management)
- [🖼️ Satellite Image Analysis](#️-satellite-image-analysis)
- [🎯 YOLO Object Detection](#-yolo-object-detection)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [🧰 Geospatial Toolkits](#-geospatial-toolkits)
- [🎯 Example Use Cases](#-example-use-cases)
- [🌐 Web Interface](#-web-interface)
- [📄 License](#-license)
- [🖊️ Citation](#️-citation)
- [🤝 Contributing](#-contributing)

# 🔥 Overview

Wildfire Agent addresses critical challenges in wildfire disaster management by providing:

- **Real-time wildfire monitoring** with satellite imagery and remote sensing analysis
- **Emergency evacuation planning** with route optimization and population analysis
- **Risk assessment** for communities, infrastructure, and ecosystems
- **Resource allocation** for firefighting efforts and emergency response
- **Post-disaster damage assessment** and recovery planning

## Core Capabilities

### 🌍 Geospatial Analysis
- Advanced GIS processing with PostGIS, QGIS, and GRASS GIS integration
- Remote sensing image analysis using VLM (Vision Language Models)
- Burn area mapping and fire progression tracking
- Terrain analysis for fire behavior prediction

### 🚨 Emergency Response
- Population evacuation planning and route optimization
- Critical infrastructure vulnerability assessment
- Emergency shelter and resource allocation
- Real-time fire spread modeling and forecasting

### 📊 Data Integration
- Multi-source data fusion (satellite, weather, terrain, demographics)
- API integration for real-time weather and fire data
- Integration with emergency management systems
- Automated report generation and visualization

# ✨ Key Features

- **🛰️ Remote Sensing Integration**: Process and analyze satellite imagery for fire detection and monitoring
- **🗺️ Advanced GIS Capabilities**: Comprehensive geospatial analysis using industry-standard tools
- **🤖 Multi-Agent Architecture**: Specialized agents for different aspects of wildfire management
- **🎯 Emergency Planning**: Automated evacuation route planning and risk assessment
- **📈 Predictive Modeling**: Fire behavior prediction and spread forecasting
- **🌐 Web Interface**: User-friendly interface for emergency managers and analysts
- **🔧 Extensible Framework**: Built on proven OWL/CAMEL architecture for easy customization
- **📁 Organized Workspace**: Automatic file organization with timestamped sessions for all analysis outputs
- **🎯 YOLO Object Detection**: Automated fire, smoke, and infrastructure detection using state-of-the-art computer vision

# 📁 Workspace Management

Wildfire Agent automatically creates organized workspaces for each analysis session, ensuring all generated files are properly categorized and easily accessible.

## Automatic Workspace Structure

Each session creates a timestamped workspace with specialized directories:

```
wildfire_workspace/
├── latest/                    # Symlink to most recent session
└── session_YYYYMMDD_HHMMSS/   # Timestamped session directory
    ├── README.md              # Session documentation
    ├── satellite_imagery/     # Downloaded and processed satellite images
    ├── analysis_results/      # Analysis outputs and calculations
    ├── maps_and_visualizations/ # Generated maps and charts
    ├── code_execution/        # Analysis scripts and code
    ├── documents/             # Reports and documentation
    ├── evacuation_plans/      # Evacuation route planning
    ├── risk_assessments/      # Risk analysis outputs
    └── temp/                  # Temporary files
```

## Benefits

- **Organized Output**: All files automatically saved to appropriate directories
- **Session Tracking**: Timestamped sessions prevent file conflicts
- **Easy Access**: Quick access via `wildfire_workspace/latest` symlink
- **Comprehensive Documentation**: Auto-generated README for each session

# 🖼️ Satellite Image Analysis

Wildfire Agent includes specialized capabilities for analyzing satellite imagery and aerial photographs of wildfire events.

## Maui Wildfire Case Study

The system includes a real-world example using satellite imagery from the 2023 Maui wildfires, showing active fire infrared signatures in Lahaina and surrounding areas.

**Example Analysis Capabilities:**
- Fire hotspot detection and mapping
- Burn area assessment and progression tracking
- Infrastructure damage evaluation
- Evacuation route impact analysis
- Community risk assessment

**Sample Image:** `Maui Wildfires Image.jpg` (in project root)
- Shows infrared satellite view of Maui during active wildfire
- Identifies fire signatures in Lahaina, Kihei, and other areas
- Demonstrates multi-scale analysis from regional to local views

# 🎯 YOLO Object Detection

Wildfire Agent integrates advanced YOLO (You Only Look Once) computer vision models for automated object detection in wildfire scenarios.

## Detection Capabilities

**🔥 Fire & Smoke Detection:**
- Active fire hotspot identification
- Smoke plume detection and tracking
- Burn area boundary mapping
- Fire intensity assessment

**🏠 Infrastructure Analysis:**
- Building and structure identification
- Vehicle and evacuation asset detection
- Critical infrastructure mapping
- Population density estimation

**🌲 Environmental Assessment:**
- Vegetation type classification
- Natural feature identification
- Terrain analysis support
- Water resource detection

## YOLO Integration Features

- **Automated Analysis**: Run `analyze_wildfire_image()` for instant object detection
- **Risk Assessment**: Automatic risk level calculation based on detected objects
- **Smart Recommendations**: Context-aware emergency response suggestions
- **Visual Outputs**: Annotated images with detection boxes and confidence scores
- **Structured Results**: JSON reports with detailed detection metadata

## Usage Examples

```python
# Automated wildfire object detection
"Use YOLO to detect and analyze objects in the Maui wildfire satellite image"

# Infrastructure risk assessment
"Analyze infrastructure elements at risk using YOLO object detection"

# Combined analysis
"Combine YOLO detection with VLM analysis for comprehensive wildfire assessment"
```

# 🛠️ Installation

## Prerequisites

- Python 3.10, 3.11, or 3.12
- GDAL/OGR libraries for geospatial processing
- PyTorch and YOLO dependencies for object detection
- Optional: QGIS, GRASS GIS for advanced analysis

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install gdal-bin python3-gdal qgis grass

# Install system dependencies (macOS with Homebrew)
brew install gdal qgis grass
```

## YOLO Dependencies

For object detection capabilities, install YOLO dependencies in your conda environment:

```bash
# Create and activate conda environment
conda create -n owl python=3.10
conda activate owl

# Install YOLO and computer vision dependencies
conda install ultralytics opencv pytorch torchvision torchaudio -c pytorch -y

# Verify installation
python -c "from ultralytics import YOLO; print('✅ YOLO ready!')"
```

## Installation Options

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/wildfire-agent.git
cd wildfire-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Docker

```bash
# Use Docker for containerized deployment
docker compose up -d

# Run the agent
docker compose exec wildfire-agent bash
```

## Environment Setup

```bash
# Copy environment template
cp .env_template .env

# Configure your API keys in .env
OPENAI_API_KEY="your-openai-api-key"
QWEN_API_KEY="your-qwen-api-key"  # For vision analysis
GOOGLE_MAPS_API_KEY="your-maps-api-key"  # For routing
WEATHER_API_KEY="your-weather-api-key"
```

# 🚀 Quick Start

## Basic Usage

```bash
# Run the specialized wildfire agent with workspace management
python examples/run_qwen_wildfire.py

# Or start the web interface
python owl/webapp_wildfire.py
# Access at http://localhost:7860
```

## Workspace Output

Every run automatically creates an organized workspace:

```
🔥 Wildfire workspace created: /path/to/wildfire_workspace/session_20240624_143022
📁 Access via: /path/to/wildfire_workspace/latest

📊 Workspace structure:
   - satellite_imagery: Downloaded and processed images
   - analysis_results: Analysis outputs and calculations
   - maps_and_visualizations: Generated maps and charts
   - code_execution: Analysis scripts and code
   - documents: Reports and documentation
   - evacuation_plans: Evacuation route planning
   - risk_assessments: Risk analysis outputs
```

## Example Queries

### 🖼️ Satellite Image Analysis

```python
# Analyze the included Maui wildfire satellite image
"Analyze the Maui wildfire satellite image at /Users/kang/GitHub/wildfire-agent/Maui Wildfires Image.jpg. Identify fire hotspots, assess burn areas, and evaluate risks to Lahaina community."

# General image analysis queries
"What can you tell me about the fire progression from this satellite image?"
"Identify all active fire signatures and estimate the total burned area."
"Which communities are most at risk based on the fire locations shown?"
```

### 🎯 YOLO Object Detection

```python
# Automated object detection
"Use YOLO object detection to identify and analyze all objects in the Maui wildfire image."

# Combined YOLO + VLM analysis
"Combine YOLO detection with VLM analysis to provide comprehensive wildfire assessment of the Maui image."

# Infrastructure risk assessment
"Use YOLO to detect infrastructure elements and assess their wildfire risk level."

# Fire detection analysis
"Apply YOLO wildfire detection to identify fire indicators and generate emergency recommendations."
```

### 🔥 Fire Analysis

```python
"How large is the area currently burning in the latest satellite image?"
"What is the rate of fire spread and which direction is it moving?"
"Analyze the burn severity and vegetation loss in this area."
```

### 🚨 Emergency Planning

```python
"How many people need immediate evacuation from the threatened area?"
"What are the safest evacuation routes for the affected communities?"
"Which critical infrastructure is at risk from the current fire?"
```

### 📊 Risk Assessment

```python
"What wildlife habitats and protected areas are impacted?"
"Are water resources contaminated or at risk?"
"Which areas are most vulnerable to future fire spread?"
```

# 🧰 Geospatial Toolkits

Wildfire Agent integrates with powerful geospatial tools:

## Core GIS Tools
- **PostGIS**: Advanced spatial database operations
- **QGIS**: Professional GIS analysis and visualization  
- **GRASS GIS**: Comprehensive geospatial modeling
- **GDAL/OGR**: Geospatial data abstraction and processing
- **Whitebox Geospatial**: Specialized geomorphometric analysis

## Mapping and Visualization
- **GeoPandas**: Python-based spatial data analysis
- **Cartopy**: Advanced cartographic projections
- **Matplotlib**: Statistical plotting and visualization
- **MapClassify**: Choropleth mapping and classification

## Remote Sensing
- **Vision Language Models**: Satellite image analysis and interpretation
- **NDVI/Burn Index**: Vegetation and fire damage assessment
- **Change Detection**: Multi-temporal analysis

# 🎯 Example Use Cases

## 1. Real-time Fire Monitoring
```python
task = """
Analyze the latest MODIS satellite imagery to:
1. Detect active fire hotspots
2. Calculate total burned area
3. Assess fire intensity and behavior
4. Generate a fire status report
"""
```

## 2. Evacuation Planning
```python
task = """
Plan evacuation for communities near the active fire:
1. Identify at-risk populations
2. Calculate optimal evacuation routes
3. Locate emergency shelters
4. Estimate evacuation timeframes
"""
```

## 3. Risk Assessment
```python
task = """
Conduct comprehensive wildfire risk analysis:
1. Assess vulnerability of critical infrastructure
2. Evaluate ecosystem impacts
3. Identify water resource threats
4. Generate risk mitigation recommendations
"""
```

# 🌐 Web Interface

Launch the interactive web interface for visual analysis:

```bash
# Start the web interface
python owl/webapp.py

# Access at http://localhost:7860
```

Features:
- **Interactive Maps**: Visualize fire data and analysis results
- **Real-time Updates**: Live fire monitoring and alerts
- **Report Generation**: Automated analysis reports
- **Multi-user Support**: Collaborative emergency planning

# 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](licenses/LICENSE) for details.

# 🖊️ Citation

If you use Wildfire Agent in your research, please cite:

```bibtex
@software{wildfire_agent_2025,
  title={Wildfire Agent: AI-Powered Wildfire Management and Analysis System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/wildfire-agent}
}
```

Based on the OWL framework:
```bibtex
@article{hu2025owl,
  title={Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation},
  author={Hu, Mengkang and Zhou, Yuhang and Fan, Wendong and Nie, Yuzhou and Xia, Bowei and Sun, Tao and Ye, Ziyu and Jin, Zhaoxuan and Li, Yingru and Chen, Qiguang and others},
  journal={arXiv preprint arXiv:2505.23885},
  year={2025}
}
```

# 🤝 Contributing

We welcome contributions to improve wildfire management capabilities:

1. **Geospatial Tools**: Add new GIS and remote sensing tools
2. **Emergency Models**: Improve evacuation and risk assessment algorithms
3. **Data Sources**: Integrate additional fire and weather data APIs
4. **User Interface**: Enhance the web interface and visualization

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

**🔥 Wildfire Agent** - Protecting communities through intelligent fire management

[docs-image]: https://img.shields.io/badge/Documentation-EB3ECC
[docs-url]: https://github.com/your-username/wildfire-agent
[star-image]: https://img.shields.io/github/stars/your-username/wildfire-agent?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/your-username/wildfire-agent/stargazers
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/your-username/wildfire-agent/blob/main/licenses/LICENSE