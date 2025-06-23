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

# 🛠️ Installation

## Prerequisites

- Python 3.10, 3.11, or 3.12
- GDAL/OGR libraries for geospatial processing
- Optional: QGIS, GRASS GIS for advanced analysis

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install gdal-bin python3-gdal qgis grass

# Install system dependencies (macOS with Homebrew)
brew install gdal qgis grass
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
# Run the wildfire agent
python examples/run.py
```

## Example Queries

Try these example queries to see the agent in action:

```python
# Fire analysis queries
"How large is the area currently burning in the latest satellite image?"
"What is the rate of fire spread and which direction is it moving?"
"Analyze the burn severity and vegetation loss in this area."

# Emergency planning queries  
"How many people need immediate evacuation from the threatened area?"
"What are the safest evacuation routes for the affected communities?"
"Which critical infrastructure is at risk from the current fire?"

# Risk assessment queries
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