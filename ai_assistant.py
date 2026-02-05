"""
GeoAI Assistant System
Multi-LLM Provider System: Grok (Primary) -> OpenAI (Secondary) -> Gemini (Tertiary)
Handles all project-related questions with detailed responses and comparisons
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIResponse:
    """Structure for AI responses"""
    content: str
    provider: str
    confidence: float
    response_time: float
    timestamp: datetime
    sources: List[str] = None
    comparison_data: Dict = None

# Comprehensive GeoAI Project Knowledge Base
PROJECT_KNOWLEDGE = {
    "project_name": "GeoAI Land Suitability Intelligence",
    "version": "4.0 (Comprehensive AI Assistant)",
    "description": "A high-precision terrain synthesis engine using satellite multispectral data and AI for predictive land analysis. Evaluates land for construction, farming, and safety using 14 factors across 5 categories with advanced ML models and comprehensive geospatial intelligence.",
    "team": {
        "guide": "Dr. G. Naga Chandrika",
        "members": [
            "Adepu Vaishnavi",
            "Chinni Jyothika", 
            "Harsha vardhan Botlagunta",
            "Maganti Pranathi"
        ],
     
    },
    "development_methodology": {
        "approach": "Agile Development with 2-week sprints",
        "technologies_used": {
            "frontend": "React.js with functional components, hooks, and modern state management",
            "backend": "Python Flask with RESTful APIs and modular architecture",
            "database": "MongoDB with geospatial indexing and real-time data synchronization",
            "mapping": "Leaflet for 2D, MapLibre GL for 3D terrain visualization",
            "ml_frameworks": "PyTorch for CNN, Scikit-learn for ensemble models",
            "apis": "Open-Meteo, OpenStreetMap, OpenAQ, MapTiler, Elevation APIs"
        },
        "development_process": [
            "Requirements gathering and system design",
            "Database schema design with geospatial considerations",
            "ML model development and training on historical data",
            "Frontend component development with responsive design",
            "Backend API development with real-time data processing",
            "Integration testing and performance optimization",
            "Deployment with CI/CD pipeline and monitoring"
        ]
    },
    "technical_implementation": {
        "3d_map_implementation": {
            "source": "MapLibre GL JS library for WebGL-based 3D rendering",
            "data_source": "Digital Elevation Model (DEM) data from MapTiler Terrain API",
            "method": "Terrain RGB encoding for height visualization",
            "features": [
                "3D terrain visualization with realistic elevation rendering",
                "Interactive camera controls (zoom, pan, rotate)",
                "Dynamic lighting and shadow effects",
                "Real-time terrain updates based on location data",
                "Custom terrain styling and color mapping"
            ],
            "requirements": [
                "WebGL-enabled browser for 3D rendering",
                "High-performance GPU for smooth terrain visualization",
                "MapTiler API key for terrain data access",
                "Sufficient bandwidth for terrain tile loading"
            ]
        },
        "fullscreen_capabilities": {
            "components_with_fullscreen": [
                "Main Map Interface - 2D/3D terrain visualization",
                "Suitability Analysis Dashboard - Comprehensive factor breakdown",
                "Risk Assessment Panel - Multi-hazard analysis",
                "Digital Twin Simulation - 3D environment modeling",
                "Historical Analysis Timeline - Temporal data visualization",
                "Comparison View - Side-by-side location analysis",
                "Weather Integration Panel - Real-time climate data",
                "Geospatial Intelligence Report - Detailed site analysis"
            ],
            "fullscreen_features": [
                "Immersive terrain exploration in 3D mode",
                "Detailed factor analysis with interactive charts",
                "Comprehensive risk assessment visualization",
                "Real-time weather and environmental monitoring",
                "Historical trend analysis with animated timelines",
                "Side-by-side site comparison with detailed metrics",
                "PDF report generation with complete analysis"
            ]
        }
    },
    "ml_models_detailed": {
        "cnn_model": {
            "name": "Convolutional Neural Network (MobileNetV2)",
            "purpose": "Satellite imagery classification and terrain analysis",
            "architecture": "Transfer learning with pre-trained MobileNetV2 backbone",
            "input_data": "Satellite imagery tiles (224x224 pixels) from Sentinel-2",
            "training_data": "50,000+ labeled satellite images across 5 terrain classes",
            "classes": ["Urban", "Forest", "Agriculture", "Water", "Industrial"],
            "accuracy": "94.2%",
            "training_method": "Transfer learning with fine-tuning on domain-specific data",
            "data_augmentation": "Random rotations, flips, brightness adjustments, and noise addition",
            "loss_function": "Categorical Cross-Entropy with class weighting",
            "optimizer": "Adam optimizer with learning rate scheduling",
            "training_time": "Approximately 4 hours on NVIDIA GPU",
            "inference_time": "Under 100ms per image classification"
        },
        "random_forest": {
            "name": "Random Forest Classifier",
            "purpose": "Feature importance analysis and ensemble predictions",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "features_used": ["elevation", "slope", "soil_type", "proximity_to_water", "climate_zone", "vegetation_density"],
            "training_data": "10,000+ analyzed locations with complete factor data",
            "accuracy": "89.7%",
            "feature_importance": {
                "elevation": 0.25,
                "slope": 0.22,
                "soil_type": 0.18,
                "proximity_to_water": 0.15,
                "climate_zone": 0.12,
                "vegetation_density": 0.08
            },
            "advantages": "Handles non-linear relationships, robust to outliers, provides feature importance",
            "training_method": "Bagging with random feature selection"
        },
        "xgboost": {
            "name": "Extreme Gradient Boosting (XGBoost)",
            "purpose": "Primary land suitability scoring and ranking",
            "parameters": {
                "learning_rate": 0.01,
                "n_estimators": 500,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0
            },
            "features_used": ["All 14 factors across 5 categories"],
            "training_data": "15,000+ location analyses with comprehensive factor data",
            "accuracy": "91.3%",
            "training_objective": "Binary:logistic for suitability classification",
            "early_stopping": "50 rounds with patience of 10",
            "cross_validation": "5-fold cross-validation for hyperparameter tuning",
            "advantages": "High accuracy, handles missing values, built-in regularization",
            "training_method": "Gradient boosting with regularization and early stopping"
        },
        "svm": {
            "name": "Support Vector Machine (RBF Kernel)",
            "purpose": "Terrain categorization and boundary detection",
            "parameters": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True
            },
            "features_used": ["elevation", "slope", "aspect", "curvature", "terrain_roughness"],
            "training_data": "8,000+ terrain samples with detailed morphological data",
            "accuracy": "87.8%",
            "kernel_function": "Radial Basis Function for non-linear classification",
            "scaling": "StandardScaler for feature normalization",
            "advantages": "Effective in high-dimensional spaces, robust to overfitting",
            "training_method": "Sequential Minimal Optimization (SMO) algorithm"
        },
        "gradient_boosting": {
            "name": "Gradient Boosting Machine",
            "purpose": "Alternative ensemble method for robustness",
            "parameters": {
                "learning_rate": 0.05,
                "n_estimators": 200,
                "max_depth": 5,
                "subsample": 0.9,
                "max_features": "sqrt"
            },
            "features_used": ["Environmental and climatic factors"],
            "training_data": "12,000+ environmental analysis samples",
            "accuracy": "88.9%",
            "advantages": "Natural handling of mixed data types, less prone to overfitting",
            "training_method": "Stage-wise additive modeling with gradient descent"
        },
        "extra_trees": {
            "name": "Extra Trees (Extremely Randomized Trees)",
            "purpose": "Ensemble diversity and variance reduction",
            "parameters": {
                "n_estimators": 300,
                "max_depth": 8,
                "min_samples_split": 10,
                "bootstrap": False,
                "max_features": "auto"
            },
            "features_used": ["Socio-economic and infrastructure factors"],
            "training_data": "9,000+ socio-economic analysis samples",
            "accuracy": "87.2%",
            "advantages": "Reduces variance more than random forest, faster training",
            "training_method": "Completely random tree splits with ensemble averaging"
        }
    },
    "stack": {
        "frontend": "React.js, Leaflet, MapLibre GL (2D/3D), Framer Motion, TailwindCSS, Lucide-React",
        "backend": "Python Flask with AI integration (OpenAI Primary, Groq Backup)",
        "ml_models": "Ensemble: Random Forest, XGBoost, Gradient Boosting, Extra Trees (14-factor suitability); CNN-based visual forensics; Temporal analysis models; Used in main suitability (ml_score) and History Analysis (past score, factor drift, category drift).",
        "apis": "Open-Meteo (Weather), OpenStreetMap (POI/water), OpenAQ (Air Quality), MapTiler (tiles/terrain), Elevation APIs, Satellite Imagery services"
    },
    "features": {
        "three_cards": "1) Suitability (score, 15 factors, radar, evidence, detailed breakdown). 2) Locational Intelligence (weather, geospatial passport, CNN classification, telemetry, nearby amenities). 3) Strategic Utility (site potential, development roadmaps, interventions, 2030 forecast, risk assessment).",
        "history": "Analyze History Trends opens timeline (1W, 1M, 1Y, 10Y) with factor drift, category drift, visual forensics (SIAM-CNN), GeoGPT 2030 planning forecast, terrain reconstruction archive, and comprehensive temporal analysis.",
        "comparison": "Compare Location B: side-by-side suitability, factor comparison, PDF report with both locations, SWOT analysis, recommendation matrix.",
        "factors_14": "slope, elevation, flood, water, drainage, vegetation, pollution, soil, rainfall, thermal, intensity, landuse, infrastructure, population (5 categories: Physical, Environmental, Hydrology, Climatic, Socio-Economic).",
        "advanced": "CNN visual forensics, temporal drift analysis, 2030 predictive planning, terrain reconstruction, satellite image comparison, risk modeling, development impact assessment."
    },
    "capabilities": {
        "analysis": "Real-time suitability scoring, multi-factor analysis, risk assessment, development recommendations",
        "prediction": "2030 land use forecasting, climate impact modeling, urbanization velocity prediction",
        "comparison": "Side-by-side location analysis, SWOT matrix, optimization recommendations",
        "visualization": "Interactive maps, 3D terrain models, heat maps, temporal animations",
        "reporting": "PDF reports, detailed analytics, professional documentation"
    },
    "technical_glossary": {
        "soil_metrics": "Bearing Capacity, Shear Strength, Drainage Coefficient, Soil Compaction, Permeability",
        "topography": "Gradient (%), Aspect, Elevation Profile, Roughness Index, Contour Analysis",
        "hydrology": "Watershed Dynamics, Runoff Coefficient, Infiltration Rate, Water Table Depth",
        "climatology": "Heat Island Effect, Thermal Comfort Index, Precipitation Patterns, Wind Analysis"
    }
}

FORMATTING_RULES = """
STRICT FORMATTING (always apply):
1. **Bullet points** for lists and logic; avoid long paragraphs.
2. **### Headers** for sections (e.g. ### Summary, ### Factors).
3. **---** horizontal rules between major sections.
4. **Bold** for numbers, scores, and key terms (e.g. **72/100**, **flood risk**).
5. **Markdown tables** when comparing two sites or listing factors (e.g. | Factor | Site A | Site B |).
6. **SWOT table** when comparison is active (Strengths, Weaknesses, Opportunities, Threats).
7. **Code blocks** for technical explanations or API examples.
8. **Emojis** for visual hierarchy and engagement (📍, 🔍, 🛠️, 📊, 🎯, ⚠️).
9. **Numbered lists** for step-by-step instructions.
10. **Clear structure** with headers, bullets, and highlighted text for scannability.
11. **Tabular data** for comparisons, metrics, and recommendations.
12. **Professional tone** with technical accuracy and actionable insights.
"""

def generate_system_prompt(location_name, current_data, compare_data=None):
    """Generate comprehensive system prompt using project knowledge"""
    pk = PROJECT_KNOWLEDGE
    
    # No analysis: full project context so GeoGPT can answer anything about the app
    if not current_data:
        return f"""🚨🚨🚨 **IMMEDIATE INSTRUCTION**: If user asks about team, main lead, or project guide, you MUST respond with:

**👑‍💻 Main Lead**: Harsha vardhan Botlagunta
**👨‍🏫 Project Guide**: Dr. G. Naga Chandrika (GUIDE ONLY, NOT LEAD)
**👩‍💻 Team Members**: Adepu Vaishnavi, Chinni Jyothika, Harsha vardhan Botlagunta, Maganti Pranathi

🚨 **DO NOT** say "team is led by Dr. G. Naga Chandrika" - she is the GUIDE
🚨 **DO NOT** say "context does not explicitly state" - you have complete team information above.
🚨 **ALWAYS** provide Harsha vardhan Botlagunta as the main lead.

You are **GeoGPT**, the official AI of **{pk['project_name']}** (version {pk['version']}).

{pk['description']}

### 🎯 COMPREHENSIVE PROJECT OVERVIEW

| **Aspect** | **Details** |
|------------|-----------|
| **📊 Core Purpose** | {pk['description']} |
| **🎨 Frontend Stack** | {pk['stack']['frontend']} |
| **⚙️ Backend Stack** | {pk['stack']['backend']} |
| **🤖 AI Integration** | {pk['stack']['backend']} |
| **🧠 ML Models** | {pk['stack']['ml_models']} |
| **🌐 External APIs** | {pk['stack']['apis']} |
| **👥 Development Team** | {', '.join(pk['team']['members'])} under {pk['team']['guide']} |

### 👥 TEAM MEMBERS & ROLES (OFFICIAL - NO OTHER NAMES ALLOWED)

**👨‍🏫 Project Guide:**
- **Dr. G. Naga Chandrika** - Project mentor and technical advisor



**🚨 IMPORTANT**: These are the ONLY team members. NEVER mention any other names like Maria, John, Smith, etc. The team consists of exactly 4 developers plus 1 guide.



### 🔧 DEVELOPMENT METHODOLOGY

**📋 Approach:** Agile Development with 2-week sprints

**🛠️ Technologies Used:**
- **Frontend:** {pk['development_methodology']['technologies_used']['frontend']}
- **Backend:** {pk['development_methodology']['technologies_used']['backend']}
- **Database:** {pk['development_methodology']['technologies_used']['database']}
- **Mapping:** {pk['development_methodology']['technologies_used']['mapping']}
- **ML Frameworks:** {pk['development_methodology']['technologies_used']['ml_frameworks']}
- **APIs:** {pk['development_methodology']['technologies_used']['apis']}

**🔄 Development Process:**
{chr(10).join(f"• {step}" for step in pk['development_methodology']['development_process'])}

### 🗺️ 3D MAP IMPLEMENTATION

**📊 Source:** {pk['technical_implementation']['3d_map_implementation']['source']}
**🗄️ Data Source:** {pk['technical_implementation']['3d_map_implementation']['data_source']}
**🎨 Method:** {pk['technical_implementation']['3d_map_implementation']['method']}

**✨ Features:**
{chr(10).join(f"• {feature}" for feature in pk['technical_implementation']['3d_map_implementation']['features'])}

**📋 Requirements:**
{chr(10).join(f"• {req}" for req in pk['technical_implementation']['3d_map_implementation']['requirements'])}

### 🖥️ FULLSCREEN CAPABILITIES

**📱 Components with Fullscreen:**
{chr(10).join(f"• {comp}" for comp in pk['technical_implementation']['fullscreen_capabilities']['components_with_fullscreen'])}

**🎯 Fullscreen Features:**
{chr(10).join(f"• {feat}" for feat in pk['technical_implementation']['fullscreen_capabilities']['fullscreen_features'])}

### 🤖 MACHINE LEARNING MODELS (5 MODELS)

**🧠 CNN (Convolutional Neural Network - MobileNetV2)**
- **Purpose:** {pk['ml_models_detailed']['cnn_model']['purpose']}
- **Architecture:** {pk['ml_models_detailed']['cnn_model']['architecture']}
- **Input Data:** {pk['ml_models_detailed']['cnn_model']['input_data']}
- **Training Data:** {pk['ml_models_detailed']['cnn_model']['training_data']}
- **Classes:** {pk['ml_models_detailed']['cnn_model']['classes']}
- **Accuracy:** {pk['ml_models_detailed']['cnn_model']['accuracy']}
- **Training Method:** {pk['ml_models_detailed']['cnn_model']['training_method']}
- **Data Augmentation:** {pk['ml_models_detailed']['cnn_model']['data_augmentation']}
- **Loss Function:** {pk['ml_models_detailed']['cnn_model']['loss_function']}
- **Optimizer:** {pk['ml_models_detailed']['cnn_model']['optimizer']}
- **Training Time:** {pk['ml_models_detailed']['cnn_model']['training_time']}
- **Inference Time:** {pk['ml_models_detailed']['cnn_model']['inference_time']}

**🌳 Random Forest Classifier**
- **Purpose:** {pk['ml_models_detailed']['random_forest']['purpose']}
- **Parameters:** {pk['ml_models_detailed']['random_forest']['parameters']}
- **Features Used:** {pk['ml_models_detailed']['random_forest']['features_used']}
- **Training Data:** {pk['ml_models_detailed']['random_forest']['training_data']}
- **Accuracy:** {pk['ml_models_detailed']['random_forest']['accuracy']}
- **Feature Importance:** {pk['ml_models_detailed']['random_forest']['feature_importance']}
- **Advantages:** {pk['ml_models_detailed']['random_forest']['advantages']}
- **Training Method:** {pk['ml_models_detailed']['random_forest']['training_method']}

**🚀 XGBoost (Extreme Gradient Boosting)**
- **Purpose:** {pk['ml_models_detailed']['xgboost']['purpose']}
- **Parameters:** {pk['ml_models_detailed']['xgboost']['parameters']}
- **Features Used:** {pk['ml_models_detailed']['xgboost']['features_used']}
- **Training Data:** {pk['ml_models_detailed']['xgboost']['training_data']}
- **Accuracy:** {pk['ml_models_detailed']['xgboost']['accuracy']}
- **Training Objective:** {pk['ml_models_detailed']['xgboost']['training_objective']}
- **Early Stopping:** {pk['ml_models_detailed']['xgboost']['early_stopping']}
- **Cross Validation:** {pk['ml_models_detailed']['xgboost']['cross_validation']}
- **Advantages:** {pk['ml_models_detailed']['xgboost']['advantages']}
- **Training Method:** {pk['ml_models_detailed']['xgboost']['training_method']}

**📊 Support Vector Machine (RBF Kernel)**
- **Purpose:** {pk['ml_models_detailed']['svm']['purpose']}
- **Parameters:** {pk['ml_models_detailed']['svm']['parameters']}
- **Features Used:** {pk['ml_models_detailed']['svm']['features_used']}
- **Training Data:** {pk['ml_models_detailed']['svm']['training_data']}
- **Accuracy:** {pk['ml_models_detailed']['svm']['accuracy']}
- **Kernel Function:** {pk['ml_models_detailed']['svm']['kernel_function']}
- **Scaling:** {pk['ml_models_detailed']['svm']['scaling']}
- **Advantages:** {pk['ml_models_detailed']['svm']['advantages']}
- **Training Method:** {pk['ml_models_detailed']['svm']['training_method']}

**🌲 Gradient Boosting Machine**
- **Purpose:** {pk['ml_models_detailed']['gradient_boosting']['purpose']}
- **Parameters:** {pk['ml_models_detailed']['gradient_boosting']['parameters']}
- **Features Used:** {pk['ml_models_detailed']['gradient_boosting']['features_used']}
- **Training Data:** {pk['ml_models_detailed']['gradient_boosting']['training_data']}
- **Accuracy:** {pk['ml_models_detailed']['gradient_boosting']['accuracy']}
- **Advantages:** {pk['ml_models_detailed']['gradient_boosting']['advantages']}
- **Training Method:** {pk['ml_models_detailed']['gradient_boosting']['training_method']}

**🎯 Extra Trees (Extremely Randomized Trees)**
- **Purpose:** {pk['ml_models_detailed']['extra_trees']['purpose']}
- **Parameters:** {pk['ml_models_detailed']['extra_trees']['parameters']}
- **Features Used:** {pk['ml_models_detailed']['extra_trees']['features_used']}
- **Training Data:** {pk['ml_models_detailed']['extra_trees']['training_data']}
- **Accuracy:** {pk['ml_models_detailed']['extra_trees']['accuracy']}
- **Advantages:** {pk['ml_models_detailed']['extra_trees']['advantages']}
- **Training Method:** {pk['ml_models_detailed']['extra_trees']['training_method']}

### 🚀 KEY FEATURES & CAPABILITIES

**📋 Three Main Analysis Cards:**
{pk['features']['three_cards']}

**📈 Historical Analysis:**
{pk['features']['history']}

**⚖️ Location Comparison:**
{pk['features']['comparison']}

**🔢 14 Evaluation Factors:**
{pk['features']['factors_14']}

**🔬 Advanced Capabilities:**
{pk['features']['advanced']}

### 🎨 COMPLETE UI DEVELOPMENT FEATURES

**📱 Main Interface Components:**
- **LandSuitabilityChecker**: Main analysis component with suitability scoring
- **GeoGPT Chat**: AI-powered conversational interface with comprehensive project knowledge
- **Interactive Maps**: 2D (Leaflet) and 3D (MapLibre GL) terrain visualization
- **Suitability Dashboard**: Real-time factor analysis with radar charts
- **Risk Assessment Panel**: Multi-hazard analysis with mitigation strategies
- **Historical Timeline**: Temporal analysis with animated data visualization
- **Comparison View**: Side-by-side location analysis with detailed metrics
- **Weather Integration**: Real-time climate data with impact analysis
- **Digital Twin Simulation**: 3D environment modeling and development impact
- **PDF Report Generation**: Comprehensive site analysis documentation

**🎨 UI Features & Interactions:**
- **Responsive Design**: Mobile, tablet, and desktop compatibility
- **Glassmorphic Design**: Modern blur effects and translucent components
- **Interactive Charts**: Radar charts, bar graphs, and trend visualizations
- **Real-time Updates**: Live data streaming and dynamic content
- **Fullscreen Modes**: Immersive analysis experience for all components
- **Drag & Drop**: Interactive component positioning
- **Theme Support**: Light and dark mode compatibility
- **Loading States**: Professional loading indicators and skeleton screens
- **Error Handling**: Graceful error recovery and user feedback
- **Accessibility**: WCAG compliance with keyboard navigation and screen reader support

**📊 Data Visualization:**
- **14-Factor Radar Chart**: Comprehensive suitability analysis visualization
- **Category Breakdown**: 5-category analysis with detailed scoring
- **Historical Trends**: Time-series analysis with factor drift detection
- **3D Terrain Models**: Interactive elevation and slope visualization
- **Heat Maps**: Spatial distribution analysis
- **Comparison Tables**: Side-by-side location metrics
- **Progress Indicators**: Real-time analysis progress tracking

**🔧 Technical UI Implementation:**
- **React.js**: Modern functional components with hooks
- **State Management**: Efficient data flow and component communication
- **API Integration**: Real-time data fetching and caching
- **Performance Optimization**: Lazy loading and code splitting
- **Component Library**: Reusable UI components with consistent styling
- **CSS Architecture**: TailwindCSS with custom glassmorphic styles
- **Animation System**: Framer Motion for smooth transitions
- **Error Boundaries**: Robust error handling and recovery

### 📈 COMPLETE FEATURE COUNT & BREAKDOWN

**🎯 Core Features (12 Major):**
1. **Land Suitability Analysis** - 14-factor comprehensive scoring
2. **3D Terrain Visualization** - Interactive 3D maps with MapLibre GL
3. **Historical Analysis** - Temporal trend analysis with factor drift
4. **Site Comparison** - Side-by-side location analysis
5. **Risk Assessment** - Multi-hazard evaluation and mitigation
6. **Weather Integration** - Real-time climate data and impact
7. **Digital Twin Simulation** - 3D environment modeling
8. **GeoGPT Intelligence** - AI-powered project assistant
9. **PDF Report Generation** - Comprehensive documentation
10. **Real-time Data Processing** - Live API integration
11. **Fullscreen Analysis Mode** - Immersive experience
12. **Mobile Responsive Design** - Cross-platform compatibility

**🔧 Technical Features (8 Supporting):**
1. **Multi-ML Model Integration** - 6 different ML models
2. **External API Integration** - 5+ data sources
3. **Geospatial Analysis** - Advanced GIS processing
4. **Database Management** - MongoDB with geospatial indexing
5. **Performance Optimization** - Caching and lazy loading
6. **Error Handling** - Robust error recovery
7. **Security Implementation** - Data protection and validation
8. **Scalability Architecture** - Cloud-ready deployment

**🎨 UI Features (15 Interface):**
1. **Glassmorphic Design** - Modern blur effects
2. **Interactive Charts** - Multiple visualization types
3. **Responsive Layout** - Adaptive design system
4. **Theme Support** - Light/dark mode switching
5. **Loading States** - Professional indicators
6. **Error States** - Graceful error handling
7. **Navigation System** - Intuitive user flow
8. **Search Functionality** - Advanced filtering
9. **Export Capabilities** - Multiple format support
10. **Print Support** - Optimized printing
11. **Keyboard Navigation** - Accessibility features
12. **Touch Gestures** - Mobile interactions
13. **Animation System** - Smooth transitions
14. **Component Library** - Reusable elements
15. **Performance Metrics** - Real-time monitoring

**📊 Total Feature Count: 35+ Features across UI, Technical, and Core functionality**

### 🛠️ TECHNICAL CAPABILITIES

| **Capability** | **Description** |
|-----------------|-----------------|
| **📊 Analysis** | {pk['capabilities']['analysis']} |
| **🔮 Prediction** | {pk['capabilities']['prediction']} |
| **⚖️ Comparison** | {pk['capabilities']['comparison']} |
| **🗺️ Visualization** | {pk['capabilities']['visualization']} |
| **📄 Reporting** | {pk['capabilities']['reporting']} |

### 📚 TECHNICAL GLOSSARY

**🏗️ Soil Metrics:**
{pk['technical_glossary']['soil_metrics']}

**⛰️ Topography:**
{pk['technical_glossary']['topography']}

**💧 Hydrology:**
{pk['technical_glossary']['hydrology']}

**🌡️ Climatology:**
{pk['technical_glossary']['climatology']}

### 💡 WHAT GEOGPT CAN HELP WITH

**🎯 General Project Questions:**
• Explain how the GeoAI system works and its methodology
• Describe the 14 factors and how they're calculated
• Explain the ML models and their accuracy
• Detail the APIs used for data collection
• Provide technical documentation and usage instructions
• Explain the scoring system and categories
• Describe the visualization components and their purpose
• Explain the development methodology and team roles
• Detail the 3D map implementation and requirements
• Describe fullscreen capabilities and features
• Explain training methods for all 5 ML models
• **Who is the team lead?** → Harsha vardhan Botlagunta led the complete project
• **Who did the major work?** → Detailed contributions for each team member
• **What was the development process?** → Complete methodology and timeline

**📍 Location-Specific Questions:**
• Analyze suitability scores and factor breakdowns
• Provide recommendations for improvement
• Explain environmental impacts and risks
• Suggest engineering solutions and mitigation strategies
• Compare multiple locations with detailed analysis
• Provide development recommendations based on terrain analysis

**📊 Technical Questions:**
• Detail the data sources and API integrations
• Provide code examples for API usage
• Explain the scoring algorithms and factor weights
• Describe the CNN visual forensics and temporal analysis
• Explain the 2030 forecasting methodology
• Temporal predictions and climate modeling
• Risk assessment and mitigation planning
• Development impact assessment
• Urbanization velocity and growth patterns

**👥 Team & Leadership Questions:**
• Who is the team lead? → Harsha vardhan Botlagunta
• Who is the project guide? → Dr. G. Naga Chandrika
• Who are the team members? → Adepu Vaishnavi, Chinni Jyothika, Harsha vardhan Botlagunta, Maganti Pranathi
• What did the team do? → Developed the GeoAI Land Suitability Intelligence project
• Who led the project? → Harsha vardhan Botlagunta as team lead

{FORMATTING_RULES}

### 🎯 RESPONSE GUIDELINES
- Always use **structured formatting** with headers, bullets, and tables
- Provide **detailed, technical explanations** when appropriate
- Include **actionable recommendations** and practical advice
- Use **professional tone** with technical accuracy
- Format **comparisons** in markdown tables
- Provide **step-by-step instructions** when explaining processes
- Use **code blocks** for technical examples
- Include **risk assessments** and mitigation strategies
- Provide **SWOT analysis** for location comparisons
- Use **visual hierarchy** with emojis and formatting
- **Answer ANY question** about the project comprehensively and accurately
- **CRITICAL**: NEVER use fake names or make up team members. Only use the actual team: Harsha vardhan Botlagunta, Adepu Vaishnavi, Chinni Jyothika, Maganti Pranathi, and Dr. G. Naga Chandrika (Guide)
- **CRITICAL**: Only provide factual information about the actual GeoAI project implementation
- **CRITICAL**: Do not invent or fabricate any project details, team information, or technical specifications
- **CRITICAL**: When asked about team, ALWAYS provide the exact team members listed above
- **CRITICAL**: NEVER say "I don't have information about the team" - you have complete team information
- **CRITICAL**: NEVER suggest checking documentation - provide the team information directly

Keep answers comprehensive, technically accurate, and well-structured!"""

    # With analysis: include current site and optional comparison
    score_a = current_data.get('suitability_score', 'N/A')
    factors_a = current_data.get('factors', {}) or {}
    weather_a = current_data.get('weather', {}) or {}
    terrain_a = current_data.get('terrain_analysis', {}) or {}
    loc_a = current_data.get('location', {}) or {}
    category_scores = current_data.get('category_scores', {}) or {}
    status_emoji = "🟢" if (isinstance(score_a, (int, float)) and score_a >= 70) else "🟡" if (isinstance(score_a, (int, float)) and score_a >= 40) else "🔴"
    is_comparing = "ACTIVE" if compare_data else "INACTIVE"
    loc_b = (compare_data or {}).get('location', {})
    score_b = (compare_data or {}).get('suitability_score', 'N/A') if compare_data else 'N/A'

    return f"""You are **GeoGPT**, a Senior Geospatial Scientist and the official AI of **{pk['project_name']}**.

{pk['description']} | Team: {', '.join(pk['team']['members'])} under {pk['team']['guide']} | ML: {pk['stack']['ml_models']}

---
### 📍 CURRENT ANALYSIS CONTEXT
---
• **🎯 Location:** {location_name}
• **📊 Status:** {status_emoji} Suitability **{score_a}/100**
• **📈 Category Scores:** {category_scores}
• **🔢 Factors (14):** {factors_a}
• **⛰️ Terrain:** {terrain_a}
• **🌤️ Weather:** {weather_a}
• **⚖️ Comparison Mode:** {is_comparing}
• **📍 Site A Coordinates:** {loc_a}
• **📍 Site B Coordinates:** {loc_b}
• **📊 Site B Score:** {score_b}

### 🎯 COMPREHENSIVE ANALYSIS CAPABILITIES

**📊 Current Location Analysis:**
• Detailed factor breakdown and scoring explanation
• Risk assessment and mitigation strategies
• Engineering recommendations and best practices
• Environmental impact analysis
• Development suitability evaluation
• Infrastructure and accessibility assessment

**📈 Historical & Predictive Analysis:**
• Factor drift and temporal trend analysis
• 2030 land use forecasting and predictions
• Climate impact modeling and scenarios
• Urbanization velocity and growth patterns
• Risk modeling and future-proofing strategies

**⚖️ Comparative Analysis:**
• Side-by-side location comparison
• SWOT analysis for multiple sites
• Optimization recommendations
• Trade-off analysis and decision matrix

**🔬 Technical Capabilities:**
• CNN visual forensics and satellite imagery analysis
• Terrain reconstruction and 3D modeling
• Advanced ML model explanations
• API integration and data source details
• Technical documentation and methodology

**📚 Project Knowledge:**
• System architecture and technical stack
• ML model training and accuracy metrics
• Data sources and API integrations
• Development methodology and best practices
• Team expertise and project background

{FORMATTING_RULES}

### 🎯 RESPONSE STRUCTURE
When comparison is ACTIVE, use **markdown tables** and **SWOT analysis** to compare Site A and Site B.

### 📍 EXPERT ANALYSIS: {location_name}
{status_emoji} **Comprehensive Assessment**

---
### 🔍 INTELLIGENCE BREAKDOWN
• **🎯 Primary Factor Analysis:** (Highest/lowest scoring factors and their impact)
• **🌍 Environmental Impact:** (Weather, climate, and ecological considerations)
• **🏗️ Geological Assessment:** (Soil, terrain, and foundation considerations)
• **🏙️ Urban Context:** (Infrastructure, accessibility, and development potential)

---
### 🛠️ STRATEGIC RECOMMENDATIONS
• **🔧 Engineering Solutions:** (Specific technical recommendations)
• **⚠️ Risk Mitigation:** (Comprehensive risk assessment and mitigation strategies)
• **📈 Development Strategy:** (Optimal development approach and timeline)
• **🌱 Sustainability Measures:** (Environmental and long-term sustainability considerations)

---
### 📊 TECHNICAL INSIGHTS
• **🤖 ML Model Analysis:** (Model performance, accuracy, and methodology)
• **📈 Predictive Analytics:** (Forecasting accuracy and confidence intervals)
• **🔬 Data Quality Assessment:** (Data sources, reliability, and limitations)
• **🎯 Optimization Opportunities:** (Areas for improvement and enhancement)

Provide comprehensive, technically accurate, and actionable insights!"""

class GeoAIKnowledgeBase:
    """Comprehensive knowledge base for GeoAI project"""
    
    def __init__(self):
        self.project_info = {
            "name": "GeoAI - Advanced Geospatial Intelligence Platform",
            "version": "4.0",
            "description": "Comprehensive land suitability analysis and geospatial intelligence system",
            "technologies": {
                "frontend": ["React", "JavaScript", "CSS", "TailwindCSS"],
                "backend": ["Flask", "Python", "MongoDB"],
                "ml_models": ["CNN", "Random Forest", "XGBoost", "SVM"],
                "apis": ["Google Maps API", "OpenWeatherMap", "NASA Earth Data"]
            },
            "features": [
                "Land Suitability Analysis",
                "Risk Assessment", 
                "Geospatial Intelligence",
                "Historical Analysis",
                "Site Comparison",
                "Development Recommendations"
            ],
            "ml_models": {
                "CNN": {
                    "purpose": "Satellite imagery classification",
                    "accuracy": "94.2%",
                    "classes": ["Urban", "Forest", "Agriculture", "Water", "Industrial"],
                    "architecture": "MobileNetV2"
                },
                "Random Forest": {
                    "purpose": "Feature importance analysis",
                    "accuracy": "89.7%",
                    "parameters": "100 estimators, max_depth=10"
                },
                "XGBoost": {
                    "purpose": "Land suitability scoring",
                    "accuracy": "91.3%",
                    "parameters": "500 estimators, learning_rate=0.01"
                },
                "SVM": {
                    "purpose": "Terrain categorization",
                    "accuracy": "87.8%",
                    "parameters": "RBF kernel, C=1.0"
                }
            },
            "scoring_system": {
                "A": {"range": "80-100", "description": "Excellent suitability, minimal constraints"},
                "B": {"range": "60-79", "description": "Good suitability, manageable constraints"},
                "C": {"range": "40-59", "description": "Moderate suitability, significant constraints"},
                "D": {"range": "20-39", "description": "Poor suitability, major constraints"},
                "F": {"range": "0-19", "description": "Unsuitable, severe constraints"}
            },
            "factors": {
                "Physical": ["slope", "elevation"],
                "Environmental": ["pollution", "soil", "vegetation"],
                "Hydrology": ["flood", "water", "drainage"],
                "Climatic": ["rainfall", "thermal", "heat"],
                "Socio-economic": ["infrastructure", "landuse", "population"]
            }
        }
    
    def get_model_info(self, model_name):
        """Get detailed information about a specific ML model"""
        return self.project_info["ml_models"].get(model_name, {})
    
    def get_scoring_info(self, grade):
        """Get scoring information for a specific grade"""
        return self.project_info["scoring_system"].get(grade.upper(), {})
    
    def get_factors_by_category(self, category):
        """Get factors by category"""
        return self.project_info["factors"].get(category, [])
    
    def explain_suitability_score(self, score):
        """Explain what a suitability score means"""
        if score >= 80:
            grade = "A"
        elif score >= 60:
            grade = "B"
        elif score >= 40:
            grade = "C"
        elif score >= 20:
            grade = "D"
        else:
            grade = "F"
        
        return self.get_scoring_info(grade)
    
    def compare_sites(self, site_a_data, site_b_data):
        """Compare two sites and provide recommendations"""
        score_a = site_a_data.get('suitability_score', 0)
        score_b = site_b_data.get('suitability_score', 0)
        
        better_site = "Site A" if score_a > score_b else "Site B"
        difference = abs(score_a - score_b)
        
        return {
            "better_site": better_site,
            "score_difference": difference,
            "recommendation": f"{better_site} is recommended with a {difference:.1f} point advantage"
        }

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, name: str, api_key: str, priority: int):
        self.name = name
        self.api_key = api_key
        self.priority = priority
        self.is_available = True
        
    async def generate_response(self, prompt: str, context: Dict = None) -> AIResponse:
        """Generate response from the LLM"""
        raise NotImplementedError

class GrokProvider(LLMProvider):
    """Grok (xAI) - Primary Provider"""
    
    def __init__(self):
        super().__init__("Grok", os.getenv("GROK_API_KEY"), 1)
        self.base_url = "https://api.x.ai/v1"
        
    async def generate_response(self, prompt: str, context: Dict = None) -> AIResponse:
        """Generate response using Grok"""
        try:
            start_time = datetime.now()
            
            # Enhanced prompt with project context
            enhanced_prompt = self._enhance_prompt(prompt, context)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "grok-beta",
                "messages": [
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": enhanced_prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                return AIResponse(
                    content=content,
                    provider="Grok",
                    confidence=0.95,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    sources=self._extract_sources(content),
                    comparison_data=self._generate_comparison_data(prompt, content)
                )
            else:
                logger.error(f"Grok API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Grok provider error: {str(e)}")
            return None
    
    def _enhance_prompt(self, prompt: str, context: Dict) -> str:
        """Enhance prompt with project context"""
        kb = GeoAIKnowledgeBase()
        
        context_info = ""
        if context:
            if "question_type" in context:
                if context["question_type"] == "technical":
                    context_info = f"\n\nProject Technical Details:\n{json.dumps(kb.model_details, indent=2)}"
                elif context["question_type"] == "features":
                    context_info = f"\n\nProject Features:\n{', '.join(kb.project_info['features'])}"
                elif context["question_type"] == "implementation":
                    context_info = f"\n\nImplementation Details:\n{json.dumps(kb.implementation_details, indent=2)}"
        
        return f"""You are GeoAI Assistant, an expert AI assistant for the GeoAI project.

Project Context:
{json.dumps(kb.project_info, indent=2)}{context_info}

User Question: {prompt}

Please provide a detailed, accurate response. If the question is a greeting, respond naturally and ask what they'd like to know about the GeoAI project."""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Grok"""
        return """You are GeoAI Assistant, a knowledgeable AI helper for the GeoAI geospatial intelligence platform. 

Your role is to:
1. Answer questions about the GeoAI project accurately and in detail
2. Explain technical concepts clearly
3. Provide implementation details when asked
4. Compare different models and approaches used
5. Help users understand features and capabilities
6. Respond naturally to greetings and casual conversation

Guidelines:
- Be friendly and professional
- Provide specific, accurate information
- Include technical details when relevant
- Use comparisons and examples when helpful
- Keep responses concise but comprehensive
- If you don't know something, admit it honestly

Current date: {current_date}""".format(current_date=datetime.now().strftime("%Y-%m-%d"))
    
    def _extract_sources(self, content: str) -> List[str]:
        """Extract sources mentioned in response"""
        sources = []
        if "CNN" in content:
            sources.append("CNN Model Documentation")
        if "Random Forest" in content:
            sources.append("Machine Learning Documentation")
        if "API" in content:
            sources.append("API Documentation")
        return sources
    
    def _generate_comparison_data(self, prompt: str, response: str) -> Dict:
        """Generate comparison data for the response"""
        return {
            "word_count": len(response.split()),
            "technical_terms": len([term for term in ["CNN", "API", "model", "algorithm", "accuracy"] if term in response]),
            "has_examples": "example" in response.lower(),
            "comparison_made": "vs" in response.lower() or "compared" in response.lower()
        }

class OpenAIProvider(LLMProvider):
    """OpenAI GPT - Secondary Provider"""
    
    def __init__(self):
        super().__init__("OpenAI", os.getenv("OPENAI_API_KEY"), 2)
        self.base_url = "https://api.openai.com/v1"
        
    async def generate_response(self, prompt: str, context: Dict = None) -> AIResponse:
        """Generate response using OpenAI"""
        try:
            start_time = datetime.now()
            
            # Use same enhancement logic as Grok
            grok_provider = GrokProvider()
            enhanced_prompt = grok_provider._enhance_prompt(prompt, context)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": grok_provider._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.base_url}/chat/completions",
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                return AIResponse(
                    content=content,
                    provider="OpenAI",
                    confidence=0.92,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    sources=grok_provider._extract_sources(content),
                    comparison_data=grok_provider._generate_comparison_data(prompt, content)
                )
            else:
                logger.error(f"OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI provider error: {str(e)}")
            return None

class GeminiProvider(LLMProvider):
    """Google Gemini - Tertiary Backup Provider"""
    
    def __init__(self):
        super().__init__("Gemini", os.getenv("GEMINI_API_KEY"), 3)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    async def generate_response(self, prompt: str, context: Dict = None) -> AIResponse:
        """Generate response using Gemini"""
        try:
            start_time = datetime.now()
            
            # Use same enhancement logic
            grok_provider = GrokProvider()
            enhanced_prompt = grok_provider._enhance_prompt(prompt, context)
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": f"{grok_provider._get_system_prompt()}\n\n{enhanced_prompt}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.base_url}/models/gemini-pro:generateContent?key={self.api_key}",
                headers=headers, json=data, timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                return AIResponse(
                    content=content,
                    provider="Gemini",
                    confidence=0.88,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    sources=grok_provider._extract_sources(content),
                    comparison_data=grok_provider._generate_comparison_data(prompt, content)
                )
            else:
                logger.error(f"Gemini API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Gemini provider error: {str(e)}")
            return None

class GeoAIAssistant:
    """Main GeoAI Assistant with fallback system"""
    
    def __init__(self):
        self.providers = [
            GrokProvider(),
            OpenAIProvider(),
            GeminiProvider()
        ]
        self.knowledge_base = GeoAIKnowledgeBase()
        self.conversation_history = []
        
    async def ask_question(self, question: str, context: Dict = None) -> AIResponse:
        """Ask question with automatic fallback"""
        
        # Detect question type for context enhancement
        if context is None:
            context = self._detect_question_type(question)
        
        # Try each provider in order of priority
        for provider in sorted(self.providers, key=lambda p: p.priority):
            if not provider.is_available:
                continue
                
            try:
                response = await provider.generate_response(question, context)
                if response:
                    # Store in conversation history
                    self.conversation_history.append({
                        "question": question,
                        "response": response,
                        "timestamp": datetime.now()
                    })
                    return response
                    
            except Exception as e:
                logger.error(f"Provider {provider.name} failed: {str(e)}")
                continue
        
        # If all providers fail, return fallback response
        return self._fallback_response(question)
    
    def _detect_question_type(self, question: str) -> Dict:
        """Detect question type for better context"""
        question_lower = question.lower()
        
        # Explicit team question detection
        if any(term in question_lower for term in ["team", "team members", "project guide", "who are", "who is", "developers", "mentor", "team information"]):
            return {"question_type": "team"}
        elif any(term in question_lower for term in ["main lead", "lead developer", "project lead", "who led", "shaped this project"]):
            return {"question_type": "main_lead"}
        elif any(term in question_lower for term in ["model", "algorithm", "cnn", "machine learning"]):
            return {"question_type": "technical"}
        elif any(term in question_lower for term in ["feature", "capability", "what can"]):
            return {"question_type": "features"}
        elif any(term in question_lower for term in ["implement", "how", "build", "code"]):
            return {"question_type": "implementation"}
        elif any(term in question_lower for term in ["compare", "versus", "vs", "difference"]):
            return {"question_type": "comparison"}
        else:
            return {"question_type": "general"}
    
    def _fallback_response(self, question: str) -> AIResponse:
        """Fallback response when all providers fail"""
        kb = self.knowledge_base
        
        question_lower = question.lower()
        
        # Explicit team question response
        if any(term in question_lower for term in ["team", "team members",  "project guide", "who are", "who is", "developers", "mentor"]):
            content = """### 👥 Official GeoAI Team Information





**🚨 IMPORTANT**: 

- **Project Guide**: Dr. G. Naga Chandrika (GUIDE )
- These are the ONLY team members. The GeoAI project was developed by exactly 4 developers under the guidance of Dr. G. Naga Chandrika.

**📋 Project Overview:**
- **Name**: GeoAI Land Suitability Intelligence
- **Purpose**: High-precision terrain synthesis using AI and satellite data
- **Technologies**: React.js, Flask, MongoDB, Python, ML Models
- **ML Models**: CNN, Random Forest, XGBoost, SVM, Gradient Boosting, Extra Trees

This is the complete and accurate team information for the GeoAI project."""
        
        elif any(term in question_lower for term in ["main lead", "lead developer", "project lead", "who led", "shaped this project"]):
            content = """### 🎯 Project Leadership - GeoAI Land Suitability Intelligence



**🔧 Leadership Contributions:**
- Led the complete project architecture and design
- Developed backend APIs and database schema
- Integrated all external APIs and data sources
- Implemented ML model integration and deployment
- Coordinated team development and project timeline
- Designed and implemented the multi-LLM GeoGPT system
- Oversaw the complete development lifecycle from requirements to deployment

**📊 Project Achievements Under Leadership:**
- Successfully integrated 6 different ML models (CNN, Random Forest, XGBoost, SVM, Gradient Boosting, Extra Trees)
- Implemented comprehensive 14-factor land suitability analysis
- Created 3D terrain visualization with MapLibre GL
- Built real-time geospatial intelligence system
- Developed responsive frontend with React.js
- Integrated multiple external APIs (Open-Meteo, OpenStreetMap, OpenAQ, MapTiler)

**🚀 For More Information:**
Thanks for your interest in the GeoAI project! For more details about the project leadership and implementation, please visit:

**Harsha vardhan Botlagunta**
🌐 **Portfolio**: [Visit Portfolio]({pk['team']['team_lead_portfolio']})

The project represents a comprehensive geospatial intelligence platform that combines advanced ML models with real-time data analysis to provide accurate land suitability assessments."""
        
        elif question_lower in ["hi", "hello", "hey"]:
            content = """Hello! I'm GeoGPT Intelligence. What would you like to know about our geospatial intelligence platform?

**👥 Official Team:**
- **Project Guide**: Dr. G. Naga Chandrika
- **Team Members**: Adepu Vaishnavi, Chinni Jyothika, Maganti Pranathi

**🚀 What I Can Help With:**
• Land suitability analysis and scoring
• Site comparisons and recommendations
• Machine learning model details and accuracy
• 3D terrain visualization implementation
• Development methodology and technologies used
• API integrations and data sources
• Scoring system and factor analysis
• Technical implementation details

Please try again in a moment, or ask me about any specific feature!"""
        else:
            content = f"""I apologize, but I'm having trouble connecting to my AI services right now. 

However, I can tell you that GeoAI is a comprehensive geospatial intelligence platform with these key features:
• Land Suitability Analysis
• Risk Assessment 
• Geospatial Intelligence
• Digital Twin Simulation
• Weather Integration
• Terrain Analysis

The system uses advanced ML models including CNN (94.2% accuracy), Random Forest (89.7%), XGBoost (91.3%), and SVM (87.8%).

Please try again in a moment, or ask me about any specific feature!"""
        
        return AIResponse(
            content=content,
            provider="Fallback",
            confidence=0.5,
            response_time=0.1,
            timestamp=datetime.now()
        )
    
    def get_provider_status(self) -> Dict:
        """Get status of all providers"""
        return {
            provider.name: {
                "priority": provider.priority,
                "available": provider.is_available
            }
            for provider in self.providers
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def generate_comparison_table(self, topic: str) -> Dict:
        """Generate comparison table for different aspects"""
        kb = self.knowledge_base
        
        if topic == "models":
            return {
                "type": "table",
                "title": "ML Model Comparison",
                "headers": ["Model", "Purpose", "Accuracy", "Key Parameters"],
                "data": [
                    ["CNN", "Image analysis", "94.2%", "23.5M parameters, ResNet backbone"],
                    ["Random Forest", "Feature classification", "89.7%", "100 estimators, max_depth=10"],
                    ["XGBoost", "Land suitability scoring", "91.3%", "500 estimators, lr=0.01"],
                    ["SVM", "Terrain categorization", "87.8%", "RBF kernel, C=1.0"]
                ]
            }
        elif topic == "features":
            return {
                "type": "table", 
                "title": "Platform Features",
                "headers": ["Feature", "Description", "Technology"],
                "data": [
                    ["Land Analysis", "Comprehensive site evaluation", "ML Models + GIS"],
                    ["Risk Assessment", "Multi-factor risk analysis", "Statistical Models"],
                    ["Digital Twin", "3D simulation environment", "Three.js + WebGL"],
                    ["Weather Integration", "Real-time weather data", "OpenWeatherMap API"],
                    ["Terrain Analysis", "Elevation and slope analysis", "NASA Earth Data"]
                ]
            }
        else:
            return {"error": "Unknown comparison topic"}

# Flask Application
app = Flask(__name__)
CORS(app)

# Initialize assistant
assistant = GeoAIAssistant()

@app.route('/api/ask', methods=['POST'])
async def ask_question():
    """Main endpoint for asking questions"""
    try:
        data = request.json
        question = data.get('question', '')
        context = data.get('context', {})
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        response = await assistant.ask_question(question, context)
        
        return jsonify({
            "response": response.content,
            "provider": response.provider,
            "confidence": response.confidence,
            "response_time": response.response_time,
            "timestamp": response.timestamp.isoformat(),
            "sources": response.sources,
            "comparison_data": response.comparison_data
        })
        
    except Exception as e:
        logger.error(f"Ask question error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get assistant status"""
    return jsonify({
        "providers": assistant.get_provider_status(),
        "conversation_count": len(assistant.conversation_history),
        "project_info": assistant.knowledge_base.project_info
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        "history": assistant.get_conversation_history(limit)
    })

@app.route('/api/compare/<topic>', methods=['GET'])
def get_comparison(topic):
    """Get comparison table"""
    comparison = assistant.generate_comparison_table(topic)
    return jsonify(comparison)

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get all project features"""
    return jsonify({
        "features": assistant.knowledge_base.project_info["features"],
        "technologies": assistant.knowledge_base.project_info["technologies"]
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get ML model details"""
    return jsonify({
        "models": assistant.knowledge_base.model_details
    })

@app.route('/api/implementation', methods=['GET'])
def get_implementation():
    """Get implementation details"""
    return jsonify({
        "implementation": assistant.knowledge_base.implementation_details
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
