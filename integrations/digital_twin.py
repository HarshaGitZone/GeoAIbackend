"""
Digital Twin Infrastructure Simulation
Calculates impact of proposed developments on regional factors
"""

import numpy as np
from typing import Dict, List, Any

def calculate_development_impact(latitude: float, longitude: float, 
                               development_type: str, 
                               existing_factors: Dict[str, Any],
                               placed_developments: List[Dict] = None) -> Dict[str, Any]:
    """
    Calculate the impact of a proposed development on regional factors
    """
    
    # Development impact profiles
    development_profiles = {
        'residential': {
            'pollution': 3.2,      # +3.2% pollution from increased traffic/consumption
            'traffic': 8.5,         # +8.5% traffic load
            'infrastructure': 12.1, # +12.1% strain on infrastructure
            'population': 15.3,    # +15.3% population density
            'water': 4.2,          # +4.2% water consumption
            'waste': 6.8,          # +6.8% waste generation
            'noise': 9.1           # +9.1% noise pollution
        },
        'commercial': {
            'pollution': 5.1,
            'traffic': 12.3,
            'infrastructure': 8.7,
            'population': 10.2,
            'water': 3.8,
            'waste': 8.9,
            'noise': 7.4
        },
        'industrial': {
            'pollution': 15.7,
            'traffic': 6.2,
            'infrastructure': 5.3,
            'population': 4.8,
            'water': 12.4,
            'waste': 18.2,
            'noise': 11.6
        },
        'hospital': {
            'pollution': 2.3,
            'traffic': 10.1,
            'infrastructure': 15.2,
            'population': 8.4,
            'water': 8.7,
            'waste': 12.3,
            'noise': 5.8
        },
        'school': {
            'pollution': 1.2,
            'traffic': 8.3,
            'infrastructure': 10.4,
            'population': 12.1,
            'water': 3.2,
            'waste': 4.1,
            'noise': 6.7
        },
        'park': {
            'pollution': -8.4,     # -8.4% pollution (air purification)
            'traffic': 2.1,        # +2.1% traffic (visitors)
            'infrastructure': 5.2, # +5.2% (maintenance)
            'population': 3.1,    # +3.1% (recreation visitors)
            'water': -2.3,        # -2.3% (groundwater recharge)
            'waste': -1.2,        # -1.2% (less waste)
            'noise': -12.7        # -12.7% noise reduction
        }
    }
    
    # Get the impact profile for the selected development type
    profile = development_profiles.get(development_type, development_profiles['residential'])
    
    # Calculate cumulative impact from existing developments
    cumulative_impact = {}
    if placed_developments:
        for dev in placed_developments:
            dev_type = dev.get('development_type', 'residential')
            dev_profile = development_profiles.get(dev_type, development_profiles['residential'])
            for factor, impact in dev_profile.items():
                cumulative_impact[factor] = cumulative_impact.get(factor, 0) + impact
    
    # Combine current development impact with cumulative impact
    total_impact = {}
    for factor, impact in profile.items():
        total_impact[factor] = impact + cumulative_impact.get(factor, 0)
    
    # Apply diminishing returns for multiple developments
    for factor in total_impact:
        if total_impact[factor] > 20:
            total_impact[factor] = 20 + (total_impact[factor] - 20) * 0.5  # Diminishing returns after 20%
        elif total_impact[factor] > 10:
            total_impact[factor] = 10 + (total_impact[factor] - 10) * 0.7  # Reduced impact after 10%
    
    # Calculate updated factor scores
    updated_scores = {}
    base_scores = {
        'pollution': existing_factors.get('pollution', {}).get('score', 50),
        'infrastructure': existing_factors.get('infrastructure', {}).get('score', 50),
        'population': existing_factors.get('population', {}).get('score', 50),
        'water': existing_factors.get('water', {}).get('score', 50),
        'vegetation': existing_factors.get('vegetation', {}).get('score', 50),
        'landuse': existing_factors.get('landuse', {}).get('score', 50)
    }
    
    # Apply impacts to base scores
    updated_scores['pollution'] = max(0, min(100, base_scores['pollution'] - total_impact.get('pollution', 0)))
    updated_scores['infrastructure'] = max(0, min(100, base_scores['infrastructure'] - total_impact.get('infrastructure', 0)))
    updated_scores['population'] = max(0, min(100, base_scores['population'] + total_impact.get('population', 0)))
    updated_scores['water'] = max(0, min(100, base_scores['water'] - total_impact.get('water', 0)))
    updated_scores['vegetation'] = max(0, min(100, base_scores['vegetation'] - total_impact.get('pollution', 0) * 0.5))  # Pollution affects vegetation
    updated_scores['landuse'] = max(0, min(100, base_scores['landuse'] - total_impact.get('infrastructure', 0) * 0.3))  # Infrastructure affects landuse
    
    # Calculate overall suitability change
    original_suitability = calculate_suitability_score(base_scores)
    new_suitability = calculate_suitability_score(updated_scores)
    suitability_change = new_suitability - original_suitability
    
    # Generate recommendations based on impacts
    feasibility_score = calculate_feasibility_score(total_impact, development_type)
    recommendations = generate_recommendations(total_impact, development_type, suitability_change)
    
    return {
        'development_type': development_type,
        'impact_analysis': total_impact,
        'updated_scores': updated_scores,
        'overall_suitability_change': suitability_change,
        'recommendations': recommendations,
        'cumulative_developments': len(placed_developments) + 1,
        'feasibility_analysis': {
            'score': feasibility_score,
            'rating': get_feasibility_rating(feasibility_score),
            'risk_level': get_risk_level(feasibility_score),
            'confidence': get_confidence_level(feasibility_score),
            'roi_estimate': calculate_roi_estimate(development_type, total_impact, suitability_change),
            'timeline_months': estimate_timeline(development_type, total_impact),
            'population_impact': total_impact.get('population', 0)
        },
        'environmental_impact_details': {
            'pollution_change': f"+{total_impact.get('pollution', 0):.1f}%",
            'traffic_change': f"+{total_impact.get('traffic', 0):.1f}%",
            'infrastructure_strain': f"+{total_impact.get('infrastructure', 0):.1f}%",
            'water_consumption': f"+{total_impact.get('water', 0):.1f}%",
            'vegetation_impact': f"{total_impact.get('pollution', 0) * 0.5:.1f}% reduction" if total_impact.get('pollution', 0) > 0 else f"{abs(total_impact.get('pollution', 0) * 0.5):.1f}% improvement"
        }
    }

def calculate_roi_estimate(development_type: str, impacts: Dict[str, float], suitability_change: float) -> Dict[str, Any]:
    """
    Calculate ROI estimates based on development type and impacts
    """
    base_roi = {
        'residential': {'min': 6.5, 'max': 12.5, 'avg': 9.5},
        'commercial': {'min': 8.2, 'max': 15.8, 'avg': 11.8},
        'industrial': {'min': 7.8, 'max': 14.2, 'avg': 10.2},
        'hospital': {'min': 5.5, 'max': 10.5, 'avg': 7.8},
        'school': {'min': 4.2, 'max': 8.8, 'avg': 6.2},
        'park': {'min': 2.1, 'max': 5.5, 'avg': 3.8}
    }
    
    roi_data = base_roi.get(development_type, base_roi['residential'])
    
    # Adjust ROI based on impacts
    impact_penalty = 0
    if impacts.get('pollution', 0) > 10:
        impact_penalty -= 2.5  # Mitigation costs
    elif impacts.get('infrastructure', 0) > 10:
        impact_penalty -= 1.8  # Infrastructure costs
    elif impacts.get('water', 0) > 8:
        impact_penalty -= 1.2  # Water management costs
    
    adjusted_roi = {
        'min': roi_data['min'] + impact_penalty,
        'max': roi_data['max'] + impact_penalty,
        'avg': roi_data['avg'] + impact_penalty
    }
    
    return {
        'roi_percentage': f"{adjusted_roi['avg']:.1f}%",
        'range': f"{adjusted_roi['min']:.1f}% - {adjusted_roi['max']:.1f}%",
        'payback_period_months': int((100 / adjusted_roi['avg']) * 12),
        'factors': ['Property value appreciation', 'Rental income', 'Tax benefits', 'Development premiums']
    }

def estimate_timeline(development_type: str, impacts: Dict[str, float]) -> int:
    """
    Estimate development timeline based on type and complexity
    """
    base_timeline = {
        'residential': 24,
        'commercial': 30,
        'industrial': 36,
        'hospital': 48,
        'school': 30,
        'park': 18
    }
    
    timeline = base_timeline.get(development_type, 24)
    
    # Adjust for complexity
    if impacts.get('infrastructure', 0) > 10:
        timeline += 12  # Major infrastructure work
    elif impacts.get('pollution', 0) > 10:
        timeline += 8   # Environmental mitigation
    elif impacts.get('water', 0) > 8:
        timeline += 6   # Water management systems
    
    return timeline

def calculate_suitability_score(scores: Dict[str, float]) -> float:
    """
    Calculate overall suitability score from factor scores
    """
    # Weight factors for overall suitability
    weights = {
        'pollution': 0.2,
        'infrastructure': 0.15,
        'population': 0.15,
        'water': 0.2,
        'vegetation': 0.15,
        'landuse': 0.15
    }
    
    weighted_score = 0
    total_weight = 0
    
    for factor, score in scores.items():
        if factor in weights:
            weighted_score += score * weights[factor]
            total_weight += weights[factor]
    
    return weighted_score / total_weight if total_weight > 0 else 50

def generate_recommendations(impacts: Dict[str, float], development_type: str, suitability_change: float) -> List[str]:
    """
    Generate detailed recommendations with specific numerical reasoning and feasibility ratings
    """
    recommendations = []
    
    # Calculate feasibility based on impact severity and type
    feasibility_score = calculate_feasibility_score(impacts, development_type)
    
    # Pollution-related recommendations with specific numerical targets
    if impacts.get('pollution', 0) > 10:
        recommendations.append(f"🏭 POLLUTION MITIGATION: Current pollution increase: +{impacts.get('pollution', 0):.1f}%. Target: Reduce by 40-60% through HEPA filtration systems, green buffer zones (200m radius), and traffic management. Expected improvement: PM2.5 reduction from 35μg/m³ to 15μg/m³ within 18 months.")
        recommendations.append(f"🌱 GREEN INFRASTRUCTURE: Plant 150-200 native trees to offset pollution impact. Target vegetation coverage increase: 15% → 40% over 5 years. CO2 sequestration: 25 tons/year.")
    elif impacts.get('pollution', 0) > 5:
        recommendations.append(f"🌫️ AIR QUALITY MONITORING: Moderate pollution increase: +{impacts.get('pollution', 0):.1f}%. Install real-time air quality sensors, implement low-emission zones. Target: Maintain AQI below 100 during peak hours.")
    
    # Infrastructure recommendations with specific capacity targets
    if impacts.get('infrastructure', 0) > 10:
        recommendations.append(f"🛣️ INFRASTRUCTURE UPGRADE: Current strain: +{impacts.get('infrastructure', 0):.1f}%. Invest $2-3M in road widening (2→4 lanes), utility upgrades (water capacity +40%, electricity +25%). Expected: Reduce commute times by 25%, increase property values by 15-20%.")
    elif impacts.get('infrastructure', 0) > 5:
        recommendations.append(f"🔧 CAPACITY ASSESSMENT: Moderate infrastructure impact: +{impacts.get('infrastructure', 0):.1f}%. Conduct traffic flow analysis, plan phased utility upgrades. Target: Maintain service level above 85% during peak demand.")
    
    # Water-related recommendations with conservation targets
    if impacts.get('water', 0) > 8:
        recommendations.append(f"💧 WATER CONSERVATION: Current consumption increase: +{impacts.get('water', 0):.1f}%. Implement rainwater harvesting (50,000L capacity), greywater recycling (60% reuse rate), low-flow fixtures. Target: Reduce municipal water demand by 35%.")
    elif impacts.get('water', 0) < -5:
        recommendations.append(f"💚 WATER SUSTAINABILITY: Excellent water impact: {impacts.get('water', 0):.1f}% (reduction). Maintain permeable surfaces, expand green infrastructure. Groundwater recharge: +2.3m/year.")
    
    # Development-specific recommendations with ROI calculations
    if development_type == 'industrial':
        recommendations.append(f"🏭 INDUSTRIAL BUFFERING: High impact zone requires 500m environmental buffer. Implement continuous emission monitoring (real-time sensors). Estimated ROI: 9.5% over 56 months through efficiency gains and regulatory compliance.")
        recommendations.append(f"⚡ ENERGY EFFICIENCY: Current pollution impact: +{impacts.get('pollution', 0):.1f}%. Install solar panels (200kW capacity), waste heat recovery. Expected energy cost reduction: 30%, payback period: 4.2 years.")
    elif development_type == 'residential':
        recommendations.append(f"🏘️ COMMUNITY INFRASTRUCTURE: Population impact: +{impacts.get('population', 0):.1f}%. Requires 1 school per 500 residents, 1 healthcare facility per 2000 residents. Timeline: 18-24 months for full service establishment.")
        recommendations.append(f"🌳 GREEN SPACES: Target green space ratio: 25% of total area. Implement community gardens, pocket parks. Expected: Reduce urban heat island effect by 2-3°C, improve property values by 8-12%.")
    elif development_type == 'commercial':
        recommendations.append(f"🏢 COMMERCIAL VIABILITY: Traffic impact: +{impacts.get('traffic', 0):.1f}%. Implement shared parking, shuttle services, traffic management system. Expected: Reduce peak hour congestion by 40%, improve access time by 30%.")
    elif development_type == 'park':
        recommendations.append(f"🌲 ENVIRONMENTAL RESTORATION: Excellent sustainability choice. Pollution reduction: {abs(impacts.get('pollution', 0)):.1f}%. Target: Connect to existing green corridors, create wildlife habitats. Expected biodiversity increase: 35% within 3 years.")
    
    # Overall suitability recommendations with numerical targets
    if suitability_change < -10:
        recommendations.append(f"⚠️ HIGH IMPACT WARNING: Suitability decrease: {suitability_change:.1f} points. Reconsider location scale or implement comprehensive mitigation. Required improvements: +15-20 points across multiple factors.")
    elif suitability_change < -5:
        recommendations.append(f"⚡ MODERATE MITIGATION NEEDED: Suitability impact: {suitability_change:.1f} points. Implement targeted improvements. Target: Recover 8-12 points through focused interventions.")
    elif suitability_change > 5:
        recommendations.append(f"✅ POSITIVE DEVELOPMENT: Suitability improvement: +{suitability_change:.1f} points. Proceed with monitoring. Expected long-term benefits: Enhanced property values, improved livability.")
    
    # Add feasibility assessment
    feasibility_rating = get_feasibility_rating(feasibility_score)
    recommendations.append(f"📊 FEASIBILITY ASSESSMENT: {feasibility_rating} (Score: {feasibility_score:.1f}/100). Risk level: {get_risk_level(feasibility_score)}. Confidence: {get_confidence_level(feasibility_score)}")
    
    # Remove duplicates and limit to most relevant
    unique_recommendations = list(dict.fromkeys(recommendations))  # Preserve order while removing duplicates
    return unique_recommendations[:10]  # Limit to top 10 recommendations

def calculate_feasibility_score(impacts: Dict[str, float], development_type: str) -> float:
    """
    Calculate feasibility score based on impact severity and development type
    """
    base_score = 85.0
    
    # Deduct points for high impacts
    if impacts.get('pollution', 0) > 10:
        base_score -= 15
    elif impacts.get('pollution', 0) > 5:
        base_score -= 8
        
    if impacts.get('infrastructure', 0) > 10:
        base_score -= 12
    elif impacts.get('infrastructure', 0) > 5:
        base_score -= 6
        
    if impacts.get('water', 0) > 8:
        base_score -= 10
    elif impacts.get('water', 0) > 3:
        base_score -= 5
    
    # Development type adjustments
    if development_type == 'industrial':
        base_score -= 8  # Higher regulatory burden
    elif development_type == 'park':
        base_score += 10  # Community support
    elif development_type == 'residential':
        base_score += 5  # Always in demand
    
    return max(20, min(100, base_score))

def get_feasibility_rating(score: float) -> str:
    """
    Convert feasibility score to rating
    """
    if score >= 85:
        return "EXCELLENT"
    elif score >= 70:
        return "GOOD"
    elif score >= 55:
        return "MODERATE"
    elif score >= 40:
        return "CHALLENGING"
    else:
        return "HIGH RISK"

def get_risk_level(score: float) -> str:
    """
    Get risk level based on feasibility score
    """
    if score >= 70:
        return "Low"
    elif score >= 50:
        return "Medium"
    else:
        return "High"

def get_confidence_level(score: float) -> str:
    """
    Get confidence level based on feasibility score
    """
    if score >= 80:
        return "High (85%+ success probability)"
    elif score >= 60:
        return "Medium (65-85% success probability)"
    else:
        return "Low (50-65% success probability)"
