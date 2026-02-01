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
    recommendations = generate_recommendations(total_impact, development_type, suitability_change)
    
    return {
        'development_type': development_type,
        'impact_analysis': total_impact,
        'updated_scores': updated_scores,
        'overall_suitability_change': suitability_change,
        'recommendations': recommendations,
        'cumulative_developments': len(placed_developments) + 1
    }

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
    Generate recommendations based on development impacts
    """
    recommendations = []
    
    # Pollution-related recommendations
    if impacts.get('pollution', 0) > 10:
        recommendations.append("Implement strict emission controls and green building standards")
        recommendations.append("Consider additional green spaces to offset air quality impact")
    elif impacts.get('pollution', 0) > 5:
        recommendations.append("Monitor air quality and implement mitigation measures")
    
    # Infrastructure recommendations
    if impacts.get('infrastructure', 0) > 10:
        recommendations.append("Upgrade utilities and transportation infrastructure")
        recommendations.append("Plan for increased public transport capacity")
    elif impacts.get('infrastructure', 0) > 5:
        recommendations.append("Assess infrastructure capacity and plan upgrades")
    
    # Water-related recommendations
    if impacts.get('water', 0) > 8:
        recommendations.append("Implement water conservation and recycling systems")
        recommendations.append("Consider rainwater harvesting and groundwater recharge")
    elif impacts.get('water', 0) < -5:
        recommendations.append("Excellent water sustainability - maintain green infrastructure")
    
    # Development-specific recommendations
    if development_type == 'industrial':
        recommendations.append("Establish environmental monitoring protocols")
        recommendations.append("Create buffer zones with green infrastructure")
    elif development_type == 'residential':
        recommendations.append("Ensure adequate schools and healthcare facilities")
        recommendations.append("Plan for community amenities and public spaces")
    elif development_type == 'commercial':
        recommendations.append("Coordinate with public transport planning")
        recommendations.append("Implement shared parking and traffic management")
    elif development_type == 'park':
        recommendations.append("Excellent choice for environmental sustainability")
        recommendations.append("Connect to existing green corridors for maximum impact")
    
    # Overall suitability recommendations
    if suitability_change < -10:
        recommendations.append("⚠️ Significant suitability decrease - reconsider location or scale")
    elif suitability_change < -5:
        recommendations.append("⚡ Moderate suitability impact - implement mitigation strategies")
    elif suitability_change > 5:
        recommendations.append("✅ Positive impact on overall suitability")
    
    # Remove duplicates and limit to most relevant
    unique_recommendations = list(dict.fromkeys(recommendations))  # Preserve order while removing duplicates
    return unique_recommendations[:8]  # Limit to top 8 recommendations
