import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import requests
import qrcode
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

def _draw_wrapped_text(c, text, x, y, max_width, line_height):
    """Helper to manually wrap text within a specific width on the PDF."""
    if not text:
        return y - line_height
    words = str(text).split(' ')
    line = ""
    for word in words:
        if c.stringWidth(line + word + " ", "Helvetica", 7.5) < max_width:
            line += word + " "
        else:
            c.drawString(x, y, line)
            line = word + " "
            y -= line_height
    c.drawString(x, y, line)
    return y - line_height

# 15 factors in order of 5 categories (same as frontend)
FACTOR_ORDER_15 = [
    'slope', 'elevation', 'vegetation', 'soil', 'pollution',
    'flood', 'water', 'drainage', 'rainfall', 'thermal', 'intensity',
    'landuse', 'infrastructure', 'population'
]
FACTOR_LABELS_15 = {k: k.capitalize() for k in FACTOR_ORDER_15}
FACTOR_LABELS_15['infrastructure'] = 'Infra'

def _flatten_factors(data):
    """Convert nested factors (by category) to flat { factor_name: value } and ordered list for radar."""
    raw = data.get("factors", {})
    if not raw:
        return {}, []
    flat = {}
    if isinstance(raw, dict) and any(isinstance(v, dict) for v in raw.values()):
        for cat, cat_data in raw.items():
            if not isinstance(cat_data, dict):
                continue
            for fkey, fval in cat_data.items():
                if isinstance(fval, dict):
                    s = fval.get("scaled_score")
                    v = fval.get("value", 0)
                    flat[fkey] = max(0, min(100, float(s if s is not None else v)))
                else:
                    flat[fkey] = max(0, min(100, float(fval) if fval is not None else 0))
    else:
        for k, v in raw.items():
            flat[k] = float(v) if not isinstance(v, dict) else float(v.get("value", 0))
    ordered = [(k, flat.get(k, 0)) for k in FACTOR_ORDER_15]
    extra = [(k, v) for k, v in flat.items() if k not in FACTOR_ORDER_15]
    ordered = ordered + extra
    if not ordered:
        ordered = list(flat.items())
    return flat, ordered

def _calculate_site_potential(factors):
    """Evaluates 15-factor combinations for prescriptive insights."""
    potentials = []
    f = {k: float(v) for k, v in factors.items()}

    if any(f.get(k, 100) < 45 for k in ['flood', 'pollution', 'drainage']):
        hazards = [k.upper() for k in ['flood', 'pollution', 'drainage'] if f.get(k, 100) < 45]
        potentials.append({
            "label": "ENVIRONMENTAL CONSTRAINTS",
            "color": colors.HexColor("#ef4444"),
            "reason": f"CRITICAL RISK: Low scores in {', '.join(hazards)} indicate hazard vulnerability."
        })
    if f.get('flood', 0) > 50 and f.get('pollution', 0) > 40 and f.get('slope', 100) < 25:
        potentials.append({
            "label": "RESIDENTIAL POTENTIAL",
            "color": colors.HexColor("#10b981"),
            "reason": "Viable for residential development: stable terrain, air quality, and flood safety."
        })
    if f.get('soil', 0) > 60 or f.get('rainfall', 0) > 60 or f.get('vegetation', 0) > 50:
        potentials.append({
            "label": "AGRICULTURAL UTILITY",
            "color": colors.HexColor("#3b82f6"),
            "reason": "Agricultural potential from soil, rainfall, or vegetation indicators."
        })
    if f.get('infrastructure', 0) > 60 and f.get('landuse', 0) > 40:
        potentials.append({
            "label": "LOGISTICS & INDUSTRY",
            "color": colors.HexColor("#8b5cf6"),
            "reason": "Strategically positioned for industrial use due to infrastructure and land use."
        })
    return potentials 

def _draw_section_header(c, x, y, width, text):
    """Draws a navy blue sub-header to separate the PDF into 'Tabs'"""
    COLOR_DEEP_NAVY = colors.HexColor("#0f172a")
    c.setFillColor(COLOR_DEEP_NAVY)
    c.roundRect(x, y, width - 80, 18, 4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y + 5, text.upper())
    return y - 10

def _draw_terrain_module(c, terrain, x, y, width):
    """Draws the Terrain & Slope box"""
    if not terrain: return y
    slope = terrain.get('slope_percent', 0)
    verdict = terrain.get('verdict', 'N/A')
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(x, y - 50, width - 80, 50, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y - 12, "TERRAIN & SLOPE ANALYSIS")
    slope_color = colors.HexColor("#ef4444") if slope > 15 else (colors.HexColor("#f59e0b") if slope > 5 else colors.HexColor("#10b981"))
    c.setFillColor(slope_color)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 10, y - 28, f"{slope:.1f}% Gradient")
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(x + 10, y - 42, f"Verdict: {verdict}")
    return y - 60

def _draw_weather_module(c, weather, x, y, width):
    """Draws weather with full detail (temp, humidity, wind, pressure, code)."""
    if not weather: return y
    c.setFillColor(colors.HexColor("#f0f9ff"))
    c.roundRect(x, y - 58, width - 80, 58, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y - 12, "WEATHER (LIVE TELEMETRY)")
    c.setFont("Helvetica", 8)
    c.drawString(x + 10, y - 24, f"Temperature: {weather.get('temp', 'N/A')}°C  |  Humidity: {weather.get('humidity', 'N/A')}%")
    c.drawString(x + 10, y - 34, f"Conditions: {weather.get('description', 'N/A')}")
    c.drawString(x + 10, y - 44, f"Wind: {weather.get('wind_speed_kmh', weather.get('wind_speed', 'N/A'))} km/h  |  Pressure: {weather.get('pressure_hpa', 'N/A')} hPa")
    c.drawString(x + 10, y - 54, f"Weather code: {weather.get('weather_code', 'N/A')}")
    return y - 68

def _draw_geospatial_passport_module(c, passport, x, y, width):
    """Draws geospatial passport summary."""
    if not passport: return y
    c.setFillColor(colors.HexColor("#ecfdf5"))
    c.roundRect(x, y - 72, width - 80, 72, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y - 12, "GEOSPATIAL PASSPORT")
    c.setFont("Helvetica", 7.5)
    c.drawString(x + 10, y - 24, f"Slope: {passport.get('slope_percent', 'N/A')}% (suit. {passport.get('slope_suitability', 'N/A')})  |  Elev: {passport.get('elevation_m', 'N/A')}m")
    c.drawString(x + 10, y - 34, f"Vegetation: {passport.get('vegetation_score', 'N/A')}  |  Landuse: {passport.get('landuse_class', 'N/A')}  |  Water dist: {passport.get('water_distance_km', 'N/A')} km")
    c.drawString(x + 10, y - 44, f"Flood safety: {passport.get('flood_safety_score', 'N/A')}  |  Rainfall: {passport.get('rainfall_mm', 'N/A')} mm")
    c.drawString(x + 10, y - 54, f"Risk: {passport.get('risk_summary', 'N/A')}  |  Categories: {passport.get('category_breakdown') or 'N/A'}")
    return y - 82

def _draw_cnn_module(c, cnn, x, y, width):
    """Draws CNN classification and live telemetry."""
    if not cnn: return y
    h = 70
    c.setFillColor(colors.HexColor("#fef3c7"))
    c.roundRect(x, y - h, width - 80, h, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y - 12, "CNN CLASSIFICATION (VISUAL INTELLIGENCE)")
    c.setFont("Helvetica", 8)
    c.drawString(x + 10, y - 24, f"Class: {cnn.get('class', 'N/A')}  |  Confidence: {cnn.get('confidence_display', cnn.get('confidence', 'N/A'))}")
    y_cur = y - 32
    if cnn.get('note'):
        y_cur = _draw_wrapped_text(c, (cnn['note'] or '')[:150], x + 10, y_cur, width - 100, 8); y_cur -= 4
    tel = cnn.get("telemetry") or {}
    c.setFont("Helvetica", 7)
    c.drawString(x + 10, y_cur - 10, f"RES: {tel.get('resolution_m_per_px', 'N/A')} m/px  |  SENSOR: {tel.get('tile_url_source', 'N/A')}  |  MODEL: {tel.get('model', 'N/A')}")
    if tel.get('verified_by'):
        c.drawString(x + 10, y_cur - 20, f"Verified: {tel.get('verified_by')}")
    return y - h - 10

def _draw_telemetry_module(c, cnn, x, y, width):
    """Draws live telemetry details from CNN/forensics."""
    tel = (cnn or {}).get("telemetry") or {}
    if not tel: return y
    c.setFillColor(colors.HexColor("#e0e7ff"))
    c.roundRect(x, y - 48, width - 80, 48, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y - 12, "LIVE TELEMETRY")
    c.setFont("Helvetica", 7.5)
    for i, (k, v) in enumerate(list(tel.items())[:6]):
        if k in ('interpretation',): continue
        c.drawString(x + 10, y - 22 - i * 10, f"{k}: {v}")
    return y - 58

def _draw_location_analysis(c, data, title_prefix, width, height):
    COLOR_DANGER = colors.HexColor("#ef4444")
    COLOR_WARNING = colors.HexColor("#f59e0b")
    COLOR_SUCCESS = colors.HexColor("#10b981")
    COLOR_DEEP_NAVY = colors.HexColor("#0f172a") 

    # 1. HEADER
    c.setFillColor(COLOR_DEEP_NAVY)
    c.rect(0, height - 100, width, 100, fill=1, stroke=0)
    # --- NEW: QR CODE SECTION (Top Right) ---
    # Expecting 'shareLink' to be passed in the data payload from frontend
    share_url = data.get('shareLink')
    if share_url:
        try:
            # Generate QR Code
            qr = qrcode.QRCode(version=1, box_size=10, border=2)
            qr.add_data(share_url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to ReportLab-friendly format
            qr_buffer = io.BytesIO()
            qr_img.save(qr_buffer, format='PNG')
            qr_buffer.seek(0)
            
            # Draw QR Box and Label
            c.setFillColor(colors.white)
            c.roundRect(width - 95, height - 90, 80, 80, 4, fill=1, stroke=0)
            c.drawImage(ImageReader(qr_buffer), width - 90, height - 82, width=70, height=70)
            # --- STYLED "BUTTON" SECTION ---
            # Draw a green button-like rectangle
            btn_x, btn_y, btn_w, btn_h = width - 90, height - 88, 70, 10
            c.setFillColor(COLOR_SUCCESS)
            c.roundRect(btn_x, btn_y, btn_w, btn_h, 2, fill=1, stroke=0)

            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            # c.drawCentredString(width - 55, height - 88, "SCAN FOR LIVE")
            c.drawCentredString(width - 55, btn_y + 3, "OPEN LINK")
            # CREATE THE CLICKABLE LINK AREA
            # This makes the button area in the PDF act as a hyperlink
            c.linkURL(share_url, (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h), relative=0)
            
            
        except Exception as e:
            print(f"QR Generation Error: {e}")
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 40, "GeoAI – Land Suitability Certificate")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, height - 60, f"{title_prefix}: {data.get('locationName', 'Site Analysis')}".upper())
    
    # CRITICAL FIX: Coordinate mapping for Map/Header
    loc = data.get('location', {})
    lat = loc.get('latitude') or loc.get('lat') or 0.0
    lon = loc.get('longitude') or loc.get('lng') or 0.0
    
    timestamp = datetime.now().strftime('%d %b %Y | %H:%M:%S IST')
    c.setFont("Helvetica", 8)
    c.drawCentredString(width / 2, height - 80, f"{timestamp}  •  LAT: {lat} | LNG: {lon}")

    # 2. MINI MAP
    y_map = height - 210
    try:
        # Mini map logic must use the same fallbacks to ensure image displays
        map_url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&z=13&l=sat&size=500,140"
        map_res = requests.get(map_url, timeout=5)
        if map_res.status_code == 200:
            map_img = ImageReader(io.BytesIO(map_res.content))
            c.drawImage(map_img, 40, y_map, width=width-80, height=110)
            c.setStrokeColor(colors.white)
            c.rect(40, y_map, width-80, 110, stroke=1, fill=0)
    except Exception as e:
        print(f"Map Error: {e}")
        c.setFillColor(colors.lightgrey); c.rect(40, y_map, width-80, 110, fill=1)

    # 3. SCORECARD
    score = float(data.get('suitability_score', 0))
    score_color = COLOR_DANGER if score < 40 else (COLOR_WARNING if score < 70 else COLOR_SUCCESS)
    y_score = y_map - 45
    c.setFillColor(score_color); c.setFont("Helvetica-Bold", 28)
    c.drawString(45, y_score, f"{score:.1f}")
    c.setFont("Helvetica-Bold", 10); c.drawString(45, y_score - 15, f"GRADE: {data.get('label', 'N/A').upper()}")

    # 4. SECTION 01: SUITABILITY (15 factors: radar + bars side by side)
    y_tab1 = _draw_section_header(c, 40, y_score - 50, width, "Section 01: Suitability Intelligence (15 Factors)")
    factors_flat, factors_ordered = _flatten_factors(data)
    factors = factors_flat
    y_radar = y_tab1 - 165
    if factors_ordered:
        labels = [FACTOR_LABELS_15.get(k, k.capitalize()) for k, _ in factors_ordered]
        values = [v for _, v in factors_ordered]
        fig = plt.figure(figsize=(3, 3), dpi=150)
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ax.fill(angles + angles[:1], values + values[:1], color='#06b6d4', alpha=0.3)
        ax.plot(angles + angles[:1], values + values[:1], color='#06b6d4', linewidth=1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels([])
        chart_io = io.BytesIO(); plt.savefig(chart_io, format='png', transparent=True); plt.close(fig); chart_io.seek(0)
        c.drawImage(ImageReader(chart_io), 35, y_radar, width=160, height=160, mask='auto')

    y_bar = y_tab1 - 15
    for factor, val in (factors_ordered or list(factors.items()) if factors else []):
        fkey = factor if isinstance(factor, str) else factor
        vval = val if isinstance(val, (int, float)) else float(val) if isinstance(val, dict) else 0
        label = FACTOR_LABELS_15.get(fkey, str(fkey).capitalize())
        c.setFillColor(colors.black); c.setFont("Helvetica", 7)
        c.drawString(width/2 + 20, y_bar, label[:12])
        c.setFillColor(colors.HexColor("#e2e8f0")); c.roundRect(width/2 + 75, y_bar - 2, 80, 6, 2, fill=1)
        c.setFillColor(COLOR_DANGER if vval < 40 else (COLOR_WARNING if vval < 70 else COLOR_SUCCESS))
        c.roundRect(width/2 + 75, y_bar - 2, (float(vval)/100)*80, 6, 2, fill=1)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 6.5)
        c.drawString(width - 55, y_bar, f"{vval:.1f}%")
        y_bar -= 12

    # 5. SECTION 02: LOCATIONAL INTELLIGENCE (weather, geospatial passport, CNN, live telemetry)
    y_tab2 = _draw_section_header(c, 40, y_radar - 25, width, "Section 02: Locational Intelligence")
    y_curr = _draw_weather_module(c, data.get("weather"), 40, y_tab2 - 5, width)
    y_curr = _draw_geospatial_passport_module(c, data.get("geospatial_passport"), 40, y_curr - 10, width)
    y_curr = _draw_cnn_module(c, data.get("cnn_analysis"), 40, y_curr - 10, width)
    y_curr = _draw_telemetry_module(c, data.get("cnn_analysis"), 40, y_curr - 10, width)
    y_curr = _draw_terrain_module(c, data.get("terrain_analysis"), 40, y_curr - 10, width)

    # 6. SECTION 03: STRATEGIC UTILITY (site potential, roadmaps, interventions, AI projections)
    y_tab3 = _draw_section_header(c, 40, y_curr - 20, width, "Section 03: Strategic Utility")
    y_pot = y_tab3 - 15
    potentials = _calculate_site_potential(factors)
    for pot in potentials:
        if y_pot < 50: break
        c.setFillColor(pot['color']); c.roundRect(45, y_pot - 5, 120, 14, 4, fill=1)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6.5); c.drawString(50, y_pot, pot['label'])
        c.setFillColor(colors.black); c.setFont("Helvetica", 7)
        y_pot = _draw_wrapped_text(c, pot['reason'], 175, y_pot, width - 215, 8)
        y_pot -= 5
    intel = data.get("strategic_intelligence") or {}
    roadmap = intel.get("roadmap") or []
    for item in roadmap[:8]:
        if y_pot < 50: break
        c.setFillColor(colors.HexColor("#2d8a8a")); c.roundRect(45, y_pot - 5, 80, 10, 2, fill=1)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6); c.drawString(50, y_pot - 3, (item.get("task") or item.get("title") or str(item))[:25])
        c.setFillColor(colors.black); c.setFont("Helvetica", 7)
        y_pot = _draw_wrapped_text(c, (item.get("action") or item.get("reason") or "")[:80], 130, y_pot, width - 215, 7)
        y_pot -= 4
    interventions = intel.get("interventions") or []
    for item in interventions[:4]:
        if y_pot < 50: break
        c.setFillColor(colors.black); c.setFont("Helvetica", 7)
        y_pot = _draw_wrapped_text(c, (item.get("task") or str(item))[:100], 45, y_pot, width - 100, 7)
        y_pot -= 6
    forecast = data.get("temporal_forecast") or data.get("forecast") or {}
    if forecast:
        c.setFillColor(colors.HexColor("#0f172a")); c.roundRect(45, y_pot - 8, 200, 10, 2, fill=1)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6); c.drawString(50, y_pot - 6, "AI FUTURE PROJECTION (2030)")
        y_pot -= 12
        c.setFillColor(colors.black); c.setFont("Helvetica", 7)
        y_pot = _draw_wrapped_text(c, (forecast.get("summary") or forecast.get("narrative") or str(forecast))[:200], 45, y_pot, width - 100, 7)
        y_pot -= 8

    # 7. PAGE 2: EVIDENCE (15 factors in order)
    c.showPage()
    c.setFillColor(COLOR_DEEP_NAVY); c.rect(0, height - 60, width, 60, fill=1)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 12); c.drawString(40, height - 35, f"{title_prefix} - INTELLIGENCE EVIDENCE")
    y_ev = height - 90
    factors_meta_raw = data.get("explanation", {}).get("factors_meta", {})
    factors_meta = {}
    if isinstance(factors_meta_raw, dict):
        for k, v in factors_meta_raw.items():
            if isinstance(v, dict) and "value" not in v:
                for fk, fv in v.items():
                    factors_meta[fk] = fv if isinstance(fv, dict) else {"reason": str(fv)}
            else:
                factors_meta[k] = v if isinstance(v, dict) else {"reason": str(v)}
    for f_key in FACTOR_ORDER_15:
        if f_key not in factors and f_key not in factors_meta:
            continue
        meta = factors_meta.get(f_key) or {}
        reason = (meta.get("reason") or meta.get("evidence") or f"Score {factors.get(f_key, 0):.1f}/100.") if isinstance(meta, dict) else str(meta)
        val = float(factors.get(f_key, 0))
        if y_ev < 100: c.showPage(); y_ev = height - 80
        c.setFillColor(COLOR_DANGER if val < 40 else (COLOR_WARNING if val < 70 else COLOR_SUCCESS))
        c.rect(40, y_ev - 30, 2.5, 35, fill=1)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 9)
        c.drawString(50, y_ev, f"{f_key.upper()} ANALYSIS: {val:.1f}%")
        c.setFont("Helvetica", 8); y_ev = _draw_wrapped_text(c, reason, 50, y_ev - 15, width - 100, 10)
        y_ev -= 12
    for f_key, meta in factors_meta.items():
        if f_key in FACTOR_ORDER_15:
            continue
        if y_ev < 100: c.showPage(); y_ev = height - 80
        reason = (meta.get("reason") or meta.get("evidence", "")) if isinstance(meta, dict) else str(meta)
        val = float(factors.get(f_key, 0))
        c.setFillColor(COLOR_DANGER if val < 40 else (COLOR_WARNING if val < 70 else COLOR_SUCCESS))
        c.rect(40, y_ev - 30, 2.5, 35, fill=1)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 9)
        c.drawString(50, y_ev, f"{f_key.upper()} ANALYSIS: {val:.1f}%")
        c.setFont("Helvetica", 8); y_ev = _draw_wrapped_text(c, reason, 50, y_ev - 15, width - 100, 10)
        y_ev -= 12

def generate_land_report(data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    compare_data = data.get("compareData")
    _draw_location_analysis(c, data, "LOCATION A", width, height)
    if compare_data:
        c.showPage()
        _draw_location_analysis(c, compare_data, "LOCATION B", width, height)
    c.save(); buffer.seek(0)
    return buffer