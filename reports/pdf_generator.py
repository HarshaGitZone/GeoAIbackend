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
    """
    Helper to manually wrap text within a specific width on the PDF.
    """
    words = text.split(' ')
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

def _calculate_site_potential(factors):
    """
    Evaluates factor combinations to provide prescriptive insights.
    """
    potentials = []
    f = {k: float(v) for k, v in factors.items()}
    
    if any(f.get(k, 100) < 45 for k in ['flood', 'landslide', 'pollution']):
        hazards = [k.upper() for k in ['flood', 'landslide', 'pollution'] if f.get(k, 100) < 45]
        potentials.append({
            "label": "ENVIRONMENTAL CONSTRAINTS",
            "color": colors.HexColor("#ef4444"),
            "reason": f"CRITICAL RISK: Low safety scores in {', '.join(hazards)} indicate hazard vulnerability."
        })

    if f.get('flood', 0) > 50 and f.get('landslide', 0) > 50 and f.get('pollution', 0) > 40:
        potentials.append({
            "label": "RESIDENTIAL POTENTIAL",
            "color": colors.HexColor("#10b981"),
            "reason": "Highly viable for residential development driven by stable terrain and air quality."
        })

    if f.get('soil', 0) > 60 or f.get('rainfall', 0) > 60:
        potentials.append({
            "label": "AGRICULTURAL UTILITY",
            "color": colors.HexColor("#3b82f6"),
            "reason": "Excellent agricultural potential identified via soil nutrient density or rainfall patterns."
        })

    if f.get('proximity', 0) > 60 and f.get('landuse', 0) > 40:
        potentials.append({
            "label": "LOGISTICS & INDUSTRY",
            "color": colors.HexColor("#8b5cf6"),
            "reason": "Strategically positioned for industrial use due to infrastructure proximity."
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
    """Draws a weather context box"""
    if not weather: return y
    c.setFillColor(colors.HexColor("#f0f9ff"))
    c.roundRect(x, y - 40, width - 80, 40, 6, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y - 12, "LOCAL METEOROLOGICAL CONTEXT")
    c.setFont("Helvetica", 8)
    c.drawString(x + 10, y - 25, f"Temperature: {weather.get('temp', 'N/A')}°C  |  Humidity: {weather.get('humidity', 'N/A')}%")
    c.drawString(x + 10, y - 35, f"Current Conditions: {weather.get('description', 'N/A')}")
    return y - 50

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

    # 4. SECTION 01: SUITABILITY
    y_tab1 = _draw_section_header(c, 40, y_score - 50, width, "Section 01: Suitability Intelligence")
    
    factors = data.get("factors", {})
    y_radar = y_tab1 - 165
    if factors:
        labels = [k.capitalize() for k in factors.keys()]
        values = [float(v) for v in factors.values()]
        fig = plt.figure(figsize=(3, 3), dpi=150) 
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ax.fill(angles + angles[:1], values + values[:1], color='#06b6d4', alpha=0.3)
        ax.plot(angles + angles[:1], values + values[:1], color='#06b6d4', linewidth=1)
        ax.set_yticklabels([]); ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=6)
        chart_io = io.BytesIO(); plt.savefig(chart_io, format='png', transparent=True); plt.close(fig); chart_io.seek(0)
        c.drawImage(ImageReader(chart_io), 35, y_radar, width=160, height=160, mask='auto')

    # Factor Bars
    y_bar = y_tab1 - 15
    for factor, val in factors.items():
        c.setFillColor(colors.black); c.setFont("Helvetica", 7.5)
        c.drawString(width/2 + 20, y_bar, factor.capitalize())
        c.setFillColor(colors.HexColor("#e2e8f0")); c.roundRect(width/2 + 75, y_bar - 2, 80, 6, 2, fill=1)
        c.setFillColor(COLOR_DANGER if val < 40 else (COLOR_WARNING if val < 70 else COLOR_SUCCESS))
        c.roundRect(width/2 + 75, y_bar - 2, (float(val)/100)*80, 6, 2, fill=1)
        # Restore numerical values label
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 6.5)
        c.drawString(width - 55, y_bar, f"{val:.1f}%")
        y_bar -= 15

    # 5. SECTION 02: ENVIRONMENTAL
    y_tab2 = _draw_section_header(c, 40, y_radar - 25, width, "Section 02: Environmental Context")
    y_curr = _draw_terrain_module(c, data.get("terrain_analysis"), 40, y_tab2 - 5, width)
    y_curr = _draw_weather_module(c, data.get("weather"), 40, y_curr - 10, width)

    # 6. SECTION 03: STRATEGIC INSIGHTS
    y_tab3 = _draw_section_header(c, 40, y_curr - 20, width, "Section 03: Strategic Site Insights")
    y_pot = y_tab3 - 15
    potentials = _calculate_site_potential(factors)
    for pot in potentials:
        if y_pot < 50: break
        c.setFillColor(pot['color']); c.roundRect(45, y_pot - 5, 120, 14, 4, fill=1)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6.5); c.drawString(50, y_pot, pot['label'])
        c.setFillColor(colors.black); c.setFont("Helvetica", 7)
        y_pot = _draw_wrapped_text(c, pot['reason'], 175, y_pot, width - 215, 8)
        y_pot -= 5

    # 7. PAGE 2: EVIDENCE
    c.showPage()
    c.setFillColor(COLOR_DEEP_NAVY); c.rect(0, height - 60, width, 60, fill=1)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 12); c.drawString(40, height - 35, f"{title_prefix} - INTELLIGENCE EVIDENCE")
    
    y_ev = height - 90
    factors_meta = data.get("explanation", {}).get("factors_meta", {})
    for f_key, meta in factors_meta.items():
        if y_ev < 100: c.showPage(); y_ev = height - 80
        val = float(factors.get(f_key, 0))
        c.setFillColor(COLOR_DANGER if val < 40 else (COLOR_WARNING if val < 70 else COLOR_SUCCESS))
        c.rect(40, y_ev - 30, 2.5, 35, fill=1)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 9)
        c.drawString(50, y_ev, f"{f_key.upper()} ANALYSIS: {val:.1f}%")
        c.setFont("Helvetica", 8); y_ev = _draw_wrapped_text(c, meta.get("reason", ""), 50, y_ev - 15, width - 100, 10)
        y_ev -= 15

def generate_land_report(data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    _draw_location_analysis(c, data, "LOCATION A", width, height)
    compare_data = data.get("compareData")
    if compare_data:
        c.showPage()
        _draw_location_analysis(c, compare_data, "LOCATION B", width, height)
    c.save(); buffer.seek(0)
    return buffer