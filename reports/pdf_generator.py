# import matplotlib
# matplotlib.use('Agg')  
# import matplotlib.pyplot as plt
# import numpy as np
# import io
# import os
# import requests
# import qrcode
# from datetime import datetime
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from reportlab.lib import colors
# from reportlab.lib.utils import ImageReader

# def _draw_wrapped_text(c, text, x, y, max_width, line_height):
#     """Helper to manually wrap text within a specific width on the PDF."""
#     if not text:
#         return y - line_height
#     words = str(text).split(' ')
#     line = ""
#     for word in words:
#         if c.stringWidth(line + word + " ", "Helvetica", 7.5) < max_width:
#             line += word + " "
#         else:
#             c.drawString(x, y, line)
#             line = word + " "
#             y -= line_height
#     c.drawString(x, y, line)
#     return y - line_height

# # 15 factors in order of 5 categories (same as frontend)
# FACTOR_ORDER_15 = [
#     'slope', 'elevation', 'vegetation', 'soil', 'pollution',
#     'flood', 'water', 'drainage', 'rainfall', 'thermal', 'intensity',
#     'landuse', 'infrastructure', 'population'
# ]
# FACTOR_LABELS_15 = {k: k.capitalize() for k in FACTOR_ORDER_15}
# FACTOR_LABELS_15['infrastructure'] = 'Infra'

# def _flatten_factors(data):
#     """Convert nested factors (by category) to flat { factor_name: value } and ordered list for radar."""
#     raw = data.get("factors", {})
#     if not raw:
#         return {}, []
#     flat = {}
#     if isinstance(raw, dict) and any(isinstance(v, dict) for v in raw.values()):
#         for cat, cat_data in raw.items():
#             if not isinstance(cat_data, dict):
#                 continue
#             for fkey, fval in cat_data.items():
#                 if isinstance(fval, dict):
#                     s = fval.get("scaled_score")
#                     v = fval.get("value", 0)
#                     flat[fkey] = max(0, min(100, float(s if s is not None else v)))
#                 else:
#                     flat[fkey] = max(0, min(100, float(fval) if fval is not None else 0))
#     else:
#         for k, v in raw.items():
#             flat[k] = float(v) if not isinstance(v, dict) else float(v.get("value", 0))
#     ordered = [(k, flat.get(k, 0)) for k in FACTOR_ORDER_15]
#     extra = [(k, v) for k, v in flat.items() if k not in FACTOR_ORDER_15]
#     ordered = ordered + extra
#     if not ordered:
#         ordered = list(flat.items())
#     return flat, ordered

# def _calculate_site_potential(factors):
#     """Evaluates 15-factor combinations for prescriptive insights."""
#     potentials = []
#     f = {k: float(v) for k, v in factors.items()}

#     if any(f.get(k, 100) < 45 for k in ['flood', 'pollution', 'drainage']):
#         hazards = [k.upper() for k in ['flood', 'pollution', 'drainage'] if f.get(k, 100) < 45]
#         potentials.append({
#             "label": "ENVIRONMENTAL CONSTRAINTS",
#             "color": colors.HexColor("#ef4444"),
#             "reason": f"CRITICAL RISK: Low scores in {', '.join(hazards)} indicate hazard vulnerability."
#         })
#     if f.get('flood', 0) > 50 and f.get('pollution', 0) > 40 and f.get('slope', 100) < 25:
#         potentials.append({
#             "label": "RESIDENTIAL POTENTIAL",
#             "color": colors.HexColor("#10b981"),
#             "reason": "Viable for residential development: stable terrain, air quality, and flood safety."
#         })
#     if f.get('soil', 0) > 60 or f.get('rainfall', 0) > 60 or f.get('vegetation', 0) > 50:
#         potentials.append({
#             "label": "AGRICULTURAL UTILITY",
#             "color": colors.HexColor("#3b82f6"),
#             "reason": "Agricultural potential from soil, rainfall, or vegetation indicators."
#         })
#     if f.get('infrastructure', 0) > 60 and f.get('landuse', 0) > 40:
#         potentials.append({
#             "label": "LOGISTICS & INDUSTRY",
#             "color": colors.HexColor("#8b5cf6"),
#             "reason": "Strategically positioned for industrial use due to infrastructure and land use."
#         })
#     return potentials 

# def _draw_section_header(c, x, y, width, text):
#     """Draws a navy blue sub-header to separate the PDF into 'Tabs'"""
#     COLOR_DEEP_NAVY = colors.HexColor("#0f172a")
#     c.setFillColor(COLOR_DEEP_NAVY)
#     c.roundRect(x, y, width - 80, 18, 4, fill=1, stroke=0)
#     c.setFillColor(colors.white)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(x + 10, y + 5, text.upper())
#     return y - 10

# def _draw_terrain_module(c, terrain, x, y, width):
#     """Draws the Terrain & Slope box - Always displays with fallback values"""
#     # Always show terrain analysis, even if data is missing
#     slope = 0
#     verdict = "Data Unavailable"
    
#     if terrain:
#         slope = terrain.get('slope_percent', 0) or 0
#         verdict = terrain.get('verdict', 'N/A') or 'N/A'
#     else:
#         # Fallback values when terrain data is missing
#         verdict = "Terrain data not available"
    
#     c.setFillColor(colors.HexColor("#f8fafc"))
#     c.roundRect(x, y - 50, width - 80, 50, 6, fill=1, stroke=0)
#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(x + 10, y - 12, "TERRAIN & SLOPE ANALYSIS")
    
#     # Color coding based on slope value
#     slope_color = colors.HexColor("#ef4444") if slope > 15 else (colors.HexColor("#f59e0b") if slope > 5 else colors.HexColor("#10b981"))
#     c.setFillColor(slope_color)
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(x + 10, y - 28, f"{slope:.1f}% Gradient")
#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Oblique", 7.5)
#     c.drawString(x + 10, y - 42, f"Verdict: {verdict}")
#     return y - 60

# def _draw_weather_module(c, weather, x, y, width):
#     """Draws weather with full detail (temp, humidity, wind, pressure, code)."""
#     if not weather: return y
#     c.setFillColor(colors.HexColor("#f0f9ff"))
#     c.roundRect(x, y - 58, width - 80, 58, 6, fill=1, stroke=0)
#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(x + 10, y - 12, "WEATHER (LIVE TELEMETRY)")
#     c.setFont("Helvetica", 8)
#     c.drawString(x + 10, y - 24, f"Temperature: {weather.get('temp', 'N/A')}°C  |  Humidity: {weather.get('humidity', 'N/A')}%")
#     c.drawString(x + 10, y - 34, f"Conditions: {weather.get('description', 'N/A')}")
#     c.drawString(x + 10, y - 44, f"Wind: {weather.get('wind_speed_kmh', weather.get('wind_speed', 'N/A'))} km/h  |  Pressure: {weather.get('pressure_hpa', 'N/A')} hPa")
#     c.drawString(x + 10, y - 54, f"Weather code: {weather.get('weather_code', 'N/A')}")
#     return y - 68

# def _draw_geospatial_passport_module(c, passport, x, y, width):
#     """Draws geospatial passport summary."""
#     if not passport: return y
#     c.setFillColor(colors.HexColor("#ecfdf5"))
#     c.roundRect(x, y - 72, width - 80, 72, 6, fill=1, stroke=0)
#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(x + 10, y - 12, "GEOSPATIAL PASSPORT")
#     c.setFont("Helvetica", 7.5)
#     c.drawString(x + 10, y - 24, f"Slope: {passport.get('slope_percent', 'N/A')}% (suit. {passport.get('slope_suitability', 'N/A')})  |  Elev: {passport.get('elevation_m', 'N/A')}m")
#     c.drawString(x + 10, y - 34, f"Vegetation: {passport.get('vegetation_score', 'N/A')}  |  Landuse: {passport.get('landuse_class', 'N/A')}  |  Water dist: {passport.get('water_distance_km', 'N/A')} km")
#     c.drawString(x + 10, y - 44, f"Flood safety: {passport.get('flood_safety_score', 'N/A')}  |  Rainfall: {passport.get('rainfall_mm', 'N/A')} mm")
#     c.drawString(x + 10, y - 54, f"Risk: {passport.get('risk_summary', 'N/A')}  |  Categories: {passport.get('category_breakdown') or 'N/A'}")
#     return y - 82

# def _draw_cnn_module(c, cnn, x, y, width):
#     """Draws CNN classification and live telemetry."""
#     if not cnn: return y
#     h = 70
#     c.setFillColor(colors.HexColor("#fef3c7"))
#     c.roundRect(x, y - h, width - 80, h, 6, fill=1, stroke=0)
#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(x + 10, y - 12, "CNN CLASSIFICATION (VISUAL INTELLIGENCE)")
#     c.setFont("Helvetica", 8)
#     c.drawString(x + 10, y - 24, f"Class: {cnn.get('class', 'N/A')}  |  Confidence: {cnn.get('confidence_display', cnn.get('confidence', 'N/A'))}")
#     y_cur = y - 32
#     if cnn.get('note'):
#         y_cur = _draw_wrapped_text(c, (cnn['note'] or '')[:150], x + 10, y_cur, width - 100, 8); y_cur -= 4
#     tel = cnn.get("telemetry") or {}
#     c.setFont("Helvetica", 7)
#     c.drawString(x + 10, y_cur - 10, f"RES: {tel.get('resolution_m_per_px', 'N/A')} m/px  |  SENSOR: {tel.get('tile_url_source', 'N/A')}  |  MODEL: {tel.get('model', 'N/A')}")
#     if tel.get('verified_by'):
#         c.drawString(x + 10, y_cur - 20, f"Verified: {tel.get('verified_by')}")
#     return y - h - 10

# def _draw_telemetry_module(c, cnn, x, y, width):
#     """Draws live telemetry details from CNN/forensics."""
#     tel = (cnn or {}).get("telemetry") or {}
#     if not tel: return y
#     c.setFillColor(colors.HexColor("#e0e7ff"))
#     c.roundRect(x, y - 48, width - 80, 48, 6, fill=1, stroke=0)
#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(x + 10, y - 12, "LIVE TELEMETRY")
#     c.setFont("Helvetica", 7.5)
#     for i, (k, v) in enumerate(list(tel.items())[:6]):
#         if k in ('interpretation',): continue
#         c.drawString(x + 10, y - 22 - i * 10, f"{k}: {v}")
#     return y - 58

# def _draw_location_analysis(c, data, title_prefix, width, height):
#     COLOR_DANGER = colors.HexColor("#ef4444")
#     COLOR_WARNING = colors.HexColor("#f59e0b")
#     COLOR_SUCCESS = colors.HexColor("#10b981")
#     COLOR_DEEP_NAVY = colors.HexColor("#0f172a") 

#     # 1. HEADER
#     c.setFillColor(COLOR_DEEP_NAVY)
#     c.rect(0, height - 100, width, 100, fill=1, stroke=0)
#     # --- NEW: QR CODE SECTION (Top Right) ---
#     # Expecting 'shareLink' to be passed in the data payload from frontend
#     share_url = data.get('shareLink')
#     if share_url:
#         try:
#             # Generate QR Code
#             qr = qrcode.QRCode(version=1, box_size=10, border=2)
#             qr.add_data(share_url)
#             qr.make(fit=True)
#             qr_img = qr.make_image(fill_color="black", back_color="white")
            
#             # Convert to ReportLab-friendly format
#             qr_buffer = io.BytesIO()
#             qr_img.save(qr_buffer, format='PNG')
#             qr_buffer.seek(0)
            
#             # Draw QR Box and Label
#             c.setFillColor(colors.white)
#             c.roundRect(width - 95, height - 90, 80, 80, 4, fill=1, stroke=0)
#             c.drawImage(ImageReader(qr_buffer), width - 90, height - 82, width=70, height=70)
#             # --- STYLED "BUTTON" SECTION ---
#             # Draw a green button-like rectangle
#             btn_x, btn_y, btn_w, btn_h = width - 90, height - 88, 70, 10
#             c.setFillColor(COLOR_SUCCESS)
#             c.roundRect(btn_x, btn_y, btn_w, btn_h, 2, fill=1, stroke=0)

#             c.setFillColor(colors.white)
#             c.setFont("Helvetica-Bold", 6)
#             # c.drawCentredString(width - 55, height - 88, "SCAN FOR LIVE")
#             c.drawCentredString(width - 55, btn_y + 3, "OPEN LINK")
#             # CREATE THE CLICKABLE LINK AREA
#             # This makes the button area in the PDF act as a hyperlink
#             c.linkURL(share_url, (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h), relative=0)
            
            
#         except Exception as e:
#             print(f"QR Generation Error: {e}")
#     c.setFillColor(colors.white)
#     c.setFont("Helvetica-Bold", 20)
#     c.drawCentredString(width / 2, height - 40, "GeoAI – Land Suitability Certificate")
#     c.setFont("Helvetica-Bold", 12)
#     c.drawCentredString(width / 2, height - 60, f"{title_prefix}: {data.get('locationName', 'Site Analysis')}".upper())
    
#     # CRITICAL FIX: Coordinate mapping for Map/Header
#     loc = data.get('location', {})
#     lat = loc.get('latitude') or loc.get('lat') or 0.0
#     lon = loc.get('longitude') or loc.get('lng') or 0.0
    
#     timestamp = datetime.now().strftime('%d %b %Y | %H:%M:%S IST')
#     c.setFont("Helvetica", 8)
#     c.drawCentredString(width / 2, height - 80, f"{timestamp}  •  LAT: {lat} | LNG: {lon}")

#     # 2. MINI MAP
#     y_map = height - 210
#     try:
#         # Mini map logic must use the same fallbacks to ensure image displays
#         map_url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&z=13&l=sat&size=500,140"
#         map_res = requests.get(map_url, timeout=5)
#         if map_res.status_code == 200:
#             map_img = ImageReader(io.BytesIO(map_res.content))
#             c.drawImage(map_img, 40, y_map, width=width-80, height=110)
#             c.setStrokeColor(colors.white)
#             c.rect(40, y_map, width-80, 110, stroke=1, fill=0)
#     except Exception as e:
#         print(f"Map Error: {e}")
#         c.setFillColor(colors.lightgrey); c.rect(40, y_map, width-80, 110, fill=1)

#     # 3. SCORECARD
#     score = float(data.get('suitability_score', 0))
#     score_color = COLOR_DANGER if score < 40 else (COLOR_WARNING if score < 70 else COLOR_SUCCESS)
#     y_score = y_map - 45
#     c.setFillColor(score_color); c.setFont("Helvetica-Bold", 28)
#     c.drawString(45, y_score, f"{score:.1f}")
#     c.setFont("Helvetica-Bold", 10); c.drawString(45, y_score - 15, f"GRADE: {data.get('label', 'N/A').upper()}")

#     # 4. SECTION 01: SUITABILITY (15 factors: radar + bars side by side)
#     y_tab1 = _draw_section_header(c, 40, y_score - 50, width, "Section 01: Suitability Intelligence (15 Factors)")
#     factors_flat, factors_ordered = _flatten_factors(data)
#     factors = factors_flat
#     y_radar = y_tab1 - 165
#     if factors_ordered:
#         labels = [FACTOR_LABELS_15.get(k, k.capitalize()) for k, _ in factors_ordered]
#         values = [v for _, v in factors_ordered]
#         fig = plt.figure(figsize=(3, 3), dpi=150)
#         ax = fig.add_subplot(111, polar=True)
#         angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#         ax.fill(angles + angles[:1], values + values[:1], color='#06b6d4', alpha=0.3)
#         ax.plot(angles + angles[:1], values + values[:1], color='#06b6d4', linewidth=1)
#         ax.set_xticks(angles)
#         ax.set_xticklabels(labels, fontsize=5)
#         ax.set_yticks([0, 25, 50, 75, 100])
#         ax.set_yticklabels([])
#         chart_io = io.BytesIO(); plt.savefig(chart_io, format='png', transparent=True); plt.close(fig); chart_io.seek(0)
#         c.drawImage(ImageReader(chart_io), 35, y_radar, width=160, height=160, mask='auto')

#     y_bar = y_tab1 - 15
#     for factor, val in (factors_ordered or list(factors.items()) if factors else []):
#         fkey = factor if isinstance(factor, str) else factor
#         vval = val if isinstance(val, (int, float)) else float(val) if isinstance(val, dict) else 0
#         label = FACTOR_LABELS_15.get(fkey, str(fkey).capitalize())
#         c.setFillColor(colors.black); c.setFont("Helvetica", 7)
#         c.drawString(width/2 + 20, y_bar, label[:12])
#         c.setFillColor(colors.HexColor("#e2e8f0")); c.roundRect(width/2 + 75, y_bar - 2, 80, 6, 2, fill=1)
#         c.setFillColor(COLOR_DANGER if vval < 40 else (COLOR_WARNING if vval < 70 else COLOR_SUCCESS))
#         c.roundRect(width/2 + 75, y_bar - 2, (float(vval)/100)*80, 6, 2, fill=1)
#         c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 6.5)
#         c.drawString(width - 55, y_bar, f"{vval:.1f}%")
#         y_bar -= 12

#     # 5. SECTION 02: LOCATIONAL INTELLIGENCE (weather, geospatial passport, CNN, live telemetry)
#     y_tab2 = _draw_section_header(c, 40, y_radar - 25, width, "Section 02: Locational Intelligence")
#     y_curr = _draw_weather_module(c, data.get("weather"), 40, y_tab2 - 5, width)
#     y_curr = _draw_geospatial_passport_module(c, data.get("geospatial_passport"), 40, y_curr - 10, width)
#     y_curr = _draw_cnn_module(c, data.get("cnn_analysis"), 40, y_curr - 10, width)
#     y_curr = _draw_telemetry_module(c, data.get("cnn_analysis"), 40, y_curr - 10, width)
#     y_curr = _draw_terrain_module(c, data.get("terrain_analysis"), 40, y_curr - 10, width)

#     # 6. SECTION 03: STRATEGIC UTILITY (site potential, roadmaps, interventions, AI projections)
#     y_tab3 = _draw_section_header(c, 40, y_curr - 20, width, "Section 03: Strategic Utility")
#     y_pot = y_tab3 - 15
#     potentials = _calculate_site_potential(factors)
#     for pot in potentials:
#         if y_pot < 50: break
#         c.setFillColor(pot['color']); c.roundRect(45, y_pot - 5, 120, 14, 4, fill=1)
#         c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6.5); c.drawString(50, y_pot, pot['label'])
#         c.setFillColor(colors.black); c.setFont("Helvetica", 7)
#         y_pot = _draw_wrapped_text(c, pot['reason'], 175, y_pot, width - 215, 8)
#         y_pot -= 5
#     intel = data.get("strategic_intelligence") or {}
#     roadmap = intel.get("roadmap") or []
#     for item in roadmap[:8]:
#         if y_pot < 50: break
#         c.setFillColor(colors.HexColor("#2d8a8a")); c.roundRect(45, y_pot - 5, 80, 10, 2, fill=1)
#         c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6); c.drawString(50, y_pot - 3, (item.get("task") or item.get("title") or str(item))[:25])
#         c.setFillColor(colors.black); c.setFont("Helvetica", 7)
#         y_pot = _draw_wrapped_text(c, (item.get("action") or item.get("reason") or "")[:80], 130, y_pot, width - 215, 7)
#         y_pot -= 4
#     interventions = intel.get("interventions") or []
#     for item in interventions[:4]:
#         if y_pot < 50: break
#         c.setFillColor(colors.black); c.setFont("Helvetica", 7)
#         y_pot = _draw_wrapped_text(c, (item.get("task") or str(item))[:100], 45, y_pot, width - 100, 7)
#         y_pot -= 6
#     forecast = data.get("temporal_forecast") or data.get("forecast") or {}
#     if forecast:
#         c.setFillColor(colors.HexColor("#0f172a")); c.roundRect(45, y_pot - 8, 200, 10, 2, fill=1)
#         c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 6); c.drawString(50, y_pot - 6, "AI FUTURE PROJECTION (2030)")
#         y_pot -= 12
#         c.setFillColor(colors.black); c.setFont("Helvetica", 7)
#         y_pot = _draw_wrapped_text(c, (forecast.get("summary") or forecast.get("narrative") or str(forecast))[:200], 45, y_pot, width - 100, 7)
#         y_pot -= 8

#     # 7. PAGE 2: EVIDENCE (15 factors in order)
#     c.showPage()
#     c.setFillColor(COLOR_DEEP_NAVY); c.rect(0, height - 60, width, 60, fill=1)
#     c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 12); c.drawString(40, height - 35, f"{title_prefix} - INTELLIGENCE EVIDENCE")
#     y_ev = height - 90
#     factors_meta_raw = data.get("explanation", {}).get("factors_meta", {})
#     factors_meta = {}
#     if isinstance(factors_meta_raw, dict):
#         for k, v in factors_meta_raw.items():
#             if isinstance(v, dict) and "value" not in v:
#                 for fk, fv in v.items():
#                     factors_meta[fk] = fv if isinstance(fv, dict) else {"reason": str(fv)}
#             else:
#                 factors_meta[k] = v if isinstance(v, dict) else {"reason": str(v)}
#     for f_key in FACTOR_ORDER_15:
#         if f_key not in factors and f_key not in factors_meta:
#             continue
#         meta = factors_meta.get(f_key) or {}
#         reason = (meta.get("reason") or meta.get("evidence") or f"Score {factors.get(f_key, 0):.1f}/100.") if isinstance(meta, dict) else str(meta)
#         val = float(factors.get(f_key, 0))
#         if y_ev < 100: c.showPage(); y_ev = height - 80
#         c.setFillColor(COLOR_DANGER if val < 40 else (COLOR_WARNING if val < 70 else COLOR_SUCCESS))
#         c.rect(40, y_ev - 30, 2.5, 35, fill=1)
#         c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 9)
#         c.drawString(50, y_ev, f"{f_key.upper()} ANALYSIS: {val:.1f}%")
#         c.setFont("Helvetica", 8); y_ev = _draw_wrapped_text(c, reason, 50, y_ev - 15, width - 100, 10)
#         y_ev -= 12
#     for f_key, meta in factors_meta.items():
#         if f_key in FACTOR_ORDER_15:
#             continue
#         if y_ev < 100: c.showPage(); y_ev = height - 80
#         reason = (meta.get("reason") or meta.get("evidence", "")) if isinstance(meta, dict) else str(meta)
#         val = float(factors.get(f_key, 0))
#         c.setFillColor(COLOR_DANGER if val < 40 else (COLOR_WARNING if val < 70 else COLOR_SUCCESS))
#         c.rect(40, y_ev - 30, 2.5, 35, fill=1)
#         c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 9)
#         c.drawString(50, y_ev, f"{f_key.upper()} ANALYSIS: {val:.1f}%")
#         c.setFont("Helvetica", 8); y_ev = _draw_wrapped_text(c, reason, 50, y_ev - 15, width - 100, 10)
#         y_ev -= 12

# def generate_land_report(data):
#     buffer = io.BytesIO()
#     c = canvas.Canvas(buffer, pagesize=A4)
#     width, height = A4
#     compare_data = data.get("compareData")
#     _draw_location_analysis(c, data, "LOCATION A", width, height)
#     if compare_data:
#         c.showPage()
#         _draw_location_analysis(c, compare_data, "LOCATION B", width, height)
#     c.save(); buffer.seek(0)
#     return buffer

# import matplotlib
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import numpy as np
# import io
# import requests
# import qrcode
# from datetime import datetime

# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from reportlab.lib import colors
# from reportlab.lib.utils import ImageReader


# # ============================================================
# # GLOBAL ORDER (MATCHES YOUR FRONTEND)
# # ============================================================

# CATEGORY_ORDER = [
#     "physical",
#     "hydrology",
#     "environmental",
#     "climatic",
#     "socio_econ",
#     "risk_resilience",
# ]

# CATEGORY_TITLES = {
#     "physical": "PHYSICAL",
#     "hydrology": "HYDROLOGY",
#     "environmental": "ENVIRONMENTAL",
#     "climatic": "CLIMATIC",
#     "socio_econ": "SOCIO-ECONOMIC",
#     "risk_resilience": "RISK & RESILIENCE",
# }

# FACTOR_ORDER_BY_CATEGORY = {
#     "physical": ["elevation", "ruggedness", "slope", "stability"],
#     "environmental": ["biodiversity", "heat_island", "pollution", "soil", "vegetation"],
#     "hydrology": ["drainage", "flood", "groundwater", "water"],
#     "climatic": ["intensity", "rainfall", "thermal"],
#     "socio_econ": ["infrastructure", "landuse", "population"],
#     "risk_resilience": ["climate_change", "habitability", "multi_hazard", "recovery"],
# }

# FACTOR_LABELS = {
#     "slope": "SLOPE",
#     "elevation": "ELEVATION",
#     "ruggedness": "RUGGEDNESS",
#     "stability": "STABILITY",
#     "flood": "FLOOD RISK",
#     "water": "WATER PROXIMITY",
#     "drainage": "DRAINAGE",
#     "groundwater": "GROUNDWATER",
#     "vegetation": "VEGETATION",
#     "soil": "SOIL QUALITY",
#     "pollution": "AIR POLLUTION",
#     "biodiversity": "BIODIVERSITY",
#     "heat_island": "HEAT ISLAND",
#     "rainfall": "RAINFALL",
#     "thermal": "THERMAL COMFORT",
#     "intensity": "HEAT STRESS",
#     "landuse": "LANDUSE",
#     "infrastructure": "INFRASTRUCTURE",
#     "population": "POPULATION",
#     "multi_hazard": "MULTI-HAZARD",
#     "climate_change": "CLIMATE CHANGE",
#     "recovery": "RECOVERY CAPACITY",
#     "habitability": "HABITABILITY",
# }


# # ============================================================
# # COLORS
# # ============================================================

# COLOR_DEEP_NAVY = colors.HexColor("#0f172a")
# COLOR_DANGER = colors.HexColor("#ef4444")
# COLOR_WARNING = colors.HexColor("#f59e0b")
# COLOR_SUCCESS = colors.HexColor("#10b981")

# COLOR_CARD_BG = colors.HexColor("#f8fafc")
# COLOR_BORDER = colors.HexColor("#e2e8f0")


# # ============================================================
# # BASIC HELPERS
# # ============================================================

# def _score_color(score: float):
#     if score < 40:
#         return COLOR_DANGER
#     if score < 70:
#         return COLOR_WARNING
#     return COLOR_SUCCESS


# def _safe_num(v, default=0.0):
#     try:
#         if v is None:
#             return float(default)
#         return float(v)
#     except Exception:
#         return float(default)


# def _wrap_text_lines(c, text, max_width, font_name="Helvetica", font_size=8):
#     """Returns wrapped lines list for text."""
#     if not text:
#         return []

#     c.setFont(font_name, font_size)
#     words = str(text).split()
#     lines = []
#     line = ""

#     for w in words:
#         trial = (line + " " + w).strip()
#         if c.stringWidth(trial, font_name, font_size) <= max_width:
#             line = trial
#         else:
#             if line:
#                 lines.append(line)
#             line = w

#     if line:
#         lines.append(line)

#     return lines


# # ============================================================
# # PAGINATION ENGINE (THIS IS THE FIX)
# # ============================================================

# class PDFCursor:
#     def __init__(self, c, width, height):
#         self.c = c
#         self.width = width
#         self.height = height
#         self.x_margin = 40
#         self.y_top = height - 20
#         self.y = height - 20
#         self.page = 1

#     def new_page(self):
#         self.c.showPage()
#         self.page += 1
#         self.y = self.height - 20

#     def ensure_space(self, needed_height, min_y=55):
#         if self.y - needed_height < min_y:
#             self.new_page()


# # ============================================================
# # FACTOR FLATTENING (23 factors)
# # ============================================================

# def _flatten_factors_23(data):
#     raw = data.get("factors", {}) or {}
#     flat = {}

#     if isinstance(raw, dict):
#         for cat, cat_data in raw.items():
#             if not isinstance(cat_data, dict):
#                 continue
#             for fkey, fval in cat_data.items():
#                 if isinstance(fval, dict):
#                     s = fval.get("scaled_score")
#                     v = fval.get("value", 0)
#                     flat[fkey] = max(0, min(100, _safe_num(s if s is not None else v, 0)))
#                 else:
#                     flat[fkey] = max(0, min(100, _safe_num(fval, 0)))

#     return flat


# def _get_factor_list_in_ui_order(data):
#     raw = data.get("factors", {}) or {}
#     ordered = []

#     for cat in CATEGORY_ORDER:
#         cat_data = raw.get(cat) or {}
#         if not isinstance(cat_data, dict):
#             continue
#         keys = FACTOR_ORDER_BY_CATEGORY.get(cat, list(cat_data.keys()))
#         for k in keys:
#             if k in cat_data:
#                 ordered.append(k)

#     # Add any extras at end (rare)
#     for cat, cat_data in raw.items():
#         if not isinstance(cat_data, dict):
#             continue
#         for k in cat_data.keys():
#             if k not in ordered:
#                 ordered.append(k)

#     return ordered


# # ============================================================
# # HEADER + MINIMAP
# # ============================================================

# def _draw_main_header(cur: PDFCursor, data, title_prefix):
#     c = cur.c
#     w, h = cur.width, cur.height

#     c.setFillColor(COLOR_DEEP_NAVY)
#     c.rect(0, h - 100, w, 100, fill=1, stroke=0)

#     # QR + Button
#     share_url = data.get("shareLink")
#     if share_url:
#         try:
#             qr = qrcode.QRCode(version=1, box_size=10, border=2)
#             qr.add_data(share_url)
#             qr.make(fit=True)
#             qr_img = qr.make_image(fill_color="black", back_color="white")

#             qr_buffer = io.BytesIO()
#             qr_img.save(qr_buffer, format="PNG")
#             qr_buffer.seek(0)

#             c.setFillColor(colors.white)
#             c.roundRect(w - 95, h - 90, 80, 80, 4, fill=1, stroke=0)
#             c.drawImage(ImageReader(qr_buffer), w - 90, h - 82, width=70, height=70)

#             btn_x, btn_y, btn_w, btn_h = w - 90, h - 88, 70, 10
#             c.setFillColor(COLOR_SUCCESS)
#             c.roundRect(btn_x, btn_y, btn_w, btn_h, 2, fill=1, stroke=0)

#             c.setFillColor(colors.white)
#             c.setFont("Helvetica-Bold", 6)
#             c.drawCentredString(w - 55, btn_y + 3, "OPEN LINK")
#             c.linkURL(share_url, (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h), relative=0)

#         except Exception:
#             pass

#     c.setFillColor(colors.white)
#     c.setFont("Helvetica-Bold", 20)
#     c.drawCentredString(w / 2, h - 40, "GeoAI – Land Suitability Report")

#     c.setFont("Helvetica-Bold", 12)
#     c.drawCentredString(w / 2, h - 60, f"{title_prefix}: {data.get('locationName', 'Site')}".upper())

#     loc = data.get("location", {}) or {}
#     lat = loc.get("latitude") or loc.get("lat") or 0.0
#     lon = loc.get("longitude") or loc.get("lng") or 0.0

#     timestamp = datetime.now().strftime("%d %b %Y | %H:%M:%S IST")
#     c.setFont("Helvetica", 8)
#     c.drawCentredString(w / 2, h - 80, f"{timestamp}  •  LAT: {lat} | LNG: {lon}")

#     cur.y = h - 110


# def _draw_minimap(cur: PDFCursor, data):
#     c = cur.c
#     w = cur.width

#     loc = data.get("location", {}) or {}
#     lat = loc.get("latitude") or loc.get("lat") or 0.0
#     lon = loc.get("longitude") or loc.get("lng") or 0.0

#     cur.ensure_space(130)
#     y_map = cur.y - 120

#     try:
#         map_url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&z=13&l=sat&size=500,140"
#         map_res = requests.get(map_url, timeout=6)
#         if map_res.status_code == 200:
#             map_img = ImageReader(io.BytesIO(map_res.content))
#             c.drawImage(map_img, 40, y_map, width=w - 80, height=110)
#             c.setStrokeColor(colors.white)
#             c.rect(40, y_map, w - 80, 110, stroke=1, fill=0)
#         else:
#             c.setFillColor(colors.lightgrey)
#             c.rect(40, y_map, w - 80, 110, fill=1, stroke=0)
#     except Exception:
#         c.setFillColor(colors.lightgrey)
#         c.rect(40, y_map, w - 80, 110, fill=1, stroke=0)

#     cur.y = y_map - 20


# def _draw_scorecard(cur: PDFCursor, data):
#     c = cur.c

#     score = _safe_num(data.get("suitability_score"), 0)
#     grade = str(data.get("label", "N/A")).upper()
#     col = _score_color(score)

#     cur.ensure_space(40)
#     c.setFillColor(col)
#     c.setFont("Helvetica-Bold", 28)
#     c.drawString(45, cur.y - 10, f"{score:.1f}")

#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Bold", 10)
#     c.drawString(45, cur.y - 25, f"GRADE: {grade}")

#     cur.y -= 45


# # ============================================================
# # SECTION HEADER (TAB)
# # ============================================================

# def _draw_tab_header(cur: PDFCursor, text):
#     c = cur.c
#     w = cur.width

#     cur.ensure_space(26)
#     c.setFillColor(COLOR_DEEP_NAVY)
#     c.roundRect(40, cur.y - 18, w - 80, 18, 4, fill=1, stroke=0)
#     c.setFillColor(colors.white)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(50, cur.y - 13, text.upper())
#     cur.y -= 30


# # ============================================================
# # CARD DRAWER (Generic)
# # ============================================================

# def _draw_card_box(cur: PDFCursor, title, height, fill_color=COLOR_CARD_BG):
#     c = cur.c
#     w = cur.width

#     cur.ensure_space(height + 10)
#     y_top = cur.y

#     c.setFillColor(fill_color)
#     c.roundRect(40, y_top - height, w - 80, height, 8, fill=1, stroke=0)

#     c.setFillColor(colors.black)
#     c.setFont("Helvetica-Bold", 9)
#     c.drawString(50, y_top - 14, title)

#     return y_top - height - 12


# # ============================================================
# # RADAR + BARS (SIDE BY SIDE)
# # ============================================================

# def _draw_radar_and_bars(cur: PDFCursor, data):
#     c = cur.c
#     w = cur.width

#     flat = _flatten_factors_23(data)
#     order = _get_factor_list_in_ui_order(data)

#     if not order:
#         return

#     # Radar chart wants not too many labels. 23 is okay if small.
#     labels = [FACTOR_LABELS.get(k, k.upper()) for k in order]
#     values = [flat.get(k, 0) for k in order]

#     cur.ensure_space(190)

#     # Radar on left
#     radar_x = 35
#     radar_y = cur.y - 175

#     fig = plt.figure(figsize=(3, 3), dpi=150)
#     ax = fig.add_subplot(111, polar=True)

#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     ax.fill(angles + angles[:1], values + values[:1], color="#06b6d4", alpha=0.25)
#     ax.plot(angles + angles[:1], values + values[:1], color="#06b6d4", linewidth=1)

#     ax.set_xticks(angles)
#     ax.set_xticklabels(labels, fontsize=4.2)
#     ax.set_yticks([0, 25, 50, 75, 100])
#     ax.set_yticklabels([])

#     chart_io = io.BytesIO()
#     plt.savefig(chart_io, format="png", transparent=True)
#     plt.close(fig)
#     chart_io.seek(0)

#     c.drawImage(ImageReader(chart_io), radar_x, radar_y, width=170, height=170, mask="auto")

#     # Bars on right
#     y_bar = cur.y - 15
#     bar_x = w / 2 + 10
#     bar_w = 150

#     for k in order:
#         v = float(flat.get(k, 0))
#         label = FACTOR_LABELS.get(k, k.upper())[:16]

#         if y_bar < 75:
#             cur.y = y_bar
#             cur.ensure_space(200)
#             y_bar = cur.y - 15

#         c.setFillColor(colors.black)
#         c.setFont("Helvetica", 7)
#         c.drawString(bar_x, y_bar, label)

#         c.setFillColor(COLOR_BORDER)
#         c.roundRect(bar_x + 80, y_bar - 2, bar_w, 6, 2, fill=1, stroke=0)

#         c.setFillColor(_score_color(v))
#         c.roundRect(bar_x + 80, y_bar - 2, (v / 100) * bar_w, 6, 2, fill=1, stroke=0)

#         c.setFillColor(colors.black)
#         c.setFont("Helvetica-Bold", 6.5)
#         c.drawString(bar_x + 80 + bar_w + 8, y_bar, f"{v:.1f}%")

#         y_bar -= 11

#     cur.y = radar_y - 25


# # ============================================================
# # EVIDENCE (CATEGORY-BY-CATEGORY, UI ORDER)
# # ============================================================

# def _draw_evidence(cur: PDFCursor, data):
#     c = cur.c
#     w = cur.width

#     meta = (data.get("explanation", {}) or {}).get("factors_meta", {}) or {}
#     factors = data.get("factors", {}) or {}
#     category_scores = data.get("category_scores") or (data.get("explanation", {}) or {}).get("category_scores") or {}

#     for cat in CATEGORY_ORDER:
#         cat_meta = meta.get(cat) or {}
#         cat_factors = factors.get(cat) or {}
#         if not isinstance(cat_meta, dict) or not isinstance(cat_factors, dict):
#             continue
#         if not cat_meta and not cat_factors:
#             continue

#         # Category header
#         cat_score = _safe_num(category_scores.get(cat), 0)
#         cur.ensure_space(55)

#         c.setFillColor(_score_color(cat_score))
#         c.roundRect(40, cur.y - 22, w - 80, 20, 6, fill=1, stroke=0)

#         c.setFillColor(colors.white)
#         c.setFont("Helvetica-Bold", 9)
#         c.drawString(50, cur.y - 16, f"{CATEGORY_TITLES.get(cat, cat.upper())}  ({cat_score:.1f}/100)")

#         cur.y -= 32

#         # Factors inside category
#         order = FACTOR_ORDER_BY_CATEGORY.get(cat, list(cat_meta.keys()))

#         for f_key in order:
#             f_obj = cat_meta.get(f_key) or cat_factors.get(f_key) or {}
#             if not f_obj:
#                 continue

#             val = 0.0
#             if isinstance(f_obj, dict):
#                 val = _safe_num(f_obj.get("value") or f_obj.get("scaled_score"), 0)

#             evidence = ""
#             if isinstance(f_obj, dict):
#                 evidence = (
#                     f_obj.get("evidence")
#                     or f_obj.get("reason")
#                     or f"Score {val:.1f}/100."
#                 )
#             else:
#                 evidence = str(f_obj)

#             title = f"{FACTOR_LABELS.get(f_key, f_key.upper())}: {val:.1f}/100"
#             needed = 45 + len(evidence) * 0.03
#             cur.ensure_space(needed)

#             # Left color bar
#             c.setFillColor(_score_color(val))
#             c.rect(40, cur.y - 30, 3, 32, fill=1, stroke=0)

#             c.setFillColor(colors.black)
#             c.setFont("Helvetica-Bold", 8.5)
#             c.drawString(50, cur.y - 6, title)

#             # Evidence text
#             max_w = w - 100
#             lines = _wrap_text_lines(c, evidence, max_w, "Helvetica", 8)

#             y_txt = cur.y - 18
#             c.setFont("Helvetica", 8)

#             for ln in lines[:20_000]:  # unlimited effectively
#                 if y_txt < 60:
#                     cur.y = y_txt
#                     cur.new_page()
#                     y_txt = cur.y - 20

#                 c.drawString(50, y_txt, ln)
#                 y_txt -= 10

#             cur.y = y_txt - 12


# # ============================================================
# # LOCATIONAL INTELLIGENCE CARDS
# # ============================================================

# def _draw_weather_card(cur: PDFCursor, weather):
#     if not weather:
#         return

#     y_next = _draw_card_box(cur, "WEATHER (LIVE TELEMETRY)", 62, colors.HexColor("#f0f9ff"))
#     c = cur.c

#     c.setFont("Helvetica", 8)
#     c.drawString(50, cur.y - 26, f"Temperature: {weather.get('temp', 'N/A')}°C  |  Humidity: {weather.get('humidity', 'N/A')}%")
#     c.drawString(50, cur.y - 38, f"Conditions: {weather.get('description', 'N/A')}")
#     c.drawString(50, cur.y - 50, f"Wind: {weather.get('wind_speed_kmh', weather.get('wind_speed', 'N/A'))} km/h  |  Pressure: {weather.get('pressure_hpa', 'N/A')} hPa")
#     c.drawString(50, cur.y - 62, f"Weather code: {weather.get('weather_code', 'N/A')}")

#     cur.y = y_next


# def _draw_cnn_card(cur: PDFCursor, cnn):
#     if not cnn:
#         return

#     y_next = _draw_card_box(cur, "CNN CLASSIFICATION (VISUAL INTELLIGENCE)", 78, colors.HexColor("#fef3c7"))
#     c = cur.c

#     c.setFont("Helvetica", 8)
#     c.drawString(50, cur.y - 26, f"Class: {cnn.get('class', 'N/A')}  |  Confidence: {cnn.get('confidence_display', cnn.get('confidence', 'N/A'))}")

#     note = cnn.get("note") or ""
#     lines = _wrap_text_lines(c, note[:400], cur.width - 120, "Helvetica", 7.5)
#     y_txt = cur.y - 38
#     for ln in lines[:4]:
#         c.drawString(50, y_txt, ln)
#         y_txt -= 9

#     tel = cnn.get("telemetry") or {}
#     c.setFont("Helvetica", 7)
#     c.drawString(50, cur.y - 70, f"RES: {tel.get('resolution_m_per_px', 'N/A')} m/px  |  SENSOR: {tel.get('tile_url_source', 'N/A')}  |  MODEL: {tel.get('model', 'N/A')}")

#     cur.y = y_next


# def _draw_hazards_card(cur: PDFCursor, hazards):
#     if not hazards:
#         return

#     # Dynamic height
#     items = []
#     if isinstance(hazards, dict):
#         items = list(hazards.items())[:10]

#     h = 40 + len(items) * 12
#     y_next = _draw_card_box(cur, "HAZARDS INTELLIGENCE", h, colors.HexColor("#fff7ed"))
#     c = cur.c

#     y = cur.y - 26
#     c.setFont("Helvetica", 8)

#     if not items:
#         c.drawString(50, y, "No hazards data available.")
#     else:
#         for k, v in items:
#             if y < 60:
#                 cur.y = y
#                 cur.new_page()
#                 y = cur.y - 30
#             c.drawString(50, y, f"{str(k).replace('_', ' ').title()}: {str(v)[:120]}")
#             y -= 12

#     cur.y = y_next


# def _draw_snapshot_card(cur: PDFCursor, snap):
#     if not snap:
#         return

#     items = []
#     if isinstance(snap, dict):
#         items = list(snap.items())[:14]

#     h = 45 + len(items) * 11
#     y_next = _draw_card_box(cur, "SNAPSHOT GEO INTELLIGENCE", h, colors.HexColor("#ecfdf5"))
#     c = cur.c

#     y = cur.y - 26
#     c.setFont("Helvetica", 8)

#     for k, v in items:
#         if y < 60:
#             cur.y = y
#             cur.new_page()
#             y = cur.y - 30
#         c.drawString(50, y, f"{str(k).replace('_', ' ').title()}: {str(v)[:120]}")
#         y -= 11

#     cur.y = y_next


# # ============================================================
# # STRATEGIC UTILITY (MATCH UI ORDER)
# # ============================================================

# def _draw_potential_card(cur: PDFCursor, data):
#     flat = _flatten_factors_23(data)
#     score = _safe_num(data.get("suitability_score"), 0)

#     # Basic potential logic
#     potentials = []
#     if any(flat.get(k, 100) < 45 for k in ["flood", "pollution", "drainage"]):
#         hazards = [k.upper() for k in ["flood", "pollution", "drainage"] if flat.get(k, 100) < 45]
#         potentials.append(("ENVIRONMENTAL CONSTRAINTS", f"Critical risk in {', '.join(hazards)}."))

#     if flat.get("flood", 0) > 50 and flat.get("pollution", 0) > 40 and flat.get("slope", 100) < 25:
#         potentials.append(("RESIDENTIAL POTENTIAL", "Stable terrain + flood safety + air quality."))

#     if flat.get("soil", 0) > 60 or flat.get("rainfall", 0) > 60 or flat.get("vegetation", 0) > 50:
#         potentials.append(("AGRICULTURAL UTILITY", "Soil/rainfall/vegetation indicators are favorable."))

#     if flat.get("infrastructure", 0) > 60 and flat.get("landuse", 0) > 40:
#         potentials.append(("LOGISTICS & INDUSTRY", "Infrastructure + landuse feasibility is favorable."))

#     h = 55 + len(potentials) * 16
#     y_next = _draw_card_box(cur, "SITE POTENTIAL ANALYSIS", h, colors.HexColor("#f1f5f9"))
#     c = cur.c

#     c.setFont("Helvetica", 8)
#     c.drawString(50, cur.y - 26, f"Suitability Score: {score:.1f}/100")

#     y = cur.y - 40
#     if not potentials:
#         c.drawString(50, y, "No dominant development potential detected (balanced profile).")
#     else:
#         for label, reason in potentials:
#             if y < 60:
#                 cur.y = y
#                 cur.new_page()
#                 y = cur.y - 30
#             c.setFont("Helvetica-Bold", 7.8)
#             c.drawString(50, y, f"• {label}")
#             c.setFont("Helvetica", 7.8)
#             c.drawString(200, y, reason[:120])
#             y -= 16

#     cur.y = y_next


# def _draw_sustainability_card(cur: PDFCursor, data):
#     f = _flatten_factors_23(data)

#     landuse = f.get("landuse", 50)
#     pollution = f.get("pollution", 50)
#     soil = f.get("soil", 50)
#     water = f.get("water", 50)

#     esg = int(round((soil + (100 - pollution) + water) / 3))

#     y_next = _draw_card_box(cur, "SUSTAINABILITY INTELLIGENCE (ESG)", 75, colors.HexColor("#ecfdf5"))
#     c = cur.c

#     c.setFont("Helvetica", 8)
#     c.drawString(50, cur.y - 26, f"ESG Score (derived): {esg}/100")
#     c.drawString(50, cur.y - 40, f"Landuse: {landuse:.1f}  |  Soil: {soil:.1f}  |  Water: {water:.1f}")
#     c.drawString(50, cur.y - 54, f"Pollution (lower is better): {pollution:.1f}")

#     cur.y = y_next


# def _draw_roadmap_card(cur: PDFCursor, intel):
#     roadmap = (intel or {}).get("roadmap") or []
#     h = 50 + min(len(roadmap), 25) * 14

#     y_next = _draw_card_box(cur, "DYNAMIC IMPROVEMENT ROADMAP", h, colors.HexColor("#f1f5f9"))
#     c = cur.c

#     y = cur.y - 26
#     if not roadmap:
#         c.setFont("Helvetica", 8)
#         c.drawString(50, y, "No roadmap generated.")
#         cur.y = y_next
#         return

#     for item in roadmap[:25]:
#         task = item.get("task") or item.get("title") or "Task"
#         note = item.get("note") or item.get("action") or item.get("reason") or ""
#         cost = item.get("estimated_cost")
#         timeline = item.get("timeline")

#         if y < 60:
#             cur.y = y
#             cur.new_page()
#             y = cur.y - 30

#         c.setFont("Helvetica-Bold", 7.8)
#         c.drawString(50, y, f"• {task[:60]}")
#         y -= 10

#         c.setFont("Helvetica", 7.6)
#         lines = _wrap_text_lines(c, note[:240], cur.width - 120, "Helvetica", 7.6)
#         for ln in lines[:3]:
#             if y < 60:
#                 cur.y = y
#                 cur.new_page()
#                 y = cur.y - 30
#             c.drawString(70, y, ln)
#             y -= 9

#         if cost or timeline:
#             meta = f"{('Cost: ' + str(cost)) if cost else ''}   {('Timeline: ' + str(timeline)) if timeline else ''}"
#             c.setFont("Helvetica-Oblique", 7.2)
#             c.drawString(70, y, meta[:120])
#             y -= 12
#         else:
#             y -= 6

#     cur.y = y_next


# def _draw_interventions_card(cur: PDFCursor, intel):
#     interventions = (intel or {}).get("interventions") or []
#     h = 50 + min(len(interventions), 30) * 14

#     y_next = _draw_card_box(cur, "AI-DRIVEN STRATEGIC INTERVENTIONS", h, colors.HexColor("#fff7ed"))
#     c = cur.c

#     y = cur.y - 26
#     if not interventions:
#         c.setFont("Helvetica", 8)
#         c.drawString(50, y, "No interventions generated.")
#         cur.y = y_next
#         return

#     for item in interventions[:30]:
#         if isinstance(item, str):
#             action = item
#             urgency = None
#             rationale = ""
#         else:
#             action = item.get("action") or item.get("task") or "Intervention"
#             urgency = item.get("urgency")
#             rationale = item.get("rationale") or ""

#         if y < 60:
#             cur.y = y
#             cur.new_page()
#             y = cur.y - 30

#         c.setFont("Helvetica-Bold", 7.8)
#         urg = f" ({urgency})" if urgency else ""
#         c.drawString(50, y, f"• {action[:70]}{urg}")
#         y -= 10

#         c.setFont("Helvetica", 7.6)
#         for ln in _wrap_text_lines(c, rationale[:220], cur.width - 120, "Helvetica", 7.6)[:2]:
#             c.drawString(70, y, ln)
#             y -= 9

#         y -= 6

#     cur.y = y_next


# def _draw_projection_card(cur: PDFCursor, intel, base_score):
#     expected = (intel or {}).get("expected_score")
#     proj = (intel or {}).get("projection_analysis") or {}
#     metrics = (intel or {}).get("metrics") or {}

#     h = 95
#     y_next = _draw_card_box(cur, "ADVANCED AI PROJECTION (2036)", h, colors.HexColor("#eef2ff"))
#     c = cur.c

#     c.setFont("Helvetica", 8)
#     c.drawString(50, cur.y - 26, f"Current score: {base_score:.1f}  →  Projected: {expected if expected is not None else 'N/A'}")

#     trend = proj.get("trend_direction")
#     conf = proj.get("confidence_level")
#     if trend or conf:
#         c.drawString(50, cur.y - 40, f"Trend: {trend or 'N/A'}  |  Confidence: {conf or 'N/A'}")

#     drivers = proj.get("key_drivers") or []
#     if drivers:
#         c.drawString(50, cur.y - 54, f"Key Drivers: {', '.join(drivers)[:120]}")

#     # Metrics
#     y = cur.y - 70
#     for k, v in list(metrics.items())[:5]:
#         c.drawString(50, y, f"{str(k).replace('_',' ').title()}: {str(v)[:60]}")
#         y -= 11

#     cur.y = y_next


# def _draw_terrain_card(cur: PDFCursor, data):
#     # Prefer your terrain_analysis object, fallback to factor slope/elevation
#     terrain = data.get("terrain_analysis") or {}
#     f = data.get("factors", {}) or {}
#     slope_val = _safe_num(((f.get("physical") or {}).get("slope") or {}).get("value"), None)

#     if slope_val is None:
#         slope_val = _safe_num(terrain.get("slope_percent"), 0)

#     elev_val = _safe_num(((f.get("physical") or {}).get("elevation") or {}).get("value"), None)
#     if elev_val is None:
#         elev_val = _safe_num(terrain.get("elevation_m"), 0)

#     h = 95
#     y_next = _draw_card_box(cur, "TERRAIN & SLOPE ANALYSIS", h, colors.HexColor("#f8fafc"))
#     c = cur.c

#     c.setFont("Helvetica", 8)
#     c.drawString(50, cur.y - 26, f"Slope: {slope_val:.1f}°")
#     c.drawString(50, cur.y - 40, f"Elevation: {elev_val:.0f} m")

#     verdict = terrain.get("verdict") or "N/A"
#     c.setFont("Helvetica-Bold", 8)
#     c.drawString(50, cur.y - 56, f"Verdict: {verdict}")

#     cur.y = y_next


# def _draw_digital_twin_summary(cur: PDFCursor):
#     # You cannot embed interactive twin. This is a placeholder summary.
#     h = 55
#     y_next = _draw_card_box(cur, "DIGITAL TWIN INFRASTRUCTURE SIMULATION", h, colors.HexColor("#f1f5f9"))
#     c = cur.c

#     c.setFont("Helvetica", 8)
#     c.drawString(50, cur.y - 26, "Interactive simulation is available in the web app.")
#     c.drawString(50, cur.y - 40, "Recommendation: export a screenshot if you need it embedded in PDF.")

#     cur.y = y_next


# # ============================================================
# # FULL SITE REPORT (ONE SITE)
# # ============================================================

# def _draw_site_report(cur: PDFCursor, data, title_prefix):
#     # Header + minimap
#     _draw_main_header(cur, data, title_prefix)
#     _draw_minimap(cur, data)
#     _draw_scorecard(cur, data)

#     # TAB 1
#     _draw_tab_header(cur, "TAB 1: SUITABILITY")
#     _draw_radar_and_bars(cur, data)

#     _draw_tab_header(cur, "SUITABILITY: EVIDENCE DETAILS")
#     _draw_evidence(cur, data)

#     # TAB 2
#     _draw_tab_header(cur, "TAB 2: LOCATIONAL INTELLIGENCE")
#     _draw_cnn_card(cur, data.get("cnn_analysis"))
#     _draw_weather_card(cur, data.get("weather"))
#     _draw_hazards_card(cur, data.get("hazards_analysis") or (data.get("snapshot_identity") or {}).get("hazards_analysis"))
#     _draw_snapshot_card(cur, data.get("snapshot_identity"))

#     # TAB 3
#     _draw_tab_header(cur, "TAB 3: STRATEGIC UTILITY")
#     _draw_potential_card(cur, data)
#     _draw_sustainability_card(cur, data)

#     intel = data.get("strategic_intelligence") or {}
#     _draw_roadmap_card(cur, intel)
#     _draw_interventions_card(cur, intel)
#     _draw_projection_card(cur, intel, _safe_num(data.get("suitability_score"), 0))
#     _draw_terrain_card(cur, data)
#     _draw_digital_twin_summary(cur)


# # ============================================================
# # PUBLIC FUNCTION
# # ============================================================

# def generate_land_report(data):
#     buffer = io.BytesIO()
#     c = canvas.Canvas(buffer, pagesize=A4)
#     width, height = A4

#     cur = PDFCursor(c, width, height)

#     # SITE A
#     _draw_site_report(cur, data, "LOCATION A")

#     # SITE B (COMPARE)
#     compare_data = data.get("compareData")
#     if compare_data:
#         cur.new_page()
#         _draw_site_report(cur, compare_data, "LOCATION B")

#     c.save()
#     buffer.seek(0)
#     return buffer

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import io
import requests
import qrcode
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader


# ============================================================
# GLOBAL ORDER (MATCHES YOUR FRONTEND)
# ============================================================

CATEGORY_ORDER = [
    "physical",
    "hydrology",
    "environmental",
    "climatic",
    "socio_econ",
    "risk_resilience",
]

CATEGORY_TITLES = {
    "physical": "PHYSICAL",
    "hydrology": "HYDROLOGY",
    "environmental": "ENVIRONMENTAL",
    "climatic": "CLIMATIC",
    "socio_econ": "SOCIO-ECONOMIC",
    "risk_resilience": "RISK & RESILIENCE",
}

FACTOR_ORDER_BY_CATEGORY = {
    "physical": ["elevation", "ruggedness", "slope", "stability"],
    "environmental": ["biodiversity", "heat_island", "pollution", "soil", "vegetation"],
    "hydrology": ["drainage", "flood", "groundwater", "water"],
    "climatic": ["intensity", "rainfall", "thermal"],
    "socio_econ": ["infrastructure", "landuse", "population"],
    "risk_resilience": ["climate_change", "habitability", "multi_hazard", "recovery"],
}

FACTOR_LABELS = {
    "slope": "SLOPE",
    "elevation": "ELEVATION",
    "ruggedness": "RUGGEDNESS",
    "stability": "STABILITY",
    "flood": "FLOOD RISK",
    "water": "WATER PROXIMITY",
    "drainage": "DRAINAGE",
    "groundwater": "GROUNDWATER",
    "vegetation": "VEGETATION",
    "soil": "SOIL QUALITY",
    "pollution": "AIR POLLUTION",
    "biodiversity": "BIODIVERSITY",
    "heat_island": "HEAT ISLAND",
    "rainfall": "RAINFALL",
    "thermal": "THERMAL COMFORT",
    "intensity": "HEAT STRESS",
    "landuse": "LANDUSE",
    "infrastructure": "INFRASTRUCTURE",
    "population": "POPULATION",
    "multi_hazard": "MULTI-HAZARD",
    "climate_change": "CLIMATE CHANGE",
    "recovery": "RECOVERY CAPACITY",
    "habitability": "HABITABILITY",
}


# ============================================================
# COLORS
# ============================================================

COLOR_DEEP_NAVY = colors.HexColor("#0f172a")
COLOR_DANGER = colors.HexColor("#ef4444")
COLOR_WARNING = colors.HexColor("#f59e0b")
COLOR_SUCCESS = colors.HexColor("#10b981")

COLOR_CARD_BG = colors.HexColor("#f8fafc")
COLOR_BORDER = colors.HexColor("#e2e8f0")


# ============================================================
# BASIC HELPERS
# ============================================================

def _score_color(score: float):
    if score < 40:
        return COLOR_DANGER
    if score < 70:
        return COLOR_WARNING
    return COLOR_SUCCESS


def _safe_num(v, default=0.0):
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_str(v, default="N/A"):
    try:
        if v is None:
            return default
        s = str(v).strip()
        return s if s else default
    except Exception:
        return default


def _wrap_text_lines(c, text, max_width, font_name="Helvetica", font_size=8):
    """Returns wrapped lines list for text."""
    if not text:
        return []

    c.setFont(font_name, font_size)
    words = str(text).split()
    lines = []
    line = ""

    for w in words:
        trial = (line + " " + w).strip()
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = w

    if line:
        lines.append(line)

    return lines


def _normalize_meta(meta):
    """
    Your backend sometimes sends:
    factors_meta = {cat: {factor: {...}}}
    OR factors_meta = {factor: {...}}

    This normalizes to:
    {cat: {factor: {...}}}
    """
    if not isinstance(meta, dict):
        return {}

    # If keys look like categories
    if any(k in CATEGORY_ORDER for k in meta.keys()):
        return meta

    # Else treat as flat factors_meta
    out = {cat: {} for cat in CATEGORY_ORDER}
    for k, v in meta.items():
        # try to find factor in any category
        placed = False
        for cat, order in FACTOR_ORDER_BY_CATEGORY.items():
            if k in order:
                out[cat][k] = v
                placed = True
                break
        if not placed:
            # put unknown factors into environmental
            out.setdefault("environmental", {})[k] = v
    return out


# ============================================================
# PAGINATION ENGINE (NO LIMIT PAGES)
# ============================================================

class PDFCursor:
    def __init__(self, c, width, height):
        self.c = c
        self.width = width
        self.height = height
        self.x_margin = 40
        self.y = height - 20
        self.page = 1

    def new_page(self):
        self.c.showPage()
        self.page += 1
        self.y = self.height - 20

    def ensure_space(self, needed_height, min_y=55):
        if self.y - needed_height < min_y:
            self.new_page()


# ============================================================
# FACTOR FLATTENING (23 factors + weights + evidence status)
# ============================================================

def _flatten_factors_23(data):
    """
    Returns:
    flat_score[factor] = scaled_score or value (0-100)
    flat_weight[factor] = weight if present
    flat_raw[factor] = raw dict
    """
    raw = data.get("factors", {}) or {}
    flat_score = {}
    flat_weight = {}
    flat_raw = {}

    if isinstance(raw, dict):
        for cat, cat_data in raw.items():
            if not isinstance(cat_data, dict):
                continue

            for fkey, fval in cat_data.items():
                flat_raw[fkey] = fval

                if isinstance(fval, dict):
                    s = fval.get("scaled_score")
                    v = fval.get("value", 0)
                    w = fval.get("weight") or fval.get("w") or fval.get("wt")
                    score = _safe_num(s if s is not None else v, 0)
                    flat_score[fkey] = max(0, min(100, score))

                    if w is not None:
                        flat_weight[fkey] = _safe_num(w, None)
                else:
                    score = _safe_num(fval, 0)
                    flat_score[fkey] = max(0, min(100, score))

    return flat_score, flat_weight, flat_raw


def _get_factor_list_in_ui_order(data):
    raw = data.get("factors", {}) or {}
    ordered = []

    for cat in CATEGORY_ORDER:
        cat_data = raw.get(cat) or {}
        if not isinstance(cat_data, dict):
            continue
        keys = FACTOR_ORDER_BY_CATEGORY.get(cat, list(cat_data.keys()))
        for k in keys:
            if k in cat_data:
                ordered.append(k)

    # Add any extras at end (rare)
    for cat, cat_data in raw.items():
        if not isinstance(cat_data, dict):
            continue
        for k in cat_data.keys():
            if k not in ordered:
                ordered.append(k)

    return ordered


# ============================================================
# HEADER + MINIMAP
# ============================================================

def _draw_main_header(cur: PDFCursor, data, title_prefix):
    c = cur.c
    w, h = cur.width, cur.height

    c.setFillColor(COLOR_DEEP_NAVY)
    c.rect(0, h - 100, w, 100, fill=1, stroke=0)

    # QR + Button
    share_url = data.get("shareLink")
    if share_url:
        try:
            qr = qrcode.QRCode(version=1, box_size=10, border=2)
            qr.add_data(share_url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")

            qr_buffer = io.BytesIO()
            qr_img.save(qr_buffer, format="PNG")
            qr_buffer.seek(0)

            c.setFillColor(colors.white)
            c.roundRect(w - 95, h - 90, 80, 80, 4, fill=1, stroke=0)
            c.drawImage(ImageReader(qr_buffer), w - 90, h - 82, width=70, height=70)

            btn_x, btn_y, btn_w, btn_h = w - 90, h - 88, 70, 10
            c.setFillColor(COLOR_SUCCESS)
            c.roundRect(btn_x, btn_y, btn_w, btn_h, 2, fill=1, stroke=0)

            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            c.drawCentredString(w - 55, btn_y + 3, "OPEN LINK")
            c.linkURL(share_url, (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h), relative=0)

        except Exception:
            pass

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(w / 2, h - 40, "GeoAI – Land Suitability Report")

    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(w / 2, h - 60, f"{title_prefix}: {data.get('locationName', 'Site')}".upper())

    loc = data.get("location", {}) or {}
    lat = loc.get("latitude") or loc.get("lat") or 0.0
    lon = loc.get("longitude") or loc.get("lng") or 0.0

    timestamp = datetime.now().strftime("%d %b %Y | %H:%M:%S IST")
    c.setFont("Helvetica", 8)
    c.drawCentredString(w / 2, h - 80, f"{timestamp}  •  LAT: {lat} | LNG: {lon}")

    cur.y = h - 110


def _draw_minimap(cur: PDFCursor, data):
    c = cur.c
    w = cur.width

    loc = data.get("location", {}) or {}
    lat = loc.get("latitude") or loc.get("lat") or 0.0
    lon = loc.get("longitude") or loc.get("lng") or 0.0

    cur.ensure_space(130)
    y_map = cur.y - 120

    try:
        map_url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&z=13&l=sat&size=500,140"
        map_res = requests.get(map_url, timeout=6)
        if map_res.status_code == 200:
            map_img = ImageReader(io.BytesIO(map_res.content))
            c.drawImage(map_img, 40, y_map, width=w - 80, height=110)
            c.setStrokeColor(colors.white)
            c.rect(40, y_map, w - 80, 110, stroke=1, fill=0)
        else:
            c.setFillColor(colors.lightgrey)
            c.rect(40, y_map, w - 80, 110, fill=1, stroke=0)
    except Exception:
        c.setFillColor(colors.lightgrey)
        c.rect(40, y_map, w - 80, 110, fill=1, stroke=0)

    cur.y = y_map - 20


def _draw_scorecard(cur: PDFCursor, data):
    c = cur.c

    score = _safe_num(data.get("suitability_score"), 0)
    grade = str(data.get("label", "N/A")).upper()
    col = _score_color(score)

    cur.ensure_space(40)
    c.setFillColor(col)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(45, cur.y - 10, f"{score:.1f}")

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(45, cur.y - 25, f"GRADE: {grade}")

    cur.y -= 45


# ============================================================
# TAB HEADER
# ============================================================

def _draw_tab_header(cur: PDFCursor, text):
    c = cur.c
    w = cur.width

    cur.ensure_space(26)
    c.setFillColor(COLOR_DEEP_NAVY)
    c.roundRect(40, cur.y - 18, w - 80, 18, 4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, cur.y - 13, text.upper())
    cur.y -= 30


# ============================================================
# GENERIC CARD
# ============================================================

def _draw_card_box(cur: PDFCursor, title, height, fill_color=COLOR_CARD_BG):
    c = cur.c
    w = cur.width

    cur.ensure_space(height + 12)
    y_top = cur.y

    c.setFillColor(fill_color)
    c.roundRect(40, y_top - height, w - 80, height, 8, fill=1, stroke=0)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y_top - 14, title)

    return y_top - height - 12


# ============================================================
# RADAR + BARS (NO OVERLAP, PAGINATES)
# ============================================================

def _draw_radar_and_bars(cur: PDFCursor, data):
    c = cur.c
    w = cur.width

    flat_score, flat_weight, _ = _flatten_factors_23(data)
    order = _get_factor_list_in_ui_order(data)

    if not order:
        return

    # Radar (limit labels for readability)
    radar_order = order[:23]  # allow all 23 but tiny labels

    labels = [FACTOR_LABELS.get(k, k.upper()) for k in radar_order]
    values = [flat_score.get(k, 0) for k in radar_order]

    cur.ensure_space(200)

    # Radar left
    radar_x = 35
    radar_y = cur.y - 175

    fig = plt.figure(figsize=(3, 3), dpi=150)
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ax.fill(angles + angles[:1], values + values[:1], color="#06b6d4", alpha=0.25)
    ax.plot(angles + angles[:1], values + values[:1], color="#06b6d4", linewidth=1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=4.1)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([])

    chart_io = io.BytesIO()
    plt.savefig(chart_io, format="png", transparent=True)
    plt.close(fig)
    chart_io.seek(0)

    c.drawImage(ImageReader(chart_io), radar_x, radar_y, width=170, height=170, mask="auto")

    # Bars right
    y_bar = cur.y - 15
    bar_x = w / 2 + 10
    bar_w = 150

    # NOTE: bars must not collide with radar bottom on same page
    radar_bottom_safe_y = radar_y + 10

    for k in order:
        v = float(flat_score.get(k, 0))
        wt = flat_weight.get(k, None)
        label = FACTOR_LABELS.get(k, k.upper())[:18]

        # If bars go below radar, push to next page
        if y_bar < radar_bottom_safe_y:
            cur.new_page()
            _draw_tab_header(cur, "TAB 1: SUITABILITY (CONTINUED)")
            y_bar = cur.y - 10

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7)
        c.drawString(bar_x, y_bar, label)

        c.setFillColor(COLOR_BORDER)
        c.roundRect(bar_x + 80, y_bar - 2, bar_w, 6, 2, fill=1, stroke=0)

        c.setFillColor(_score_color(v))
        c.roundRect(bar_x + 80, y_bar - 2, (v / 100) * bar_w, 6, 2, fill=1, stroke=0)

        # right score
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 6.5)
        c.drawString(bar_x + 80 + bar_w + 8, y_bar, f"{v:.1f}%")

        # weight
        if wt is not None:
            c.setFont("Helvetica", 6.3)
            c.setFillColor(colors.HexColor("#334155"))
            c.drawRightString(bar_x + 75, y_bar, f"W: {wt:.1f}")

        y_bar -= 11

    # move cursor below radar
    cur.y = radar_y - 25


# ============================================================
# EVIDENCE (MATCH UI: score + weighted + status + evidence text)
# ============================================================

def _extract_factor_meta(meta_obj):
    """
    UI style expects:
    - evidence / reason
    - status (good/moderate/poor etc)
    - weighted_score
    """
    if not isinstance(meta_obj, dict):
        return {"evidence": str(meta_obj), "status": None, "weighted_score": None}

    evidence = meta_obj.get("evidence") or meta_obj.get("reason") or meta_obj.get("explanation") or ""
    status = meta_obj.get("status") or meta_obj.get("badge") or meta_obj.get("level")
    weighted = meta_obj.get("weighted_score") or meta_obj.get("weighted") or meta_obj.get("weightedScore")

    return {
        "evidence": evidence,
        "status": status,
        "weighted_score": weighted,
    }


def _draw_evidence(cur: PDFCursor, data):
    c = cur.c
    w = cur.width

    meta_raw = (data.get("explanation", {}) or {}).get("factors_meta", {}) or {}
    meta = _normalize_meta(meta_raw)

    factors = data.get("factors", {}) or {}
    category_scores = (
        data.get("category_scores")
        or (data.get("explanation", {}) or {}).get("category_scores")
        or {}
    )

    for cat in CATEGORY_ORDER:
        cat_meta = meta.get(cat) or {}
        cat_factors = factors.get(cat) or {}

        if not isinstance(cat_meta, dict):
            cat_meta = {}
        if not isinstance(cat_factors, dict):
            cat_factors = {}

        if not cat_meta and not cat_factors:
            continue

        cat_score = _safe_num(category_scores.get(cat), 0)

        cur.ensure_space(55)

        c.setFillColor(_score_color(cat_score))
        c.roundRect(40, cur.y - 22, w - 80, 20, 6, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(50, cur.y - 16, f"{CATEGORY_TITLES.get(cat, cat.upper())}  ({cat_score:.1f}/100)")

        cur.y -= 32

        order = FACTOR_ORDER_BY_CATEGORY.get(cat, list(cat_factors.keys()))

        for f_key in order:
            f_obj = cat_factors.get(f_key) or {}
            m_obj = cat_meta.get(f_key) or {}

            if not f_obj and not m_obj:
                continue

            # factor score
            if isinstance(f_obj, dict):
                val = _safe_num(f_obj.get("scaled_score") if f_obj.get("scaled_score") is not None else f_obj.get("value"), 0)
                wt = f_obj.get("weight")
            else:
                val = _safe_num(f_obj, 0)
                wt = None

            meta_info = _extract_factor_meta(m_obj)
            evidence = meta_info.get("evidence") or f"Score {val:.1f}/100."
            status = meta_info.get("status")
            weighted_score = meta_info.get("weighted_score")

            title = f"{FACTOR_LABELS.get(f_key, f_key.upper())}: {val:.1f}/100"

            # dynamic height based on evidence length
            max_w = w - 120
            lines = _wrap_text_lines(c, evidence, max_w, "Helvetica", 8)
            needed = 38 + len(lines) * 10
            cur.ensure_space(needed)

            # left color bar
            c.setFillColor(_score_color(val))
            c.rect(40, cur.y - 30, 3, 32, fill=1, stroke=0)

            # title line
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(50, cur.y - 6, title)

            # badges on right: STATUS + WEIGHTED SCORE
            x_right = w - 50
            y_badge = cur.y - 6

            if weighted_score is None and wt is not None:
                # if UI weighted_score missing, compute approx
                weighted_score = _safe_num(wt, 0) * (_safe_num(val, 0) / 100)

            if weighted_score is not None:
                c.setFont("Helvetica-Bold", 6.5)
                c.setFillColor(colors.HexColor("#1e293b"))
                c.drawRightString(x_right, y_badge, f"WEIGHTED: {_safe_num(weighted_score, 0):.2f}")

            if status:
                c.setFont("Helvetica-Bold", 6.5)
                c.setFillColor(colors.HexColor("#475569"))
                c.drawRightString(x_right, y_badge - 10, f"STATUS: {str(status).upper()[:20]}")

            # evidence text
            y_txt = cur.y - 18
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)

            for ln in lines:
                if y_txt < 60:
                    cur.y = y_txt
                    cur.new_page()
                    y_txt = cur.y - 20
                c.drawString(50, y_txt, ln)
                y_txt -= 10

            cur.y = y_txt - 12


# ============================================================
# LOCATIONAL INTELLIGENCE CARDS
# ============================================================

def _draw_weather_card(cur: PDFCursor, weather):
    if not isinstance(weather, dict) or not weather:
        return

    # expanded keys
    temp = weather.get("temp") or weather.get("temperature_c")
    feels = weather.get("feels_like") or weather.get("feels_like_c")
    humidity = weather.get("humidity")
    wind = weather.get("wind_speed_kmh") or weather.get("wind_speed")
    wind_dir = weather.get("wind_direction")
    pressure = weather.get("pressure_hpa")
    clouds = weather.get("cloud_cover") or weather.get("clouds")
    visibility = weather.get("visibility_km") or weather.get("visibility")
    uv = weather.get("uv_index") or weather.get("uv")
    sunrise = weather.get("sunrise")
    sunset = weather.get("sunset")
    code = weather.get("weather_code")
    desc = weather.get("description") or weather.get("condition")

    rows = []
    rows.append(f"Temperature: {_safe_str(temp)}°C   |   Feels like: {_safe_str(feels)}°C")
    rows.append(f"Humidity: {_safe_str(humidity)}%   |   Pressure: {_safe_str(pressure)} hPa")
    rows.append(f"Wind: {_safe_str(wind)} km/h   |   Direction: {_safe_str(wind_dir)}")
    rows.append(f"Cloud Cover: {_safe_str(clouds)}   |   Visibility: {_safe_str(visibility)}")
    rows.append(f"UV Index: {_safe_str(uv)}")
    rows.append(f"Conditions: {_safe_str(desc)}   |   Weather Code: {_safe_str(code)}")
    if sunrise or sunset:
        rows.append(f"Sunrise: {_safe_str(sunrise)}   |   Sunset: {_safe_str(sunset)}")

    h = 40 + len(rows) * 12
    y_next = _draw_card_box(cur, "WEATHER (LIVE TELEMETRY)", h, colors.HexColor("#f0f9ff"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 8)
    for r in rows:
        c.drawString(50, y, r[:140])
        y -= 12

    cur.y = y_next


def _draw_geospatial_passport_card(cur: PDFCursor, passport):
    if not isinstance(passport, dict) or not passport:
        return

    rows = [
        f"Slope: {_safe_str(passport.get('slope_percent'))}%   |   Suitability: {_safe_str(passport.get('slope_suitability'))}",
        f"Elevation: {_safe_str(passport.get('elevation_m'))} m   |   Landuse: {_safe_str(passport.get('landuse_class'))}",
        f"Vegetation Score: {_safe_str(passport.get('vegetation_score'))}   |   Water Distance: {_safe_str(passport.get('water_distance_km'))} km",
        f"Flood Safety: {_safe_str(passport.get('flood_safety_score'))}   |   Rainfall: {_safe_str(passport.get('rainfall_mm'))} mm",
        f"Risk Summary: {_safe_str(passport.get('risk_summary'))}",
    ]

    h = 40 + len(rows) * 12
    y_next = _draw_card_box(cur, "GEOSPATIAL PASSPORT", h, colors.HexColor("#ecfdf5"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 8)
    for r in rows:
        c.drawString(50, y, r[:140])
        y -= 12

    cur.y = y_next


def _draw_cnn_card(cur: PDFCursor, cnn):
    if not isinstance(cnn, dict) or not cnn:
        return

    note = cnn.get("note") or ""
    tel = cnn.get("telemetry") or {}

    rows = [
        f"Class: {_safe_str(cnn.get('class'))}   |   Confidence: {_safe_str(cnn.get('confidence_display') or cnn.get('confidence'))}",
        f"Model: {_safe_str(tel.get('model'))}   |   Sensor: {_safe_str(tel.get('tile_url_source'))}",
        f"Resolution: {_safe_str(tel.get('resolution_m_per_px'))} m/px   |   Verified: {_safe_str(tel.get('verified_by'))}",
    ]

    # note lines
    note_lines = _wrap_text_lines(cur.c, note[:700], cur.width - 120, "Helvetica", 7.6)
    rows.extend(note_lines[:6])

    h = 40 + len(rows) * 10
    y_next = _draw_card_box(cur, "CNN CLASSIFICATION (VISUAL INTELLIGENCE)", h, colors.HexColor("#fef3c7"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 7.8)
    for r in rows:
        c.drawString(50, y, str(r)[:150])
        y -= 10

    cur.y = y_next


def _draw_hazards_card(cur: PDFCursor, hazards):
    if not hazards:
        return

    items = []
    if isinstance(hazards, dict):
        items = list(hazards.items())
    elif isinstance(hazards, list):
        items = [(f"Hazard {i+1}", v) for i, v in enumerate(hazards[:20])]
    else:
        items = [("Hazards", str(hazards))]

    h = 40 + min(len(items), 18) * 12
    y_next = _draw_card_box(cur, "HAZARDS INTELLIGENCE", h, colors.HexColor("#fff7ed"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 8)

    for k, v in items[:18]:
        if y < 60:
            cur.y = y
            cur.new_page()
            y = cur.y - 30
        c.drawString(50, y, f"{str(k).replace('_', ' ').title()}: {str(v)[:120]}")
        y -= 12

    cur.y = y_next


def _draw_snapshot_card(cur: PDFCursor, snap):
    if not isinstance(snap, dict) or not snap:
        return

    # remove huge nested objects
    items = []
    for k, v in snap.items():
        if isinstance(v, (dict, list)):
            continue
        items.append((k, v))

    h = 45 + min(len(items), 16) * 11
    y_next = _draw_card_box(cur, "SNAPSHOT IDENTITY", h, colors.HexColor("#ecfdf5"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 8)

    for k, v in items[:16]:
        if y < 60:
            cur.y = y
            cur.new_page()
            y = cur.y - 30
        c.drawString(50, y, f"{str(k).replace('_', ' ').title()}: {str(v)[:120]}")
        y -= 11

    cur.y = y_next


def _draw_nearby_places_card(cur: PDFCursor, nearby):
    if not isinstance(nearby, dict) or not nearby:
        return

    places = nearby.get("places") or []
    if not isinstance(places, list) or not places:
        return

    h = 50 + min(len(places), 12) * 12
    y_next = _draw_card_box(cur, "NEARBY AMENITIES (1.5 KM)", h, colors.HexColor("#f1f5f9"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 8)

    for p in places[:12]:
        name = p.get("name") if isinstance(p, dict) else str(p)
        dist = p.get("distance_km") if isinstance(p, dict) else ""
        typ = p.get("type") if isinstance(p, dict) else ""
        c.drawString(50, y, f"• {str(name)[:40]}  |  {typ}  |  {dist} km")
        y -= 12

    cur.y = y_next


# ============================================================
# STRATEGIC UTILITY (MATCH UI STYLE)
# ============================================================

def _draw_potential_card(cur: PDFCursor, data):
    flat_score, _, _ = _flatten_factors_23(data)
    score = _safe_num(data.get("suitability_score"), 0)

    potentials = []
    if any(flat_score.get(k, 100) < 45 for k in ["flood", "pollution", "drainage"]):
        hazards = [k.upper() for k in ["flood", "pollution", "drainage"] if flat_score.get(k, 100) < 45]
        potentials.append(("ENVIRONMENTAL CONSTRAINTS", f"Critical risk in {', '.join(hazards)}."))

    if flat_score.get("flood", 0) > 50 and flat_score.get("pollution", 0) > 40 and flat_score.get("slope", 100) < 25:
        potentials.append(("RESIDENTIAL POTENTIAL", "Stable terrain + flood safety + air quality."))

    if flat_score.get("soil", 0) > 60 or flat_score.get("rainfall", 0) > 60 or flat_score.get("vegetation", 0) > 50:
        potentials.append(("AGRICULTURAL UTILITY", "Soil/rainfall/vegetation indicators are favorable."))

    if flat_score.get("infrastructure", 0) > 60 and flat_score.get("landuse", 0) > 40:
        potentials.append(("LOGISTICS & INDUSTRY", "Infrastructure + landuse feasibility is favorable."))

    h = 55 + max(1, len(potentials)) * 16
    y_next = _draw_card_box(cur, "SITE POTENTIAL ANALYSIS", h, colors.HexColor("#f1f5f9"))
    c = cur.c

    c.setFont("Helvetica", 8)
    c.drawString(50, cur.y - 26, f"Suitability Score: {score:.1f}/100")

    y = cur.y - 40
    if not potentials:
        c.drawString(50, y, "No dominant development potential detected (balanced profile).")
    else:
        for label, reason in potentials:
            c.setFont("Helvetica-Bold", 7.8)
            c.drawString(50, y, f"• {label}")
            c.setFont("Helvetica", 7.8)
            c.drawString(200, y, reason[:120])
            y -= 16

    cur.y = y_next


def _draw_sustainability_card(cur: PDFCursor, data):
    flat_score, _, _ = _flatten_factors_23(data)

    pollution = flat_score.get("pollution", 50)
    soil = flat_score.get("soil", 50)
    water = flat_score.get("water", 50)

    esg = int(round((soil + (100 - pollution) + water) / 3))

    y_next = _draw_card_box(cur, "SUSTAINABILITY INTELLIGENCE (ESG)", 75, colors.HexColor("#ecfdf5"))
    c = cur.c

    c.setFont("Helvetica", 8)
    c.drawString(50, cur.y - 26, f"ESG Score (derived): {esg}/100")
    c.drawString(50, cur.y - 40, f"Soil: {soil:.1f}  |  Water: {water:.1f}")
    c.drawString(50, cur.y - 54, f"Pollution (lower is better): {pollution:.1f}")

    cur.y = y_next


def _draw_roadmap_card(cur: PDFCursor, intel):
    roadmap = (intel or {}).get("roadmap") or []
    if not roadmap:
        return

    h = 50 + min(len(roadmap), 18) * 16
    y_next = _draw_card_box(cur, "DYNAMIC IMPROVEMENT ROADMAP", h, colors.HexColor("#f1f5f9"))
    c = cur.c

    y = cur.y - 26

    for item in roadmap[:18]:
        task = item.get("task") or item.get("title") or "Task"
        action = item.get("action") or item.get("reason") or item.get("note") or ""
        timeline = item.get("timeline")
        cost = item.get("estimated_cost")

        c.setFont("Helvetica-Bold", 7.8)
        c.drawString(50, y, f"• {task[:70]}")
        y -= 10

        c.setFont("Helvetica", 7.6)
        for ln in _wrap_text_lines(c, action[:260], cur.width - 120, "Helvetica", 7.6)[:2]:
            c.drawString(70, y, ln)
            y -= 9

        if timeline or cost:
            meta = f"{('Timeline: ' + str(timeline)) if timeline else ''}   {('Cost: ' + str(cost)) if cost else ''}"
            c.setFont("Helvetica-Oblique", 7.2)
            c.drawString(70, y, meta[:120])
            y -= 12
        else:
            y -= 6

    cur.y = y_next


def _draw_interventions_card(cur: PDFCursor, intel):
    interventions = (intel or {}).get("interventions") or []
    if not interventions:
        return

    h = 55 + min(len(interventions), 16) * 18
    y_next = _draw_card_box(cur, "AI-DRIVEN STRATEGIC INTERVENTIONS", h, colors.HexColor("#fff7ed"))
    c = cur.c

    y = cur.y - 26

    for item in interventions[:16]:
        if isinstance(item, str):
            action = item
            urgency = None
            rationale = ""
            expected_impact = ""
        else:
            action = item.get("action") or item.get("task") or "Intervention"
            urgency = item.get("urgency")
            rationale = item.get("rationale") or ""
            expected_impact = item.get("expected_impact") or ""

        urg = f" [{str(urgency).upper()}]" if urgency else ""

        c.setFont("Helvetica-Bold", 7.8)
        c.drawString(50, y, f"• {action[:70]}{urg}")
        y -= 10

        if rationale:
            c.setFont("Helvetica", 7.6)
            for ln in _wrap_text_lines(c, rationale[:240], cur.width - 120, "Helvetica", 7.6)[:2]:
                c.drawString(70, y, ln)
                y -= 9

        if expected_impact:
            c.setFont("Helvetica-Oblique", 7.3)
            c.drawString(70, y, f"Expected Impact: {expected_impact[:90]}")
            y -= 11

        y -= 4

    cur.y = y_next


def _draw_projection_card(cur: PDFCursor, intel, base_score):
    if not intel:
        return

    expected = intel.get("expected_score")
    proj = intel.get("projection_analysis") or {}
    metrics = intel.get("metrics") or {}

    drivers = proj.get("key_drivers") or []
    trend = proj.get("trend_direction")
    conf = proj.get("confidence_level")
    mitigation = proj.get("mitigation_potential")

    rows = [
        f"Current Score: {base_score:.1f}   →   Projected: {expected if expected is not None else 'N/A'}",
        f"Trend: {_safe_str(trend)}   |   Confidence: {_safe_str(conf)}",
    ]

    if drivers:
        rows.append(f"Key Drivers: {', '.join(drivers)[:120]}")

    if mitigation is not None:
        rows.append(f"Mitigation Potential: +{_safe_num(mitigation, 0):.1f}%")

    # metrics rows
    for k, v in list(metrics.items())[:6]:
        rows.append(f"{str(k).replace('_',' ').title()}: {str(v)[:70]}")

    h = 40 + len(rows) * 12
    y_next = _draw_card_box(cur, "ADVANCED AI PROJECTION (2036)", h, colors.HexColor("#eef2ff"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 8)
    for r in rows:
        c.drawString(50, y, r[:150])
        y -= 12

    cur.y = y_next


def _draw_terrain_card(cur: PDFCursor, data):
    terrain = data.get("terrain_analysis") or {}
    f = data.get("factors", {}) or {}

    slope_val = _safe_num(((f.get("physical") or {}).get("slope") or {}).get("value"), None)
    if slope_val is None:
        slope_val = _safe_num(terrain.get("slope_percent"), 0)

    elev_val = _safe_num(((f.get("physical") or {}).get("elevation") or {}).get("value"), None)
    if elev_val is None:
        elev_val = _safe_num(terrain.get("elevation_m"), 0)

    verdict = terrain.get("verdict") or "N/A"

    rows = [
        f"Slope: {slope_val:.1f}°",
        f"Elevation: {elev_val:.0f} m",
        f"Verdict: {verdict}",
    ]

    h = 40 + len(rows) * 12
    y_next = _draw_card_box(cur, "TERRAIN & SLOPE ANALYSIS", h, colors.HexColor("#f8fafc"))
    c = cur.c

    y = cur.y - 26
    c.setFont("Helvetica", 8)
    for r in rows:
        c.drawString(50, y, r[:140])
        y -= 12

    cur.y = y_next


def _draw_digital_twin_summary(cur: PDFCursor):
    h = 55
    y_next = _draw_card_box(cur, "DIGITAL TWIN INFRASTRUCTURE SIMULATION", h, colors.HexColor("#f1f5f9"))
    c = cur.c

    c.setFont("Helvetica", 8)
    c.drawString(50, cur.y - 26, "Interactive simulation is available in the web app.")
    c.drawString(50, cur.y - 40, "Recommendation: export a screenshot if you need it embedded in PDF.")

    cur.y = y_next


# ============================================================
# FULL SITE REPORT (ONE SITE)
# ============================================================

def _draw_site_report(cur: PDFCursor, data, title_prefix):
    # Header + minimap
    _draw_main_header(cur, data, title_prefix)
    _draw_minimap(cur, data)
    _draw_scorecard(cur, data)

    # TAB 1
    _draw_tab_header(cur, "TAB 1: SUITABILITY")
    _draw_radar_and_bars(cur, data)

    _draw_tab_header(cur, "SUITABILITY: EVIDENCE DETAILS")
    _draw_evidence(cur, data)

    # TAB 2
    _draw_tab_header(cur, "TAB 2: LOCATIONAL INTELLIGENCE")
    _draw_geospatial_passport_card(cur, data.get("geospatial_passport"))
    _draw_cnn_card(cur, data.get("cnn_analysis"))
    _draw_weather_card(cur, data.get("weather"))

    # hazards + snapshot (safe)
    hazards = data.get("hazards_analysis")
    if not hazards and isinstance(data.get("snapshot_identity"), dict):
        hazards = (data.get("snapshot_identity") or {}).get("hazards_analysis")
    _draw_hazards_card(cur, hazards)

    _draw_snapshot_card(cur, data.get("snapshot_identity"))
    _draw_nearby_places_card(cur, data.get("nearby_places"))

    # TAB 3
    _draw_tab_header(cur, "TAB 3: STRATEGIC UTILITY")
    _draw_potential_card(cur, data)
    _draw_sustainability_card(cur, data)

    intel = data.get("strategic_intelligence") or {}
    _draw_roadmap_card(cur, intel)
    _draw_interventions_card(cur, intel)
    _draw_projection_card(cur, intel, _safe_num(data.get("suitability_score"), 0))
    _draw_terrain_card(cur, data)
    _draw_digital_twin_summary(cur)


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def generate_land_report(data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    cur = PDFCursor(c, width, height)

    # SITE A
    _draw_site_report(cur, data, "LOCATION A")

    # SITE B (COMPARE)
    compare_data = data.get("compareData")
    if compare_data:
        cur.new_page()
        _draw_site_report(cur, compare_data, "LOCATION B")

    c.save()
    buffer.seek(0)
    return buffer
