# This file is part of the Warehouse Management App.
#
# Copyright (C) 2025 MAROUANE DAOUKI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import streamlit as st

# --- Page Config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Carton Compliance Analyzer", layout="wide", initial_sidebar_state="expanded", page_icon="📦")

from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import time
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.metric_cards import style_metric_cards
# from streamlit_extras.stoggle import stoggle
# from streamlit_extras.stylable_container import stylable_container

# --- Configuration ---
MODEL_DIR = Path(__file__).resolve().parent / "weights"
MODEL_PATH = MODEL_DIR / "best.pt"
MAX_ZONES = 5
DATA_PATH = Path(__file__).resolve().parent / "data"
HISTORY_FILE = DATA_PATH / "analysis_history.json"
SETTINGS_FILE = DATA_PATH / "user_settings.json"

DATA_PATH.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Theme Colors ---
THEME = {
    "primary": "#0066cc", "secondary": "#6c757d", "success": "#10b981", "warning": "#ff9800",
    "danger": "#dc3545", "info": "#0dcaf0", "dark": "#212529", "light": "#f8f9fa",
    "accent": "#7b2cbf", "neutral": "#64748b",
}

# --- Custom CSS ---
# --- Custom CSS ---
# --- Custom CSS ---
def load_custom_css():
    st.markdown(f"""
    <style>
        :root {{
            --primary: {THEME["primary"]}; 
            --secondary: {THEME["secondary"]}; 
            --success: {THEME["success"]};
            --warning: {THEME["warning"]}; 
            --danger: {THEME["danger"]}; 
            --info: {THEME["info"]};
            --dark: {THEME["dark"]}; 
            --light: {THEME["light"]}; 
            --accent: {THEME["accent"]}; 
            --neutral: {THEME["neutral"]}; /* A good dark gray for text on light backgrounds */
            --page-bg: #1E293B; /* Dark blue-gray for overall page background */
            --content-bg: #FFFFFF; /* White for the main content block where dashboard elements sit */
            --card-bg: #FFFFFF; /* White for metric cards and styled cards */
            --text-dark: #212529; /* Standard dark text */
            --text-neutral: #4A5568; /* Slightly lighter dark text, good for labels */
            --text-light-on-dark: #E2E8F0; /* Light text for dark backgrounds (like header description) */
        }}

        /* Overall page background (outside the main content blocks) */
        body {{
            background-color: var(--page-bg) !important; 
            color: var(--text-dark); /* Default text color for elements not otherwise styled */
        }}

        /* Main content area where dashboard and other pages render */
        .main .block-container {{
            background-color: var(--content-bg) !important; /* Explicitly light */
            padding: 1.5rem;
            border-radius: 10px;
            color: var(--text-dark); /* Default text color for content inside this block */
        }}
        
        /* Colored Header from streamlit-extras styling */
        /* This targets the div streamlit-extras creates. We want its text to be light if its own background is dark. */
        div[data-testid="stVerticalBlock"] > div[style*="background-color: rgb(30, 41, 59)"], /* Example for a dark header bg */
        div[data-testid="stVerticalBlock"] > div[style*="background-color: {THEME["dark"]}"] {{
            color: var(--text-light-on-dark) !important;
        }}
        /* Make description text in colored_header lighter if the header itself is dark */
        h1 + p, h2 + p, h3 + p {{ /* Assuming description is a <p> after <h1> etc. from colored_header */
             /* This is tricky because colored_header injects its own styles.
                The component itself might need an option for description color or we inspect its output.
                For now, let's assume the description under colored_header needs to be visible on the --page-bg */
            color: var(--text-light-on-dark) !important; /* If header description is on the dark page bg */
        }}


        /* General styled cards (like summary on analysis page) */
        .styled-card {{ 
            background-color: var(--card-bg); 
            border-radius: 10px; 
            padding: 1.5rem; 
            margin-bottom: 1rem; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
            transition: transform 0.3s, box-shadow 0.3s; 
            color: var(--text-dark) !important; 
        }}
        .styled-card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 15px rgba(0,0,0,0.1); }}
        
        /* Metric card enhancements (Dashboard Cards) */
        div[data-testid="stMetric"] {{
            background-color: var(--card-bg) !important; /* White background */
            border-radius: 10px; 
            padding: 1.5rem; /* Increased padding for better look */
            box-shadow: 0 4px 8px rgba(0,0,0,0.07); /* Slightly more pronounced shadow */
            text-align: center;
            border: 1px solid #e9ecef; /* Lighter border */
            /* color: var(--text-dark) !important; /* This might be too general, let's rely on specific label/value */
        }}

        /* THIS IS THE CRITICAL FIX for metric card label text */
        div[data-testid="stMetric"] label {{ /* Targets the <label> element directly */
            font-size: 0.95rem !important; 
            color: var(--text-neutral) !important; /* Use a defined darkish gray */
            font-weight: 500 !important;
            display: block; /* Ensure it takes full width for centering if needed */
            margin-bottom: 0.5rem; /* Add some space below label */
        }}

        /* And for the metric card value text */
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ 
            font-size: 2.2rem !important; /* Larger value */
            font-weight: 700 !important; 
            color: var(--primary) !important; 
            line-height: 1.2 !important;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{ /* If you use delta values */
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }}
        
        /* Other styles */
        .stButton button {{ border-radius: 8px !important; font-weight: 600 !important; padding: 0.5rem 1rem !important; transition: all 0.3s !important; border: none !important; }}
        .stButton button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }}
        .stButton button[kind="primary"] {{ background-color: var(--primary) !important; color: white !important; }}
        .stButton button[kind="secondary"]:not(:hover) {{ background-color: var(--light) !important; color: var(--text-dark) !important; border: 1px solid var(--neutral) !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
        .stTabs [data-baseweb="tab"] {{ border-radius: 6px 6px 0px 0px; padding: 12px 18px; background-color: var(--light); font-weight: 500; color: var(--text-neutral); }}
        .stTabs [aria-selected="true"] {{ background-color: var(--content-bg); border-bottom: 3px solid var(--primary); box-shadow: 0 -2px 5px rgba(0,0,0,0.05); color: var(--primary); }}
        .custom-divider {{ height: 2px; background-image: linear-gradient(to right, var(--primary), var(--accent)); border-radius: 3px; margin: 1.5rem 0; }}
        .status-badge {{ padding: 5px 12px; border-radius: 16px; font-weight: 600; font-size: 0.85rem; display: inline-block; text-align: center; }}
        .status-ok {{ background-color: rgba(16,185,129,0.15); color: {THEME["success"]}; }}
        .status-warning {{ background-color: rgba(255,152,0,0.15); color: {THEME["warning"]}; }}
        .status-error {{ background-color: rgba(220,53,69,0.15); color: {THEME["danger"]}; }}
        /* ... rest of your CSS ... */
    </style>""", unsafe_allow_html=True)

def initialize_session_state():
    defaults = {
        'confidence_thresh': 0.25, 'iou_thresh': 0.45, 'expected_count': 0,
        'enable_pairwise_overhang': True, 'pairwise_max_overhang': 0.25, 'pairwise_vertical_prox': 10,
        'enable_advanced_stack_analysis': True, 'max_stack_height': 5, 'stack_vertical_prox': 10, 'stack_align_thresh_ratio': 0.15,
        'enable_zone_counting': True,
        'defined_zones': [{'name': f"Zone {i+1}", 'coords': (0,0,100,100), 'defined': False, 'count': 0, 'color': THEME[list(THEME.keys())[i % len(THEME.keys())]]} for i in range(MAX_ZONES)],
        'results': {'detected_count': 0, 'boxes_data': [], 'pairwise_issues': [], 'pairwise_overhang_ids': [], 'stacks_data': [], 'zones_with_counts': [], 'compliance_score': 100, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'processing_time_detection': 0, 'processing_time_analysis': 0, 'image_filename': 'N/A', 'settings_snapshot': {}},
        'analysis_history': [], 'current_page': "dashboard", 'current_image': None, 'processed_image': None, 'dark_mode': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def update_and_save_history(new_result=None):
    if new_result:
        st.session_state.analysis_history.insert(0, new_result)
        st.session_state.analysis_history = st.session_state.analysis_history[:100]
    try:
        with open(HISTORY_FILE, 'w') as f: json.dump(st.session_state.analysis_history, f, indent=4, default=str)
    except Exception as e: st.error(f"Error saving analysis history: {e}")

def load_user_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f: settings = json.load(f)
            for key, value in settings.items():
                if key == 'defined_zones':
                    loaded_zones = value
                    default_zones_template = [{'name': f"Zone {i+1}", 'coords': (0,0,100,100), 'defined': False, 'count': 0, 'color': THEME[list(THEME.keys())[i % len(THEME.keys())]]} for i in range(MAX_ZONES)]
                    merged_zones = [loaded_zones[i] if i < len(loaded_zones) and isinstance(loaded_zones[i], dict) and loaded_zones[i].get('name') else default_zones_template[i] for i in range(MAX_ZONES)]
                    st.session_state.defined_zones = merged_zones[:MAX_ZONES]
                elif key in st.session_state: st.session_state[key] = value
        except Exception as e: st.error(f"Error loading settings: {e}. Using defaults.")

def save_user_settings():
    settings_to_save = {key: st.session_state[key] for key in [
        'confidence_thresh', 'iou_thresh', 'expected_count', 'enable_pairwise_overhang', 'pairwise_max_overhang',
        'pairwise_vertical_prox', 'enable_advanced_stack_analysis', 'max_stack_height', 'stack_vertical_prox',
        'stack_align_thresh_ratio', 'enable_zone_counting', 'defined_zones', 'dark_mode']}
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(settings_to_save, f, indent=4, default=str)
        st.toast("Settings saved!", icon="✅")
    except Exception as e: st.error(f"Error saving settings: {e}")

def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

@st.cache_resource
def load_yolo_model(model_path_str):
    model_path_obj = Path(model_path_str)
    if not model_path_obj.exists():
        st.error(f"Model file not found: {model_path_obj}. Place 'best.pt' in '{MODEL_DIR.name}'.")
        return None
    try:
        model = YOLO(model_path_obj)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def perform_detection(model, image_pil, confidence_threshold=0.25, iou_threshold=0.45):
    if model is None: return None, 0, [], 0
    try:
        start_time = time.time()
        results = model.predict(source=image_pil, conf=confidence_threshold, iou=iou_threshold, verbose=False)
        detected_count = len(results[0].boxes)
        boxes_data = []
        if detected_count > 0:
            for i in range(len(results[0].boxes.xyxy)):
                b = results[0].boxes.xyxy[i].cpu().numpy()
                conf, cls = results[0].boxes.conf[i].cpu().numpy(), results[0].boxes.cls[i].cpu().numpy()
                w, h_box = b[2] - b[0], b[3] - b[1]
                boxes_data.append({'xyxy': b.tolist(), 'confidence': float(conf), 'id': i, 'class_id': int(cls), 'center_x': (b[0] + b[2]) / 2, 'center_y': (b[1] + b[3]) / 2, 'width': w, 'height': h_box, 'used_in_stack': False, 'area': w * h_box, 'aspect_ratio': w / h_box if h_box > 0 else 0})
        processing_time = time.time() - start_time
        return image_pil, detected_count, boxes_data, processing_time
    except Exception as e:
        st.error(f"Error during object detection: {e}")
        return None, 0, [], 0

def get_box_width(box_data): return box_data['width']

def analyze_pairwise_overhang(boxes_data, max_overhang_percent=0.25, vertical_proximity_threshold=10):
    if not boxes_data: return [], set()
    issues, non_compliant_box_ids = [], set()
    for i in range(len(boxes_data)):
        top_box, top_xyxy, top_id = boxes_data[i], np.array(boxes_data[i]['xyxy']), boxes_data[i].get('id', i)
        for j in range(len(boxes_data)):
            if i == j: continue
            bottom_box, bottom_xyxy, bottom_id = boxes_data[j], np.array(boxes_data[j]['xyxy']), boxes_data[j].get('id', j)
            if not ((top_xyxy[1] < bottom_xyxy[1]) and (abs(top_xyxy[3] - bottom_xyxy[1]) < vertical_proximity_threshold)): continue
            horizontal_overlap = max(0, min(top_xyxy[2], bottom_xyxy[2]) - max(top_xyxy[0], bottom_xyxy[0]))
            if horizontal_overlap <= min(get_box_width(top_box), get_box_width(bottom_box)) * 0.3: continue
            allowed_px = get_box_width(bottom_box) * max_overhang_percent
            over_left, over_right = max(0, bottom_xyxy[0] - top_xyxy[0]), max(0, top_xyxy[2] - bottom_xyxy[2])
            if over_left > allowed_px or over_right > allowed_px:
                desc_parts = []
                if over_left > allowed_px: desc_parts.append(f"left by ~{int(over_left)}px ({int(over_left/get_box_width(bottom_box)*100)}%)")
                if over_right > allowed_px: desc_parts.append(f"right by ~{int(over_right)}px ({int(over_right/get_box_width(bottom_box)*100)}%)")
                desc = f"Box {top_id} overhangs Box {bottom_id} on the " + " and ".join(desc_parts) + "."
                if not any(f"Box {top_id} overhangs Box {bottom_id}" in existing_issue.get('desc','') for existing_issue in issues):
                    issues.append({'desc': desc, 'top_id': top_id, 'bottom_id': bottom_id})
                non_compliant_box_ids.update([top_id, bottom_id])
    return [item['desc'] for item in issues], non_compliant_box_ids

def is_box_on_top(top_box, bottom_box, vertical_prox_thresh, min_horiz_overlap_ratio=0.4):
    top_xyxy, bottom_xyxy = np.array(top_box['xyxy']), np.array(bottom_box['xyxy'])
    if not ((top_xyxy[1] < bottom_xyxy[1]) and (abs(top_xyxy[3] - bottom_xyxy[1]) < vertical_prox_thresh)): return False
    overlap = max(0, min(top_xyxy[2], bottom_xyxy[2]) - max(top_xyxy[0], bottom_xyxy[0]))
    return overlap >= (min(top_box['width'], bottom_box['width']) * min_horiz_overlap_ratio)

def identify_stacks(boxes_data, vertical_prox_thresh):
    if not boxes_data: return []
    for box in boxes_data: box['used_in_stack'] = False
    sorted_boxes = sorted(boxes_data, key=lambda b: (b['xyxy'][3], b['xyxy'][0]))
    identified_stacks, stack_id_counter = [], 0
    for base_box in sorted_boxes:
        if base_box['used_in_stack']: continue
        current_stack, current_top = [base_box], base_box
        base_box['used_in_stack'] = True
        while True:
            potential_above = [b for b in boxes_data if not b['used_in_stack'] and b['xyxy'][3] < current_top['xyxy'][1]]
            best_candidate, min_center_diff = None, float('inf')
            for cand in potential_above:
                if is_box_on_top(cand, current_top, vertical_prox_thresh):
                    center_diff = abs(cand['center_x'] - current_top['center_x'])
                    if center_diff < min_center_diff: best_candidate, min_center_diff = cand, center_diff
            if best_candidate:
                current_stack.append(best_candidate); best_candidate['used_in_stack'] = True; current_top = best_candidate
            else: break
        if current_stack:
            identified_stacks.append({'id': stack_id_counter, 'boxes': current_stack[::-1], 'num_boxes': len(current_stack), 'base_width': current_stack[0]['width'], 'base_height': current_stack[0]['height'], 'base_area': current_stack[0]['area'], 'total_height': sum(b['height'] for b in current_stack), 'avg_box_size': sum(b['area'] for b in current_stack) / len(current_stack) if current_stack else 0}); stack_id_counter += 1
    return identified_stacks

def analyze_identified_stacks(stacks, max_stack_height, stack_align_thresh_ratio):
    for stack in stacks:
        stack['height_compliant'] = stack['num_boxes'] <= max_stack_height
        total_dev_ratio, align_issues, stack['deviation_points'] = 0, 0, []
        if stack['num_boxes'] > 1:
            for i in range(stack['num_boxes'] - 1):
                top_b, bottom_b = stack['boxes'][i], stack['boxes'][i+1]
                center_dev = abs(top_b['center_x'] - bottom_b['center_x'])
                dev_ratio = center_dev / (bottom_b['width'] if bottom_b['width'] > 0 else 1)
                total_dev_ratio += dev_ratio
                stack['deviation_points'].append({'top_box_id': top_b['id'], 'bottom_box_id': bottom_b['id'], 'center_deviation_px': center_dev, 'deviation_ratio': dev_ratio, 'severe': dev_ratio > stack_align_thresh_ratio})
                if dev_ratio > stack_align_thresh_ratio: align_issues += 1
            avg_dev_ratio = total_dev_ratio / (stack['num_boxes'] - 1)
            stack.update({'avg_dev_ratio': avg_dev_ratio, 'alignment_issues': align_issues, 'max_deviation_ratio': max((p['deviation_ratio'] for p in stack['deviation_points']), default=0)})
            align_score = max(0, (100 - (avg_dev_ratio * 200)) - (align_issues * 20))
            stack['alignment_score'] = round(align_score, 1)
            stack['alignment_quality'] = "Poor" if align_issues > 0 else ("Fair" if avg_dev_ratio > stack_align_thresh_ratio / 2 else "Good")
        else: stack.update({'avg_dev_ratio': 0, 'alignment_issues': 0, 'alignment_quality': "N/A", 'alignment_score': 100, 'max_deviation_ratio': 0, 'deviation_points':[]})
        stack['overall_status'] = "Height Issue" if not stack['height_compliant'] else ("Alignment Issue" if stack.get('alignment_quality') in ["Poor", "Fair"] else "OK")
        height_score = 100 if stack['height_compliant'] else max(0, 100 - ((stack['num_boxes'] - max_stack_height) * 25))
        stack['stability_score'] = round((height_score * 0.6 + stack.get('alignment_score', 100) * 0.4), 1)
    return stacks

def assign_cartons_to_zones(boxes_data, defined_zones_snapshot):
    current_zones_analysis = [dict(z) for z in defined_zones_snapshot]
    for zone in current_zones_analysis:
        if zone.get('defined', False): zone.update({'count': 0, 'boxes': [], 'total_area': 0})
    for box in boxes_data:
        cx, cy, box_assigned = box['center_x'], box['center_y'], False
        for zone in current_zones_analysis:
            if zone.get('defined', False):
                zx_min, zy_min, zx_max, zy_max = zone['coords']
                if zx_min <= cx < zx_max and zy_min <= cy < zy_max:
                    zone['count'] += 1; zone['boxes'].append(box['id']); zone['total_area'] += box['area']
                    box_assigned = True; break
        box['unassigned_to_zone'] = not box_assigned
    for zone in current_zones_analysis:
        if zone.get('defined', False):
            zw, zh = zone['coords'][2] - zone['coords'][0], zone['coords'][3] - zone['coords'][1]
            za = zw * zh if zw > 0 and zh > 0 else 0
            zone['utilization'] = (zone['total_area'] / za) if za > 0 else 0
            zone['density'] = (zone['count'] * 1000 / za) if za > 0 else 0
    return [z for z in current_zones_analysis if z.get('defined')]

def calculate_compliance_score(results, expected_count):
    base_score, deductions = 100, 0
    if results['pairwise_issues']: deductions += min(30, len(results['pairwise_issues']) * 5)
    if results['stacks_data']:
        problem_stacks = sum(1 for s in results['stacks_data'] if s['overall_status'] != "OK")
        deductions += min(40, problem_stacks * 10 + sum(max(0, 75 - s['stability_score'])*0.1 for s in results['stacks_data'] if s['overall_status'] != "OK" ))
    if results.get('zones_with_counts') and results['settings_snapshot']['enable_zone_counting']:
        assigned_boxes = sum(zone['count'] for zone in results['zones_with_counts'])
        unassigned_boxes = max(0, results['detected_count'] - assigned_boxes)
        if unassigned_boxes > 0: deductions += min(20, unassigned_boxes * 3)
    if expected_count > 0 and results['detected_count'] != expected_count:
        deductions += min(15, abs(results['detected_count'] - expected_count) * 5)
    return max(0, int(base_score - deductions))

def draw_on_image(image_np_rgb, boxes_data, pairwise_overhang_ids, stacks_data, defined_zones_with_counts, show_box_ids=True, show_stack_labels=True, show_zone_labels=True):
    vis_image = image_np_rgb.copy()
    font_scale, thickness = max(0.5, vis_image.shape[0] / 1500), max(1, vis_image.shape[0] // 500)
    if st.session_state.enable_zone_counting and defined_zones_with_counts:
        for zone_data in defined_zones_with_counts:
            if zone_data.get('defined', False):
                x1,y1,x2,y2 = [int(c) for c in zone_data['coords']]; color_bgr = hex_to_rgb(zone_data['color'])
                overlay = vis_image.copy(); cv2.rectangle(overlay, (x1,y1), (x2,y2), color_bgr, -1)
                vis_image = cv2.addWeighted(overlay, 0.10, vis_image, 0.90, 0)
                cv2.rectangle(vis_image, (x1,y1), (x2,y2), color_bgr, thickness + 1)
                if show_zone_labels:
                    label_text = f"{zone_data['name']}: {zone_data['count']}"
                    (w_text, h_text),_ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.9, thickness)
                    cv2.rectangle(vis_image, (x1,y1-h_text-5), (x1+w_text+4,y1-2), color_bgr, -1)
                    cv2.putText(vis_image, label_text, (x1+2,y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.9, (255,255,255), thickness)
    for box in boxes_data:
        x1,y1,x2,y2 = [int(c) for c in box['xyxy']]; box_id = box['id']
        color_hex, current_thickness = (THEME["danger"], thickness+1) if box_id in pairwise_overhang_ids else (THEME["primary"], thickness)
        cv2.rectangle(vis_image, (x1,y1), (x2,y2), hex_to_rgb(color_hex), current_thickness)
        if show_box_ids:
            label_id_text = f"ID:{box_id}"; (w_id_text, h_id_text),_ = cv2.getTextSize(label_id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, current_thickness)
            cv2.rectangle(vis_image, (x1,y1-h_id_text-4), (x1+w_id_text,y1), hex_to_rgb(color_hex), -1)
            cv2.putText(vis_image, label_id_text, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255,255,255), current_thickness)
    if st.session_state.enable_advanced_stack_analysis and stacks_data:
        for stack in stacks_data:
            stack_color_hex = THEME["accent"] if stack['overall_status'] == "OK" else THEME["warning"]
            stack_color_rgb = hex_to_rgb(stack_color_hex)
            for i in range(len(stack['boxes']) -1):
                box_top, box_bottom = stack['boxes'][i], stack['boxes'][i+1]
                c1,c2 = (int(box_top['center_x']),int(box_top['center_y'])), (int(box_bottom['center_x']),int(box_bottom['center_y']))
                cv2.line(vis_image, c1, c2, stack_color_rgb, thickness + 1, cv2.LINE_AA)
            if show_stack_labels and stack['boxes']:
                highest_box = stack['boxes'][0]; sx,sy = int(highest_box['xyxy'][0]), int(highest_box['xyxy'][1])
                label_stack_text = f"S{stack['id']}({stack['num_boxes']})"
                (w_stack_text, h_stack_text),_ = cv2.getTextSize(label_stack_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_y_pos = sy - 5
                rect_y1_pos, rect_y2_pos = label_y_pos - h_stack_text - 2, label_y_pos + 2
                cv2.rectangle(vis_image, (sx, rect_y1_pos), (sx+w_stack_text+4, rect_y2_pos), stack_color_rgb, -1)
                cv2.putText(vis_image, label_stack_text, (sx+2, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return vis_image

def process_image_and_analyze(image_pil, model):
    if image_pil is None or model is None: st.warning("Upload image & ensure model loaded."); return
    img_draw, count, boxes, t_det = perform_detection(model, image_pil, st.session_state.confidence_thresh, st.session_state.iou_thresh)
    if img_draw is None: st.error("Detection failed."); return
    st.session_state.current_image = image_pil
    results = {'image_filename': getattr(image_pil, 'name', f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"), 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), 'detected_count': count, 'boxes_data': boxes, 'pairwise_issues': [], 'pairwise_overhang_ids': [], 'stacks_data': [], 'zones_with_counts': [], 'compliance_score': 100, 'processing_time_detection': t_det, 'processing_time_analysis': 0, 'settings_snapshot': {key: st.session_state[key] for key in ['confidence_thresh', 'iou_thresh', 'expected_count', 'enable_pairwise_overhang', 'pairwise_max_overhang', 'pairwise_vertical_prox', 'enable_advanced_stack_analysis', 'max_stack_height', 'stack_vertical_prox', 'stack_align_thresh_ratio', 'enable_zone_counting']}}
    results['settings_snapshot']['defined_zones_snapshot'] = [dict(z) for z in st.session_state.defined_zones if z.get('defined')]

    t_ana_start = time.time()
    if count > 0:
        if results['settings_snapshot']['enable_pairwise_overhang']:
            iss, ids = analyze_pairwise_overhang(boxes, results['settings_snapshot']['pairwise_max_overhang'], results['settings_snapshot']['pairwise_vertical_prox'])
            results['pairwise_issues'], results['pairwise_overhang_ids'] = iss, list(ids)
        if results['settings_snapshot']['enable_advanced_stack_analysis']:
            stacks = identify_stacks(boxes, results['settings_snapshot']['stack_vertical_prox'])
            results['stacks_data'] = analyze_identified_stacks(stacks, results['settings_snapshot']['max_stack_height'], results['settings_snapshot']['stack_align_thresh_ratio'])
        if results['settings_snapshot']['enable_zone_counting']:
            results['zones_with_counts'] = assign_cartons_to_zones(boxes, results['settings_snapshot']['defined_zones_snapshot'])
    results['processing_time_analysis'] = time.time() - t_ana_start
    results['compliance_score'] = calculate_compliance_score(results, results['settings_snapshot']['expected_count'])

    proc_img = draw_on_image(np.array(img_draw.convert('RGB')), results['boxes_data'], set(results['pairwise_overhang_ids']), results['stacks_data'], results['zones_with_counts'])
    st.session_state.processed_image, st.session_state.results = proc_img, results
    update_and_save_history(results)
    st.toast(f"Analysis complete! Compliance: {results['compliance_score']}%", icon="🎉")

def display_dashboard():
    colored_header(label="📈 Analytics Dashboard", description="Overview of system performance and carton analysis.", color_name="blue-70")
    add_vertical_space(1)
    history = st.session_state.analysis_history
    if not history:
        st.info("No analysis history. Perform analysis for dashboard stats.")
        if st.button("🚀 Start New Analysis", type="primary", key="dash_start_analysis_final"): # Unique key
            st.session_state.current_page = "analysis"
            st.rerun()
        return

    c1, c2, c3 = st.columns(3)
    total_analyses = len(history)
    avg_compliance = sum(item['compliance_score'] for item in history) / total_analyses if total_analyses > 0 else 0
    low_compliance_runs = sum(1 for item in history[:10] if item['compliance_score'] < 70)

    with c1: st.metric("Total Analyses", total_analyses)
    with c2: st.metric("Avg Compliance", f"{avg_compliance:.1f}%")
    with c3: st.metric("Low Compliance (Last 10)", low_compliance_runs, help="Runs <70% compliance.")
    
    style_metric_cards(border_left_color=THEME["primary"]) 
    
    add_vertical_space(2)
    st.subheader("📊 Performance Trends")
    t1, t2, t3 = st.tabs(["Compliance", "Detections", "Processing Time"])
    
    if not history: return 

    dfh = pd.DataFrame([
        {'TS': datetime.strptime(item['timestamp'].split('.')[0], "%Y-%m-%d %H:%M:%S"),
         'Compliance': item['compliance_score'], 'Detections': item['detected_count'],
         'T_Detect': item.get('processing_time_detection', 0), 'T_Analyze': item.get('processing_time_analysis', 0)}
        for item in history
    ]).sort_values('TS')
    
    common_layout = {'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}
    with t1:
        if not dfh.empty:
            fig = px.line(dfh, x='TS', y='Compliance', title="Compliance Trend", markers=True, color_discrete_sequence=[THEME["success"]])
            fig.update_layout(**common_layout, yaxis_title="Compliance (%)")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No data for compliance trend.")
    with t2:
        if not dfh.empty:
            fig = px.bar(dfh, x='TS', y='Detections', title="Detections/Analysis", color_discrete_sequence=[THEME["info"]])
            fig.update_layout(**common_layout, yaxis_title="Detected Cartons")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No data for detection activity.")
    with t3:
        if not dfh.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dfh['TS'], y=dfh['T_Detect'], mode='lines+markers', name='Detection', line={'color': THEME["accent"]}))
            fig.add_trace(go.Scatter(x=dfh['TS'], y=dfh['T_Analyze'], mode='lines+markers', name='Analysis', line={'color': THEME["warning"]}))
            fig.update_layout(title="Processing Times (s)", yaxis_title="Time (s)", **common_layout)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No data for processing time.")
    if st.button("🔍 Go to Analysis Page", type="primary", key="dash_to_analysis_final"): # Unique key
        st.session_state.current_page = "analysis"
        st.rerun()

def display_analysis_page(model):
    colored_header(label="🔍 Carton Analysis", description="Upload image to detect & analyze carton arrangements.", color_name="green-70")
    c1,c2 = st.columns([2,1])
    with c1: uploaded_file = st.file_uploader("🖼️ Choose image...", type=["jpg","jpeg","png"], key="uploader", label_visibility="collapsed")
    with c2:
        if st.button("🚀 Analyze Image", type="primary", disabled=uploaded_file is None, use_container_width=True, key="analyze_btn"):
            if uploaded_file:
                with st.spinner("🔬 Analyzing image..."): process_image_and_analyze(Image.open(uploaded_file), model)
            else: st.warning("Please upload an image.")
    if st.session_state.get('processed_image') is not None:
        add_vertical_space(1); st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True); add_vertical_space(1)
        st.subheader("🖼️ Processed Image & Summary")
        res_c1, res_c2 = st.columns([3,2])
        with res_c1: st.image(st.session_state.processed_image, caption=f"Processed: {st.session_state.results.get('image_filename', 'N/A')}", use_container_width=True)
        with res_c2:
            results = st.session_state.results
            score_color = THEME['success'] if results['compliance_score'] >= 80 else (THEME['warning'] if results['compliance_score'] >= 50 else THEME['danger'])
            st.markdown(f"""<div class='styled-card animated'>
                <p style='font-size:1.1em; font-weight:bold; color:var(--neutral); margin-bottom:5px;'>Compliance Score</p>
                <p style='font-size:2.5em; font-weight:bold; color:{score_color}; margin-bottom:15px;'>{results['compliance_score']}%</p>
                <p style='margin-bottom:5px;'><strong>Detected Cartons:</strong> {results['detected_count']}</p>
                <p style='margin-bottom:5px;'><strong>Detection Time:</strong> {results.get('processing_time_detection',0):.2f}s</p>
                <p style='margin-bottom:15px;'><strong>Analysis Time:</strong> {results.get('processing_time_analysis',0):.2f}s</p>
            </div>""", unsafe_allow_html=True)
            if results['settings_snapshot'].get('expected_count', 0) > 0:
                exp, det = results['settings_snapshot']['expected_count'], results['detected_count']
                status_cls = "status-ok" if det == exp else "status-warning"
                st.markdown(f"<div class='styled-card animated'><p><strong>Expected vs. Detected:</strong> {exp} vs {det} <span class='status-badge {status_cls}'>{ 'Match' if det == exp else 'Mismatch'}</span></p></div>", unsafe_allow_html=True)
            if st.button("📄 Download Analysis Report (JSON)", key="dl_json_report", use_container_width=True, type="secondary"):
                json_str = json.dumps(results, indent=4, default=str); b64 = base64.b64encode(json_str.encode()).decode()
                ts_fn = results.get("timestamp", datetime.now().strftime("%Y%m%d%H%M%S%f")).replace(":", "-").replace(".","-")
                st.markdown(f'<a href="data:file/json;base64,{b64}" download="analysis_report_{ts_fn}.json" style="display:block; text-align:center; padding:0.5rem; background-color:var(--primary); color:white; border-radius:8px; text-decoration:none;">Click to Download JSON Report</a>', unsafe_allow_html=True)
            if st.button("🗑️ Clear Current Analysis", use_container_width=True, key="clear_analysis"):
                st.session_state.current_image = None; st.session_state.processed_image = None
                st.session_state.results = {'detected_count': 0, 'boxes_data': [], 'pairwise_issues': [], 'pairwise_overhang_ids': [], 'stacks_data': [], 'zones_with_counts': [], 'compliance_score': 100, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'processing_time_detection': 0, 'processing_time_analysis': 0, 'image_filename': 'N/A', 'settings_snapshot': {}}
                st.rerun()
        add_vertical_space(1); st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True); add_vertical_space(1)
        st.subheader("📋 Detailed Analysis Results"); display_detailed_results_tabs(results)
    elif st.session_state.current_image is not None: st.image(st.session_state.current_image, caption="Uploaded Image (Pending Analysis)", use_container_width=True)
    else: st.info("Upload an image and click 'Analyze Image' to see results.")

def display_detailed_results_tabs(results, is_history_view=False):
    tabs = ["🔎 Detections", "⚖️ Pairwise Overhang", "📚 Stack Analysis", "🗺️ Zone Counts", "📝 Summary"]
    det_tab, overhang_tab, stack_tab, zone_tab, summary_tab = st.tabs(tabs)
    settings = results.get('settings_snapshot', st.session_state)

    with det_tab:
        st.write(f"**Total Detected Cartons:** {results['detected_count']}")
        if results['boxes_data']:
            df_boxes = pd.DataFrame(results['boxes_data'])[['id', 'confidence', 'center_x', 'center_y', 'width', 'height', 'area', 'aspect_ratio', 'class_id']]
            df_boxes.columns = ['ID', 'Conf', 'CX', 'CY', 'W', 'H', 'Area', 'Aspect', 'Class']
            for col in ['Conf', 'CX', 'CY', 'W', 'H', 'Area', 'Aspect']:
                if col in df_boxes.columns: df_boxes[col] = df_boxes[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, np.floating)) else x)
            st.dataframe(df_boxes, use_container_width=True, height=min(300, len(df_boxes)*38 + 38))
        else: st.info("No cartons detected or data unavailable.")
    with overhang_tab:
        enabled = settings.get('enable_pairwise_overhang', False)
        if not enabled and not is_history_view: st.info("Pairwise overhang analysis disabled in Settings.")
        elif results['pairwise_issues']:
            st.warning(f"**Found {len(results['pairwise_issues'])} pairwise overhang issues:**")
            for issue in results['pairwise_issues']: st.markdown(f"- {issue}")
            st.markdown(f"**Affected Box IDs:** {', '.join(map(str, sorted(list(set(results.get('pairwise_overhang_ids',[]))))))}")
        elif enabled: st.success("No pairwise overhang issues detected. ✅")
        else: st.info("Pairwise overhang analysis was disabled for this run.")
    with stack_tab:
        enabled = settings.get('enable_advanced_stack_analysis', False)
        if not enabled and not is_history_view: st.info("Advanced stack analysis disabled in Settings.")
        elif results['stacks_data']:
            st.write(f"**Identified {len(results['stacks_data'])} Stacks:**")
            stack_summary = [{"ID": s['id'], "Boxes": s['num_boxes'], "Status": s['overall_status'], "Stability": f"{s.get('stability_score', 'N/A')}%", "Alignment": f"{s.get('alignment_quality', 'N/A')}", "Box IDs": ", ".join(map(str, [b['id'] for b in s['boxes']]))} for s in results['stacks_data']]
            st.dataframe(pd.DataFrame(stack_summary), use_container_width=True)
            for s_item in results['stacks_data']:
                with st.expander(f"Details for Stack {s_item['id']} (Stability: {s_item.get('stability_score', 'N/A')}%)"):
                    st.write(f"**Status:** {s_item['overall_status']}, **Alignment Quality:** {s_item.get('alignment_quality', 'N/A')}, **Score:** {s_item.get('alignment_score', 'N/A')}%")
                    st.write(f"**Height Compliant:** {'Yes' if s_item['height_compliant'] else 'No (Max: ' + str(settings.get('max_stack_height','N/A')) + ')'}")
                    if s_item.get('deviation_points'):
                        dev_df = pd.DataFrame(s_item['deviation_points']); dev_df['dev_ratio_pct'] = dev_df['deviation_ratio'] * 100
                        fig = px.bar(dev_df, x=dev_df.index.map(lambda i: f"Pair {dev_df.loc[i,'top_box_id']}-{dev_df.loc[i,'bottom_box_id']}"), y='dev_ratio_pct', title=f'Alignment Deviations for Stack {s_item["id"]}', labels={'x':'Pair','dev_ratio_pct':'Deviation (%)'}, color='severe', color_discrete_map={True:THEME['danger'],False:THEME['success']})
                        fig.add_hline(y=settings.get('stack_align_thresh_ratio',0.15)*100, line_dash="dash", line_color="red", annotation_text="Threshold")
                        st.plotly_chart(fig, use_container_width=True)
        elif enabled: st.success("No stacks identified or all stacks compliant. ✅")
        else: st.info("Advanced stack analysis was disabled for this run.")
    with zone_tab:
        enabled = settings.get('enable_zone_counting', False)
        if not enabled and not is_history_view: st.info("Zone counting disabled in Settings.")
        elif results.get('zones_with_counts'):
            st.write(f"**Carton Counts per Defined Zone:**")
            zone_df_data = [{"Zone": z['name'], "Count": z['count'], "Util.": f"{z.get('utilization',0)*100:.1f}%", "Density": f"{z.get('density',0):.2f}", "Box IDs": ", ".join(map(str, z.get('boxes',[])))} for z in results['zones_with_counts'] if z.get('defined', True)]
            if zone_df_data:
                st.dataframe(pd.DataFrame(zone_df_data), use_container_width=True)
                fig = px.pie(pd.DataFrame(zone_df_data), values='Count', names='Zone', title='Carton Distribution by Zone', color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No defined zones had cartons, or no zones active.")
            unassigned = results['detected_count'] - sum(z['count'] for z in results['zones_with_counts'] if z.get('defined', True))
            if unassigned > 0: st.warning(f"**{unassigned} cartons not assigned to any defined zone.**")
        elif enabled: st.success("All cartons assigned or no cartons detected. ✅")
        else: st.info("Zone counting was disabled for this run.")
    with summary_tab:
        st.subheader("Overall Summary & Recommendations")
        score = results['compliance_score']
        status_text, color = ("Excellent", THEME["success"]) if score >=90 else \
                       (("Good", THEME["success"]) if 70 <= score < 90 else \
                        (("Fair - Needs Improvement", THEME["warning"]) if 50 <= score < 70 else \
                         ("Poor - Action Required", THEME["danger"])))
        st.markdown(f"**Overall Compliance Score:** <span class='status-badge' style='background-color:{color}20; color:{color}; font-size:1.2em; padding:8px 15px;'>{score}% ({status_text})</span>", unsafe_allow_html=True)
        recs = []
        if results['pairwise_issues']: recs.append("Address carton overhangs.")
        if any(s['overall_status'] != "OK" for s in results['stacks_data']): recs.append("Review stack heights & alignment.")
        unassigned_sum = results['detected_count'] - sum(z.get('count',0) for z in results.get('zones_with_counts',[]) if z.get('defined',True))
        if unassigned_sum > 0 and settings.get('enable_zone_counting'): recs.append("Some cartons outside defined zones.")
        if settings.get('expected_count', 0) > 0 and results['detected_count'] != settings['expected_count']: recs.append(f"Count mismatch (Expected: {settings['expected_count']}, Detected: {results['detected_count']}).")
        if not recs and score >= 90: st.success("All checks passed! 👍")
        elif not recs: recs.append("Review placement for potential score improvement.")
        if recs:
            st.markdown("**Key Areas for Attention:**")
            for i, r_text in enumerate(recs):
                st.markdown(f"{i+1}. {r_text}")
        if is_history_view and 'settings_snapshot' in results:
            with st.expander("Analysis Settings Used For This Run"): st.json(results['settings_snapshot'], expanded=False)

def display_history_page():
    colored_header(label="📜 Analysis History", description="Review past analyses and their results.", color_name="violet-70")
    history = st.session_state.analysis_history
    if not history: st.info("No analysis history available."); return
    options = {f"{item['timestamp']} | Score: {item['compliance_score']}% | File: {item.get('image_filename', 'N/A')}": item for item in history}
    selected_key = st.selectbox("Select an analysis:", options=[""] + list(options.keys()), format_func=lambda x: "Select..." if x == "" else x, key="hist_sel")
    if selected_key:
        selected_result = options[selected_key]
        st.markdown(f"### Details for Analysis: {selected_result['timestamp']}")
        display_detailed_results_tabs(selected_result, is_history_view=True)
        if st.button(f"❌ Delete this entry ({selected_result['timestamp']})", key=f"del_hist_{selected_result['timestamp'].replace(':','-').replace('.','-')}", type="secondary"): # Ensure key is valid
            st.session_state.analysis_history = [item for item in st.session_state.analysis_history if item['timestamp'] != selected_result['timestamp']]
            update_and_save_history(); st.toast(f"Deleted: {selected_result['timestamp']}", icon="🗑️"); st.rerun()
    add_vertical_space(1); st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True); add_vertical_space(1)
    if st.button("⚠️ Clear Entire Analysis History", type="primary", key="clear_all_hist_btn_2"): # Unique key
        st.session_state.analysis_history = []; update_and_save_history(); st.success("History cleared."); st.rerun()

def display_settings_page():
    colored_header(label="⚙️ System Configuration", description="Adjust detection parameters, analysis modules, and zones.", color_name="orange-70")
    s_tabs = st.tabs(["🔧 Detection & Analysis", "🗺️ Zone Management", "🎨 Appearance", "💾 Data Management"])
    with s_tabs[0]:
        st.subheader("Object Detection")
        c1,c2=st.columns(2)
        st.session_state.confidence_thresh = c1.slider("Confidence Threshold", 0.01,1.0,st.session_state.confidence_thresh,0.01,help="Min confidence.")
        st.session_state.iou_thresh = c2.slider("IoU Threshold (NMS)",0.01,1.0,st.session_state.iou_thresh,0.01,help="IoU for NMS.")
        st.session_state.expected_count = st.number_input("Expected Carton Count (0 to disable)", min_value=0, max_value=10000, value=st.session_state.expected_count, step=1)
        st.markdown("---"); st.subheader("Pairwise Overhang Analysis")
        st.session_state.enable_pairwise_overhang=st.checkbox("Enable",st.session_state.enable_pairwise_overhang,key="cb_pwo")
        if st.session_state.enable_pairwise_overhang:
            c1,c2=st.columns(2); st.session_state.pairwise_max_overhang=c1.slider("Max Overhang Ratio",0.0,0.5,st.session_state.pairwise_max_overhang,0.01); st.session_state.pairwise_vertical_prox=c2.slider("Vertical Proximity (px)",1,50,st.session_state.pairwise_vertical_prox,1)
        st.markdown("---"); st.subheader("Advanced Stack Analysis")
        st.session_state.enable_advanced_stack_analysis=st.checkbox("Enable",st.session_state.enable_advanced_stack_analysis,key="cb_asa")
        if st.session_state.enable_advanced_stack_analysis:
            c1,c2,c3=st.columns(3); st.session_state.max_stack_height=c1.number_input("Max Stack Height (boxes)",1,10,st.session_state.max_stack_height,1); st.session_state.stack_vertical_prox=c2.slider("Stack Vertical Proximity (px)",1,50,st.session_state.stack_vertical_prox,1); st.session_state.stack_align_thresh_ratio=c3.slider("Stack Alignment Threshold Ratio",0.01,0.5,st.session_state.stack_align_thresh_ratio,0.01)
        st.markdown("---"); st.subheader("Zone Counting")
        st.session_state.enable_zone_counting = st.checkbox("Enable", st.session_state.enable_zone_counting, key="cb_zc")
    with s_tabs[1]:
        st.subheader("Define Analysis Zones")
        st.info("Define rectangular zones (Xmin, Ymin, Xmax, Ymax) in pixels.")
        current_zones = st.session_state.defined_zones
        if len(current_zones) < MAX_ZONES:
            for i in range(len(current_zones), MAX_ZONES): current_zones.append({'name':f"Zone {i+1}",'coords':(0,0,100,100),'defined':False,'count':0,'color':THEME[list(THEME.keys())[i%len(THEME.keys())]]})
        st.session_state.defined_zones = current_zones[:MAX_ZONES]
        for i in range(MAX_ZONES):
            zone = st.session_state.defined_zones[i]
            with st.expander(f"Zone {i+1}: {zone.get('name', '')} ({'Enabled' if zone.get('defined') else 'Disabled'})", expanded=zone.get('defined',False)):
                zone['defined'] = st.checkbox(f"Enable this Zone", value=zone.get('defined',False), key=f"zone_en_{i}")
                if zone['defined']:
                    zone['name'] = st.text_input(f"Name", value=zone.get('name', f"Zone {i+1}"), key=f"zone_nm_{i}")
                    cc1,cc2,cc3,cc4 = st.columns(4); coords = [int(c) for c in zone.get('coords',(0,0,100,100))]
                    zone['coords'] = (cc1.number_input("Xmin",value=coords[0],key=f"zxmin_{i}",min_value=0,step=10,max_value=10000), cc2.number_input("Ymin",value=coords[1],key=f"zymin_{i}",min_value=0,step=10,max_value=10000), cc3.number_input("Xmax",value=coords[2],key=f"zxmax_{i}",min_value=0,step=10,max_value=10000), cc4.number_input("Ymax",value=coords[3],key=f"zymax_{i}",min_value=0,step=10,max_value=10000))
                    avail_colors = {name.capitalize():val for name,val in THEME.items() if name not in ['light','dark','secondary','neutral']}; curr_color_val = zone.get('color',THEME["primary"]); c_idx = list(avail_colors.values()).index(curr_color_val) if curr_color_val in avail_colors.values() else 0
                    zone['color'] = avail_colors[st.selectbox("Color",options=list(avail_colors.keys()),index=c_idx,key=f"zcolor_{i}")]
    with s_tabs[2]:
        st.subheader("UI Appearance"); st.session_state.dark_mode = st.toggle("Enable Dark Mode (Conceptual)", value=st.session_state.dark_mode)
        if st.session_state.dark_mode: st.info("Dark mode conceptual. Full support needs extensive CSS.")
    with s_tabs[3]:
        st.subheader("User Data Management")
        if st.button("💾 Save All Current Settings",type="primary",key="save_all_settings_main_2"): save_user_settings() # Unique key
        if st.button("🔄 Load Saved Settings from File",key="load_settings_main_2"): load_user_settings(); st.rerun() # Unique key
        if st.button("↩️ Reset All Settings to Defaults",type="secondary",key="reset_settings_main_2"): # Unique key
            initialize_session_state();
            if SETTINGS_FILE.exists():
                try: os.remove(SETTINGS_FILE); st.toast("Settings file removed.",icon="🗑️")
                except Exception as e: st.error(f"Could not remove settings file: {e}")
            save_user_settings(); st.success("All settings reset to defaults."); st.rerun()
    if st.sidebar.button("💾 Save Current Settings", key="sidebar_save_settings_2", use_container_width=True, type="primary"): save_user_settings() # Unique key

# --- Main App ---
def main():
    load_custom_css()
    initialize_session_state()
    load_user_settings()

    if 'analysis_history' not in st.session_state: st.session_state.analysis_history = []
    if not st.session_state.analysis_history and HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f: st.session_state.analysis_history = json.load(f)
        except Exception as e: st.session_state.analysis_history = []; st.error(f"Error loading history: {e}")

    model = load_yolo_model(MODEL_PATH)

    with st.sidebar:
        st.markdown(f"<h1 style='text-align: center; color: {THEME['primary']};'>📦 Carton Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: var(--neutral); font-size:0.9em;'>Advanced Compliance & Audit Tool</p>", unsafe_allow_html=True)
        add_vertical_space(1)
        page_options = ["Dashboard", "Analysis", "History", "Settings"]
        try: default_idx = page_options.index(st.session_state.current_page.capitalize())
        except ValueError: default_idx = 0

        selected_page_title = option_menu(
            menu_title=None, options=page_options,
            icons=["speedometer2", "camera-reels", "hourglass-split", "gear-fill"],
            menu_icon="list-ul", default_index=default_idx, orientation="vertical",
            styles={
                "container": {"padding": "5px !important", "background-color": "transparent"},
                "icon": {"color": THEME["accent"], "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"3px", "--hover-color": "#e9ecef", "border-radius":"5px"},
                "nav-link-selected": {"background-color": THEME["primary"], "color": "white", "font-weight":"600"},
            })

        if st.session_state.current_page != selected_page_title.lower():
            st.session_state.current_page = selected_page_title.lower()
            st.rerun()

        add_vertical_space(2); st.markdown("---")
        with st.expander("🛠️ Quick Controls", expanded=False):
            st.session_state.confidence_thresh = st.slider("Confidence",0.01,1.0,st.session_state.confidence_thresh,0.01,key="sb_conf_q_2") # Unique key
            st.session_state.iou_thresh = st.slider("IoU",0.01,1.0,st.session_state.iou_thresh,0.01,key="sb_iou_q_2") # Unique key
            if st.button("Apply & Save Quick Controls",key="sb_save_qc_2",use_container_width=True): save_user_settings() # Unique key

    if st.session_state.current_page == "dashboard": display_dashboard()
    elif st.session_state.current_page == "analysis": display_analysis_page(model)
    elif st.session_state.current_page == "history": display_history_page()
    elif st.session_state.current_page == "settings": display_settings_page()

if __name__ == "__main__":
    main()