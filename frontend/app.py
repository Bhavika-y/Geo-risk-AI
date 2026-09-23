import streamlit as st
import streamlit.components.v1 as components
import requests
import folium
import rasterio
import numpy as np

from rasterio.warp import transform_bounds
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NASA Smart-Earth-Protection | Responsive GIS Telemetry",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

SUSCEPTIBILITY_RASTER = "outputs/sikkim_landslide_susceptibility.tif"


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "analysis_error" not in st.session_state:
    st.session_state["analysis_error"] = None


# ============================================================
# CUSTOM CSS (Responsive HUD Theme)
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* Pitch Black NASA Command Display */
.stApp {
    background-color: #000000 !important;
    color: #e5e5e5;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
}

/* Remove default Streamlit top/bottom padding */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

iframe {
    display: block;
    border: none;
    width: 100% !important;
}

/* Header HUD Elements */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1a1a1a;
    padding: 10px 20px 8px 20px;
    margin-bottom: 5px;
    font-size: 0.75rem;
    letter-spacing: 2px;
    color: #a0a0a0;
}

.nasa-logo {
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: 6px;
    color: #ffffff;
    font-family: sans-serif;
}

/* Monospace Input Labels */
label {
    font-family: 'Share Tech Mono', monospace !important;
    color: #888888 !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
}

div[data-baseweb="input"] {
    background: #050505 !important;
    border: 1px solid #222222 !important;
    border-radius: 2px !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #00e5ff !important;
}

input {
    color: #ffffff !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* HUD Wireframe Action Buttons */
.stButton > button {
    width: 100%;
    height: 42px;
    border-radius: 2px;
    border: 1px solid #444444;
    background: #080808;
    color: #ffffff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #00e5ff;
    color: #00e5ff;
    background: #021820;
}

/* Minimal Wireframe Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #040404 !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 2px !important;
}

/* Metric Display Style */
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #ffffff !important;
    font-size: 1.6rem !important;
}

/* Section Title HUD Monospace Style */
.section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    color: #888888;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 4px;
}

@media (max-width: 768px) {
    .header-container {
        font-size: 0.65rem;
        padding: 8px 10px;
    }
    .nasa-logo {
        font-size: 1.2rem;
        letter-spacing: 3px;
    }
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# NASA HUD HEADER BAR
# ============================================================

st.markdown(
    """
    <div class="header-container">
        <div>
            GEOSPATIAL-EARTH-MONITORING &nbsp;&nbsp;&nbsp; GIS-X3<br>
            GRID &nbsp;•&nbsp; WGS-84 &nbsp;&nbsp;&nbsp; TIME &nbsp;•&nbsp; 02/18/2026 12:24:34 PM
        </div>
        <div class="nasa-logo">NASA</div>
        <div style="text-align: right;">
            SATELLITE TELEMETRY / LIVE FEED
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FULLY RESPONSIVE GEOSPATIAL EARTH VIEWPORT
# ============================================================

earth_hud_html = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #000000;
    font-family: 'Share Tech Mono', monospace;
    color: #ffffff;
    box-sizing: border-box;
}

#canvas-container {
    width: 100vw;
    height: 100vh;
    position: relative;
}

/* HUD Top Left Event Text */
.hud-top-left {
    position: absolute;
    top: 15px;
    left: 20px;
    font-size: 11px;
    letter-spacing: 1.2px;
    color: #a0a0a0;
    line-height: 1.4;
    pointer-events: none;
    max-width: 260px;
    z-index: 10;
}

.hud-top-left .highlight {
    color: #00e5ff;
    font-weight: bold;
}

.timeline-line {
    width: 1px;
    height: 35px;
    background: #00e5ff;
    margin: 6px 0;
}

/* HUD Top Right Breakdown Text */
.hud-top-right {
    position: absolute;
    top: 15px;
    right: 20px;
    font-size: 11px;
    letter-spacing: 1.2px;
    color: #a0a0a0;
    pointer-events: none;
    width: 280px;
    z-index: 10;
}

.hud-top-right table {
    width: 100%;
    border-collapse: collapse;
}

.hud-top-right td {
    padding: 2px 0;
}

.hud-top-right .val {
    text-align: right;
    color: #ffffff;
}

/* Inset Viewport Box Bottom Right */
.hud-inset-box {
    position: absolute;
    bottom: 70px;
    right: 20px;
    width: 170px;
    height: 100px;
    border: 1px solid #00e5ff;
    background: rgba(0, 5, 12, 0.85);
    pointer-events: none;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 6px;
    box-sizing: border-box;
    z-index: 10;
}

.hud-inset-box .label {
    color: #00e5ff;
    font-size: 9px;
    letter-spacing: 1px;
}

/* Bottom Telemetry Bar Grid */
.hud-bottom-bar {
    position: absolute;
    bottom: 10px;
    left: 20px;
    right: 20px;
    height: 48px;
    border: 1px solid #1a1a1a;
    display: flex;
    align-items: center;
    padding: 0 12px;
    background: #020202;
    font-size: 9px;
    color: #666666;
    letter-spacing: 1px;
    z-index: 10;
    box-sizing: border-box;
}

.hud-bottom-bar .stat-group {
    margin-right: 25px;
}

.hud-bottom-bar .big-val {
    font-size: 16px;
    color: #ffffff;
    letter-spacing: 2px;
    margin-top: 1px;
}

.hud-bottom-bar .graph-container {
    flex-grow: 1;
    height: 100%;
    display: flex;
    align-items: center;
}

/* Responsive Overrides for Smaller Devices */
@media (max-width: 768px) {
    .hud-top-left { font-size: 9px; max-width: 180px; top: 10px; left: 10px; }
    .hud-top-right { font-size: 9px; width: 200px; top: 10px; right: 10px; }
    .hud-inset-box { width: 130px; height: 80px; bottom: 60px; right: 10px; }
    .hud-bottom-bar { bottom: 5px; left: 10px; right: 10px; height: 42px; font-size: 8px; }
    .hud-bottom-bar .big-val { font-size: 13px; }
    .hud-bottom-bar .stat-group { margin-right: 12px; }
}

canvas { display: block; width: 100%; height: 100%; }
</style>
</head>

<body>
<div id="canvas-container">
    <div class="hud-top-left">
        <div class="highlight">GIS TERRAIN SCANNING</div>
        <div>PROJECTION: EPSG:4326</div>
        <div class="timeline-line"></div>
        <div style="color: #ffffff;">SATELLITE RADAR MATRIX</div>
        <div style="margin-top: 4px; font-size: 9px; color: #777777;">
            REALTIME LANDCOVER & TERRAIN TELEMETRY.
        </div>
    </div>

    <div class="hud-top-right">
        <div style="color: #00e5ff; margin-bottom: 4px;">REMOTE SENSING BANDS</div>
        <table>
            <tr>
                <td>SENTINEL-1 SAR:</td>
                <td class="val">ACTIVE</td>
            </tr>
            <tr>
                <td>DEM ELEVATION:</td>
                <td class="val">30M ALOS</td>
            </tr>
            <tr>
                <td>SLOPE ANGLE:</td>
                <td class="val">REALTIME</td>
            </tr>
        </table>
    </div>

    <div class="hud-inset-box">
        <svg width="100%" height="55" viewBox="0 0 100 60" style="margin-bottom: auto;">
            <circle cx="50" cy="30" r="22" stroke="#00e5ff" stroke-width="1" fill="none" stroke-dasharray="2 2"/>
            <circle cx="50" cy="30" r="12" stroke="#00e5ff" stroke-width="1" fill="none"/>
            <line x1="50" y1="5" x2="50" y2="55" stroke="#00e5ff" stroke-width="0.8"/>
            <line x1="25" y1="30" x2="75" y2="30" stroke="#00e5ff" stroke-width="0.8"/>
            <circle cx="58" cy="22" r="2" fill="#d93838"/>
        </svg>
        <div class="label">TARGET RADAR FOCUS</div>
    </div>

    <div class="hud-bottom-bar">
        <div class="stat-group">
            <div>PIXELS RESOLVED:</div>
            <div class="big-val">108.42M</div>
        </div>
        <div class="stat-group">
            <div>BAND:</div>
            <div class="big-val">C-SAR</div>
        </div>
        <div class="graph-container">
            <canvas id="telemetryGraph"></canvas>
        </div>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const container = document.getElementById("canvas-container");
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
    38, container.clientWidth / container.clientHeight, 0.1, 1000
);

// Dynamic Camera Distance function to scale Earth based on screen width
function getResponsiveCameraZ() {
    const w = window.innerWidth;
    if (w < 480) return 5.2;       // Phone screen
    else if (w < 768) return 4.4;  // Tablet screen
    return 3.6;                    // Desktop screen
}

camera.position.set(0, 0, getResponsiveCameraZ());

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

// Earth Mesh Setup
const geometry = new THREE.SphereGeometry(0.95, 96, 96);
const loader = new THREE.TextureLoader();

const earthTexture = loader.load("https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg");
const bumpTexture = loader.load("https://threejs.org/examples/textures/planets/earth_normal_2048.jpg");

const material = new THREE.MeshPhongMaterial({
    map: earthTexture,
    normalMap: bumpTexture,
    specular: new THREE.Color(0x05101a),
    shininess: 8
});

const earth = new THREE.Mesh(geometry, material);
earth.rotation.y = -1.2;
scene.add(earth);

// Tech Grid Atmosphere Overlay
const atmosphereGeo = new THREE.SphereGeometry(0.965, 48, 48);
const atmosphereMat = new THREE.MeshBasicMaterial({
    color: 0x00e5ff,
    wireframe: true,
    transparent: true,
    opacity: 0.08
});
const techGrid = new THREE.Mesh(atmosphereGeo, atmosphereMat);
scene.add(techGrid);

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
dirLight.position.set(5, 3, 5);
scene.add(dirLight);

// Geospatial Orbit Track Lines
function createGeospatialOrbit(radiusX, radiusY, rotX, rotY, rotZ, colorHex) {
    const curve = new THREE.EllipseCurve(0, 0, radiusX, radiusY, 0, Math.PI * 2, false, 0);
    const points = curve.getPoints(150);
    const arcGeometry = new THREE.BufferGeometry().setFromPoints(
        points.map(p => new THREE.Vector3(p.x, p.y, 0))
    );
    const arcMaterial = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.65 });
    const arc = new THREE.Line(arcGeometry, arcMaterial);
    arc.rotation.x = rotX;
    arc.rotation.y = rotY;
    arc.rotation.z = rotZ;
    scene.add(arc);
    return arc;
}

const track1 = createGeospatialOrbit(1.35, 1.35, 1.2, 0.3, -0.2, 0x00e5ff);
const track2 = createGeospatialOrbit(1.45, 1.45, -0.8, 0.6, 0.5, 0xd93838);

// Active Satellite Nodes
const nodeGeo = new THREE.SphereGeometry(0.018, 16, 16);
const satNode1 = new THREE.Mesh(nodeGeo, new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
scene.add(satNode1);

const satNode2 = new THREE.Mesh(nodeGeo, new THREE.MeshBasicMaterial({ color: 0xd93838 }));
scene.add(satNode2);

// Telemetry Canvas Graph
function drawGraph() {
    const graphCanvas = document.getElementById("telemetryGraph");
    if (graphCanvas) {
        const ctx = graphCanvas.getContext("2d");
        graphCanvas.width = graphCanvas.offsetWidth;
        graphCanvas.height = graphCanvas.offsetHeight;

        ctx.strokeStyle = "#112233";
        ctx.lineWidth = 1;
        for (let x = 0; x < graphCanvas.width; x += 15) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, graphCanvas.height); ctx.stroke();
        }

        ctx.strokeStyle = "#00e5ff";
        ctx.beginPath();
        for (let x = 0; x < graphCanvas.width; x += 5) {
            const y = (graphCanvas.height / 2) + Math.sin(x * 0.08) * 6 + (Math.random() - 0.5) * 3;
            if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = "#00e5ff";
        ctx.beginPath();
        ctx.arc(graphCanvas.width * 0.6, graphCanvas.height / 2, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}
drawGraph();

let angle = 0;
function animate() {
    requestAnimationFrame(animate);
    angle += 0.008;

    earth.rotation.y += 0.0006;
    techGrid.rotation.y += 0.0006;

    satNode1.position.x = 1.35 * Math.cos(angle);
    satNode1.position.y = 1.35 * Math.sin(angle);
    satNode1.position.z = 0.2 * Math.sin(angle);

    satNode2.position.x = 1.45 * Math.cos(-angle * 0.8);
    satNode2.position.y = 1.45 * Math.sin(-angle * 0.8);
    satNode2.position.z = 0.4 * Math.cos(-angle * 0.8);

    renderer.render(scene, camera);
}
animate();

// Smooth Responsive Resize Handler
window.addEventListener("resize", function() {
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.position.z = getResponsiveCameraZ();
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
    drawGraph();
});
</script>
</body>
</html>
"""

components.html(
    earth_hud_html,
    height=600
)


# ============================================================
# COORDINATE INPUT & SYSTEM CONTROLS
# ============================================================

st.markdown(
    '<div class="section-title">TARGET LOCATION COORDINATES</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "LATITUDE",
        value=27.25,
        format="%.5f"
    )

with col2:
    longitude = st.number_input(
        "LONGITUDE",
        value=88.50,
        format="%.5f"
    )

analyze = st.button(
    "INITIATE TERRAIN TELEMETRY SCAN"
)


# ============================================================
# ANALYZE LOCATION
# ============================================================

if analyze:
    try:
        with st.spinner("FETCHING SATELLITE TELEMETRY..."):
            response = requests.get(
                "http://127.0.0.1:8000/predict",
                params={"latitude": latitude, "longitude": longitude},
                timeout=30
            )

        if response.status_code == 200:
            st.session_state["result"] = response.json()
            st.session_state["analysis_error"] = None
        else:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", "Backend error.")
            except Exception:
                error_message = response.text

            st.session_state["result"] = None
            st.session_state["analysis_error"] = f"SERVER ERROR ({response.status_code}): {error_message}"

    except requests.exceptions.ConnectionError:
        st.session_state["result"] = None
        st.session_state["analysis_error"] = "BACKEND CONNECTION FAILED."

    except Exception as e:
        st.session_state["result"] = None
        st.session_state["analysis_error"] = f"SYSTEM ERROR: {str(e)}"


# ============================================================
# DISPLAY ERROR
# ============================================================

if st.session_state["analysis_error"]:
    st.error(st.session_state["analysis_error"])


# ============================================================
# RESULTS TELEMETRY
# ============================================================

result = st.session_state["result"]

if result is not None:

    susceptibility = result.get("susceptibility", 0)
    risk = result.get("risk_category", "UNKNOWN")
    feature_data = result.get("features", {})

    st.markdown(
        '<div class="section-title">TELEMETRY ANALYSIS OUTPUT</div>',
        unsafe_allow_html=True
    )

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        with st.container(border=True):
            st.caption("SUSCEPTIBILITY INDEX")
            st.metric(label="", value=f"{susceptibility:.1f}%")

    with risk_col2:
        with st.container(border=True):
            st.caption("THREAT CATEGORY")
            st.metric(label="", value=risk)

    # MAP RADAR DISPLAY
    st.markdown(
        '<div class="section-title">GEOSPATIAL RADAR MATRIX</div>',
        unsafe_allow_html=True
    )

    risk_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        control_scale=True
    )

    try:
        with rasterio.open(SUSCEPTIBILITY_RASTER) as src:
            max_dimension = 1200
            scale_factor = min(1.0, max_dimension / max(src.width, src.height))

            if scale_factor < 1.0:
                out_width = max(1, int(src.width * scale_factor))
                out_height = max(1, int(src.height * scale_factor))
                susceptibility_raster = src.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=rasterio.enums.Resampling.bilinear
                )
            else:
                susceptibility_raster = src.read(1)

            susceptibility_raster = susceptibility_raster.astype(np.float32)
            if src.nodata is not None:
                susceptibility_raster[susceptibility_raster == src.nodata] = np.nan

            susceptibility_raster = np.clip(susceptibility_raster, 0.0, 1.0)
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        valid_mask = np.isfinite(susceptibility_raster)
        rgba_image = np.zeros((susceptibility_raster.shape[0], susceptibility_raster.shape[1], 4), dtype=np.uint8)

        normalized = np.clip(susceptibility_raster, 0.0, 1.0)
        red = np.interp(normalized, [0.0, 0.5, 1.0], [0, 217, 255])
        green = np.interp(normalized, [0.0, 0.5, 1.0], [150, 56, 0])
        blue = np.interp(normalized, [0.0, 0.5, 1.0], [255, 56, 0])

        rgba_image[:, :, 0] = np.nan_to_num(red, nan=0).astype(np.uint8)
        rgba_image[:, :, 1] = np.nan_to_num(green, nan=0).astype(np.uint8)
        rgba_image[:, :, 2] = np.nan_to_num(blue, nan=0).astype(np.uint8)
        rgba_image[:, :, 3] = np.where(valid_mask, 160, 0).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=rgba_image,
            bounds=[[south, west], [north, east]],
            opacity=0.6,
            interactive=True
        ).add_to(risk_map)

    except Exception:
        pass

    folium.Marker(
        location=[latitude, longitude],
        popup=f"TARGET: {latitude:.4f}, {longitude:.4f}",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
    ).add_to(risk_map)

    st_folium(risk_map, width=None, height=480, returned_objects=[])

    # METRICS HUD GRID
    st.markdown(
        '<div class="section-title">ENVIRONMENTAL METRICS</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        with st.container(border=True):
            st.caption("ELEVATION")
            st.metric(label="", value=f'{feature_data.get("elevation", 0):.1f} m')

    with m2:
        with st.container(border=True):
            st.caption("SLOPE")
            st.metric(label="", value=f'{feature_data.get("slope", 0):.1f}°')

    with m3:
        with st.container(border=True):
            st.caption("CURVATURE")
            st.metric(label="", value=f'{feature_data.get("curvature", 0):.4f}')

    with m4:
        with st.container(border=True):
            st.caption("PRECIPITATION")
            st.metric(label="", value=f'{feature_data.get("rainfall_2019", 0):.1f} mm')