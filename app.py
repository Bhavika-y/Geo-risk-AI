import streamlit as st
import streamlit.components.v1 as components
import requests
import folium
import rasterio
import numpy as np

from rasterio.warp import transform_bounds
from streamlit_folium import st_folium
from branca.colormap import LinearColormap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Geo-Risk AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

SUSCEPTIBILITY_RASTER = (
    "outputs/sikkim_landslide_susceptibility.tif"
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "analysis_error" not in st.session_state:
    st.session_state["analysis_error"] = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 50% 10%,
            rgba(30, 80, 120, 0.18),
            transparent 35%
        ),
        radial-gradient(
            circle at 80% 80%,
            rgba(30, 120, 100, 0.10),
            transparent 30%
        ),
        #05080d;

    color: #ffffff;
}


.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* ============================================================
   TITLE
============================================================ */

.main-title {
    text-align: center;

    font-size: 4.5rem;

    font-weight: 800;

    letter-spacing: -3px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #9ddcff,
            #65e5b0
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 0.2rem;
}


.subtitle {
    text-align: center;

    color: #8996aa;

    font-size: 1.1rem;

    letter-spacing: 2px;

    margin-bottom: 1rem;
}


.status {
    text-align: center;

    color: #65e5b0;

    font-size: 0.75rem;

    letter-spacing: 2px;

    margin-bottom: 2rem;
}


/* ============================================================
   SECTION TITLES
============================================================ */

.section-title {
    font-size: 1.1rem;

    font-weight: 700;

    letter-spacing: 1px;

    margin-top: 1rem;

    margin-bottom: 1rem;
}


/* ============================================================
   INPUTS
============================================================ */

label {
    color: #aab5c5 !important;
}


div[data-baseweb="input"] {
    background: #0b111a !important;

    border: 1px solid #202b3a !important;

    border-radius: 10px !important;
}


input {
    color: #ffffff !important;
}


/* ============================================================
   BUTTON
============================================================ */

.stButton > button {
    width: 100%;

    height: 48px;

    border-radius: 10px;

    border: 1px solid rgba(101, 229, 176, 0.4);

    background:
        linear-gradient(
            135deg,
            rgba(101, 229, 176, 0.18),
            rgba(80, 150, 255, 0.18)
        );

    color: #ffffff;

    font-weight: 700;

    letter-spacing: 1px;

    transition: all 0.25s ease;
}


.stButton > button:hover {
    border-color: #65e5b0;

    transform: translateY(-1px);

    background:
        linear-gradient(
            135deg,
            rgba(101, 229, 176, 0.28),
            rgba(80, 150, 255, 0.25)
        );
}


/* ============================================================
   STREAMLIT CARDS
============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 35, 0.88),
            rgba(7, 12, 20, 0.94)
        );

    border: 1px solid rgba(120, 150, 180, 0.12);

    border-radius: 18px;
}


/* ============================================================
   METRIC
============================================================ */

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}


/* ============================================================
   MAP
============================================================ */

.map-container {
    border-radius: 18px;

    overflow: hidden;

    border: 1px solid rgba(120, 150, 180, 0.12);
}


/* ============================================================
   FOOTER
============================================================ */

.footer {
    text-align: center;

    color: #4f5b6c;

    font-size: 0.75rem;

    margin-top: 35px;

    letter-spacing: 1px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Geo-Risk AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'GEOLOGICAL &nbsp; • &nbsp; TERRAIN &nbsp; • &nbsp; '
    'RAINFALL &nbsp; • &nbsp; MACHINE LEARNING'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="status">'
    '● GEO-RISK AI • ANALYSIS SYSTEM ONLINE'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 3D EARTH
# ============================================================

earth_html = """
<!DOCTYPE html>

<html>

<head>

<style>

html, body {

    margin: 0;
    padding: 0;

    width: 100%;
    height: 100%;

    overflow: hidden;

    background: transparent;
}

#earth-container {

    width: 100%;
    height: 100%;

    position: relative;
}

canvas {

    display: block;
}

</style>

</head>


<body>

<div id="earth-container"></div>


<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js">
</script>


<script>

const container =
    document.getElementById("earth-container");


const scene = new THREE.Scene();


const camera =
    new THREE.PerspectiveCamera(
        40,
        container.clientWidth /
        container.clientHeight,
        0.1,
        1000
    );


camera.position.z = 3.2;


const renderer =
    new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });


renderer.setPixelRatio(
    window.devicePixelRatio
);


renderer.setSize(
    container.clientWidth,
    container.clientHeight
);


container.appendChild(renderer.domElement);


// ============================================================
// EARTH
// ============================================================

const geometry =
    new THREE.SphereGeometry(
        1,
        96,
        96
    );


const loader =
    new THREE.TextureLoader();


const earthTexture =
    loader.load(
        "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg"
    );


const normalTexture =
    loader.load(
        "https://threejs.org/examples/textures/planets/earth_normal_2048.jpg"
    );


const specularTexture =
    loader.load(
        "https://threejs.org/examples/textures/planets/earth_specular_2048.jpg"
    );


const material =
    new THREE.MeshPhongMaterial({

        map: earthTexture,

        normalMap:
            normalTexture,

        specularMap:
            specularTexture,

        specular:
            new THREE.Color(
                0x333333
            ),

        shininess: 8
    });


const earth =
    new THREE.Mesh(
        geometry,
        material
    );


scene.add(earth);


// ============================================================
// ATMOSPHERE
// ============================================================

const atmosphereGeometry =
    new THREE.SphereGeometry(
        1.06,
        64,
        64
    );


const atmosphereMaterial =
    new THREE.MeshBasicMaterial({

        color: 0x4da6ff,

        transparent: true,

        opacity: 0.10,

        side:
            THREE.BackSide
    });


const atmosphere =
    new THREE.Mesh(
        atmosphereGeometry,
        atmosphereMaterial
    );


scene.add(atmosphere);


// ============================================================
// LIGHTING
// ============================================================

const ambientLight =
    new THREE.AmbientLight(
        0x667788,
        1.5
    );


scene.add(ambientLight);


const directionalLight =
    new THREE.DirectionalLight(
        0xffffff,
        2.0
    );


directionalLight.position.set(
    5,
    3,
    5
);


scene.add(
    directionalLight
);


// ============================================================
// STARS
// ============================================================

const starGeometry =
    new THREE.BufferGeometry();


const starPositions = [];


for (
    let i = 0;
    i < 1500;
    i++
) {

    const radius =
        5 + Math.random() * 5;


    const theta =
        Math.random() *
        Math.PI *
        2;


    const phi =
        Math.acos(
            2 *
            Math.random() -
            1
        );


    const x =
        radius *
        Math.sin(phi) *
        Math.cos(theta);


    const y =
        radius *
        Math.sin(phi) *
        Math.sin(theta);


    const z =
        radius *
        Math.cos(phi);


    starPositions.push(
        x,
        y,
        z
    );
}


starGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(
        starPositions,
        3
    )
);


const starMaterial =
    new THREE.PointsMaterial({

        color: 0xffffff,

        size: 0.025,

        transparent: true,

        opacity: 0.7
    });


const stars =
    new THREE.Points(
        starGeometry,
        starMaterial
    );


scene.add(stars);


// ============================================================
// ORBIT RINGS
// ============================================================

function createOrbit(
    radius,
    rotation
) {

    const curve =
        new THREE.EllipseCurve(
            0,
                       0,
            radius,
            radius * 0.35,
            0,
            Math.PI * 2,
            false,
            0
        );


    const points =
        curve.getPoints(200);


    const orbitGeometry =
        new THREE.BufferGeometry()
            .setFromPoints(
                points.map(
                    p =>
                        new THREE.Vector3(
                            p.x,
                            0,
                            p.y
                        )
                )
            );


    const orbitMaterial =
        new THREE.LineBasicMaterial({

            color: 0x3e6f87,

            transparent: true,

            opacity: 0.30
        });


    const orbit =
        new THREE.Line(
            orbitGeometry,
            orbitMaterial
        );


    orbit.rotation.x =
        rotation;


    scene.add(orbit);


    return orbit;
}


const orbit1 =
    createOrbit(
        1.35,
        0.8
    );


const orbit2 =
    createOrbit(
        1.5,
        -0.5
    );


// ============================================================
// ANIMATION
// ============================================================

function animate() {

    requestAnimationFrame(
        animate
    );


    earth.rotation.y +=
        0.0018;


    atmosphere.rotation.y +=
        0.0012;


    stars.rotation.y +=
        0.00015;


    orbit1.rotation.z +=
        0.0008;


    orbit2.rotation.z -=
        0.0006;


    renderer.render(
        scene,
        camera
    );
}


animate();


// ============================================================
// RESIZE
// ============================================================

window.addEventListener(
    "resize",
    function() {

        camera.aspect =
            container.clientWidth /
            container.clientHeight;


        camera.updateProjectionMatrix();


        renderer.setSize(
            container.clientWidth,
            container.clientHeight
        );

    }
);

</script>

</body>

</html>
"""


components.html(
    earth_html,
    height=470
)


# ============================================================
# LOCATION INPUT
# ============================================================

st.markdown(
    '<div class="section-title">📍 Select Analysis Location</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    latitude = st.number_input(
        "Latitude",
        value=27.25,
        format="%.5f"
    )


with col2:

    longitude = st.number_input(
        "Longitude",
        value=88.50,
        format="%.5f"
    )


analyze = st.button(
    "ANALYZE LOCATION"
)


# ============================================================
# ANALYZE LOCATION
# ============================================================

if analyze:

    try:

        with st.spinner(
            "Analyzing terrain, rainfall and landslide susceptibility..."
        ):

            response = requests.get(
                "http://127.0.0.1:8000/predict",
                params={
                    "latitude": latitude,
                    "longitude": longitude
                },
                timeout=30
            )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            st.session_state["result"] = response.json()

            st.session_state["analysis_error"] = None


        # ----------------------------------------------------
        # BACKEND ERROR
        # ----------------------------------------------------

        else:

            try:

                error_data = response.json()

                error_message = error_data.get(
                    "detail",
                    "Backend returned an error."
                )

            except Exception:

                error_message = response.text


            st.session_state["result"] = None

            st.session_state["analysis_error"] = (
                f"Backend error ({response.status_code}): "
                f"{error_message}"
            )


    except requests.exceptions.ConnectionError:

        st.session_state["result"] = None

        st.session_state["analysis_error"] = (
            "Could not connect to the Geo-Risk AI backend. "
            "Make sure FastAPI is running."
        )


    except requests.exceptions.Timeout:

        st.session_state["result"] = None

        st.session_state["analysis_error"] = (
            "The backend took too long to respond. "
            "Please try again."
        )


    except Exception as e:

        st.session_state["result"] = None

        st.session_state["analysis_error"] = (
            f"Unexpected error: {str(e)}"
        )


# ============================================================
# DISPLAY ERROR
# ============================================================

if st.session_state["analysis_error"]:

    st.error(
        st.session_state["analysis_error"]
    )


# ============================================================
# RESULTS
# ============================================================

result = st.session_state["result"]


if result is not None:

    susceptibility = result.get(
        "susceptibility",
        0
    )

    risk = result.get(
        "risk_category",
        "UNKNOWN"
    )

    feature_data = result.get(
        "features",
        {}
    )


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    st.markdown(
        "## 📊 Risk Assessment"
    )


    risk_col1, risk_col2 = st.columns(2)


    with risk_col1:

        with st.container(border=True):

            st.markdown(
                "### Susceptibility"
            )

            st.markdown(
                f"# {susceptibility:.1f}%"
            )

            st.caption(
                "Model-estimated landslide susceptibility "
                "for the selected location."
            )


    with risk_col2:

        with st.container(border=True):

            st.markdown(
                "### Risk Level"
            )

            st.markdown(
                f"# {risk}"
            )

            st.caption(
                "Risk category derived from the "
                "model susceptibility score."
            )


    # ========================================================
    # LOCATION RISK MAP
    # ========================================================

    st.markdown(
        "## 🗺️ Geospatial Risk Map"
    )

    st.caption(
        "ML susceptibility surface with the analyzed "
        "location and risk classification."
    )


    # --------------------------------------------------------
    # Create map
    # --------------------------------------------------------

    risk_map = folium.Map(
        location=[
            latitude,
            longitude
        ],

        zoom_start=10,

        tiles="OpenStreetMap",

        control_scale=True
    )


    # ========================================================
    # ML SUSCEPTIBILITY RASTER
    # ========================================================

    try:

        with rasterio.open(
            SUSCEPTIBILITY_RASTER
        ) as src:

            # ------------------------------------------------
            # Read raster at reduced resolution
            # ------------------------------------------------

            max_dimension = 1200

            scale_factor = min(
                1.0,
                max_dimension /
                max(src.width, src.height)
            )


            if scale_factor < 1.0:

                out_width = max(
                    1,
                    int(src.width * scale_factor)
                )

                out_height = max(
                    1,
                    int(src.height * scale_factor)
                )

                susceptibility_raster = src.read(
                    1,
                    out_shape=(
                        out_height,
                        out_width
                    ),

                    resampling=rasterio.enums.Resampling.bilinear
                )

            else:

                susceptibility_raster = src.read(1)


            # ------------------------------------------------
            # Handle nodata
            # ------------------------------------------------

            susceptibility_raster = (
                susceptibility_raster.astype(
                    np.float32
                )
            )


            if src.nodata is not None:

                susceptibility_raster[
                    susceptibility_raster ==
                    src.nodata
                ] = np.nan


            # ------------------------------------------------
            # Clip to probability range
            # ------------------------------------------------

            susceptibility_raster = np.clip(
                susceptibility_raster,
                0.0,
                1.0
            )


            # ------------------------------------------------
            # Convert raster bounds to WGS84
            # ------------------------------------------------

            west, south, east, north = (
                transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    *src.bounds
                )
            )


        # ----------------------------------------------------
        # Convert susceptibility to RGBA
        # ----------------------------------------------------

        valid_mask = np.isfinite(
            susceptibility_raster
        )


        # Green → Yellow → Orange → Red
        cmap = LinearColormap(
            colors=[
                "green",
                "yellow",
                "orange",
                "red"
            ],

            vmin=0.0,

            vmax=1.0
        )


        rgba_image = np.zeros(
            (
                susceptibility_raster.shape[0],
                susceptibility_raster.shape[1],
                4
            ),

            dtype=np.uint8
        )


        for value in [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0
        ]:

            pass


        # ----------------------------------------------------
        # Efficient color conversion
        # ----------------------------------------------------

        normalized = np.clip(
            susceptibility_raster,
            0.0,
            1.0
        )


        # Green → Yellow → Orange → Red
        red = np.interp(
            normalized,
            [0.0, 0.25, 0.50, 0.75, 1.0],
            [0, 255, 255, 255, 255]
        )


        green = np.interp(
            normalized,
            [0.0, 0.25, 0.50, 0.75, 1.0],
            [128, 255, 165, 80, 0]
        )


        blue = np.interp(
            normalized,
            [0.0, 0.25, 0.50, 0.75, 1.0],
            [0, 0, 0, 0, 0]
        )


        rgba_image[:, :, 0] = (
            np.nan_to_num(
                red,
                nan=0
            ).astype(np.uint8)
        )


        rgba_image[:, :, 1] = (
            np.nan_to_num(
                green,
                nan=0
            ).astype(np.uint8)
        )


        rgba_image[:, :, 2] = (
            np.nan_to_num(
                blue,
                nan=0
            ).astype(np.uint8)
        )


        # ----------------------------------------------------
        # Transparency
        # ----------------------------------------------------

        rgba_image[:, :, 3] = np.where(
            valid_mask,
            165,
            0
        ).astype(np.uint8)


        # ----------------------------------------------------
        # Add raster overlay
        # ----------------------------------------------------

        folium.raster_layers.ImageOverlay(
            image=rgba_image,

            bounds=[
                [south, west],
                [north, east]
            ],

            opacity=0.65,

            interactive=True,

            cross_origin=False,

            zindex=1,

            name="ML Landslide Susceptibility"

        ).add_to(risk_map)


        # ----------------------------------------------------
        # Susceptibility legend
        # ----------------------------------------------------

        susceptibility_legend = LinearColormap(
            colors=[
                "green",
                "yellow",
                "orange",
                "red"
            ],

            vmin=0.0,

            vmax=1.0,

            caption="ML Landslide Susceptibility"
        )


        susceptibility_legend.add_to(
            risk_map
        )


    except FileNotFoundError:

        st.warning(
            "The susceptibility raster could not be found. "
            "The map will still display the selected location."
        )


    except Exception as raster_error:

        st.warning(
            "Susceptibility overlay could not be loaded: "
            f"{str(raster_error)}"
        )


    # ========================================================
    # RISK COLORS
    # ========================================================

    risk_colors = {
        "LOW": "green",
        "MODERATE": "orange",
        "HIGH": "red",
        "VERY HIGH": "darkred"
    }


    marker_color = risk_colors.get(
        risk.upper(),
        "blue"
    )


    # ========================================================
    # POPUP
    # ========================================================

    popup_html = f"""
    <div style="
        font-family: Arial;
        width: 230px;
        color: #111111;
    ">

        <h3 style="margin-bottom: 10px;">
            🌍 Geo-Risk AI
        </h3>

        <b>Latitude:</b>
        {latitude:.5f}

        <br>

        <b>Longitude:</b>
        {longitude:.5f}

        <hr>

        <b>Susceptibility:</b>
        {susceptibility:.1f}%

        <br>

        <b>Risk Level:</b>
        {risk}

    </div>
    """


    # ========================================================
    # SELECTED LOCATION MARKER
    # ========================================================

    folium.Marker(
        location=[
            latitude,
            longitude
        ],

        popup=folium.Popup(
            popup_html,
            max_width=300
        ),

        tooltip=(
            f"Geo-Risk AI • {risk}"
        ),

        icon=folium.Icon(
            color=marker_color,
            icon="warning-sign"
        )

    ).add_to(risk_map)


    # ========================================================
    # ANALYSIS RADIUS
    # ========================================================

    folium.Circle(
        location=[
            latitude,
            longitude
        ],

        radius=3000,

        popup="3 km analysis reference area",

        color=marker_color,

        fill=True,

        fill_opacity=0.12

    ).add_to(risk_map)


    # ========================================================
    # LAYER CONTROL
    # ========================================================

    folium.LayerControl().add_to(
        risk_map
    )


    # ========================================================
    # RENDER MAP
    # ========================================================

    st_folium(
        risk_map,
        width=None,
        height=600,
        returned_objects=[]
    )


    # ========================================================
    # TERRAIN & ENVIRONMENT
    # ========================================================

    st.markdown(
        "## ⛰️ Terrain & Environmental Intelligence"
    )


    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        with st.container(border=True):

            st.caption(
                "Elevation"
            )

            st.metric(
                label="",
                value=(
                    f'{feature_data.get("elevation", 0):.2f} m'
                )
            )


    with metric2:

        with st.container(border=True):

            st.caption(
                "Slope"
            )

            st.metric(
                label="",
                value=(
                    f'{feature_data.get("slope", 0):.2f}°'
                )
            )


    with metric3:

        with st.container(border=True):

            st.caption(
                "Aspect"
            )

            st.metric(
                label="",
                value=(
                    f'{feature_data.get("aspect", 0):.2f}°'
                )
            )


    metric4, metric5 = st.columns(2)


    with metric4:

        with st.container(border=True):

            st.caption(
                "Curvature"
            )

            st.metric(
                label="",
                value=(
                    f'{feature_data.get("curvature", 0):.4f}'
                )
            )


    with metric5:

        with st.container(border=True):

            st.caption(
                "Rainfall"
            )

            st.metric(
                label="",
                value=(
                    f'{feature_data.get("rainfall_2019", 0):.2f} mm'
                )
            )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.markdown(
        "## 🧠 Explainable AI"
    )


    st.markdown(
        f"### Why is this location classified as {risk} risk?"
    )


    st.caption(
        "SHAP explains how individual features influence "
        "the Random Forest prediction."
    )


    explanations = result.get(
        "explanation",
        []
    )


    if explanations:

        for item in explanations:

            feature = item.get(
                "feature",
                "Unknown"
            )

            shap_value = float(
                item.get(
                    "shap_value",
                    0
                )
            )

            effect = item.get(
                "effect",
                ""
            )


            if shap_value >= 0:

                arrow = "↗"

                sign = "+"


            else:

                arrow = "↘"

                sign = ""


            with st.container(border=True):

                st.markdown(
                    f"**{arrow} {feature}**"
                )

                st.write(
                    f"{effect} "
                    f"({sign}{shap_value:.4f})"
                )


    else:

        st.info(
            "SHAP explanation is not available "
            "for this prediction."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        GEO-RISK AI &nbsp; • &nbsp;
        MACHINE LEARNING &nbsp; • &nbsp;
        GEOSPATIAL INTELLIGENCE
    </div>
    """,
    unsafe_allow_html=True
)