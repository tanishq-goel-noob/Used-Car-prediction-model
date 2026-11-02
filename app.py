"""
Used Car Price Prediction System
--------------------------------
A Streamlit web app that uses an already-trained scikit-learn Pipeline
(Gradient Boosting Regressor + One-Hot Encoding) to predict the price of
a used Ford car in British Pounds (GBP).

"""

# ----------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------
from datetime import datetime
from pathlib import Path

import joblib                # loads the saved .pkl model
import pandas as pd          # DataFrame for dataset + model input
import plotly.express as px  # interactive charts for the dashboard
import sklearn               # only used to show the version in the sidebar
import streamlit as st       # the web-app framework


# ----------------------------------------------------------------------
# 2. PAGE SETTINGS  (must be the first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
)


# ----------------------------------------------------------------------
# 3. CONSTANTS
# ----------------------------------------------------------------------
# Path of files, relative to THIS file (so it works locally AND on cloud,
# no matter from which folder the app is started).
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "ford.csv"
MODEL_PATH = BASE_DIR / "ford_price_prediction_model.pkl"

# Put your own details here (shown in the sidebar).
STUDENT_NAME = "Tanishq Goel"
UNIVERSITY = "Manipal University Jaipur"

# ---- RESULTS TABLE (copied from the notebook output, Test set) -------------
# One row = one experiment. These values are NOT calculated again here.
# Source in notebook:  Encoding comparison -> results table
#                      Feature Selection   -> SelectKBest(f_regression, k=10)
# Feature Selection was done only with One-Hot Encoding.
WITHOUT_FS = "Without Selection (all features)"
WITH_FS = "With SelectKBest (k=10)"

ALL_EXPERIMENTS = pd.DataFrame(
    [
        # Encoding,           Feature Selection, Model,               MAE,     RMSE,    R2,     Adj R2
        ("One-Hot Encoding", WITHOUT_FS, "Linear Regression", 1374.81, 1858.76, 0.8458, 0.8442),
        ("One-Hot Encoding", WITHOUT_FS, "Ridge Regression",  1374.30, 1856.37, 0.8462, 0.8446),
        ("One-Hot Encoding", WITHOUT_FS, "Gradient Boosting",  962.58, 1364.01, 0.9170, 0.9161),
        ("Label Encoding",   WITHOUT_FS, "Linear Regression", 1776.83, 2429.73, 0.7366, 0.7360),
        ("Label Encoding",   WITHOUT_FS, "Ridge Regression",  1776.82, 2429.72, 0.7366, 0.7360),
        ("Label Encoding",   WITHOUT_FS, "Gradient Boosting",  964.15, 1378.44, 0.9152, 0.9150),
        ("One-Hot Encoding", WITH_FS,    "Linear Regression", 1707.98, 2368.72, 0.7497, 0.7490),
        ("One-Hot Encoding", WITH_FS,    "Ridge Regression",  1707.70, 2368.60, 0.7497, 0.7490),
        ("One-Hot Encoding", WITH_FS,    "Gradient Boosting", 1025.53, 1474.12, 0.9030, 0.9028),
    ],
    columns=["Encoding", "Feature Selection", "Model", "MAE", "RMSE", "R² Score", "Adjusted R²"],
)

# Small table used in the "Model Information" tab:
# One-Hot Encoding, without Feature Selection (the saved model's setup).
_saved_setup = ALL_EXPERIMENTS[
    (ALL_EXPERIMENTS["Encoding"] == "One-Hot Encoding")
    & (ALL_EXPERIMENTS["Feature Selection"] == WITHOUT_FS)
]
MODEL_RESULTS = _saved_setup[["Model", "MAE", "RMSE", "R² Score", "Adjusted R²"]].rename(
    columns={"MAE": "MAE (£)", "RMSE": "RMSE (£)"}
).reset_index(drop=True)

# Best experiment = highest R² score of all experiments.
BEST_ROW = ALL_EXPERIMENTS.loc[ALL_EXPERIMENTS["R² Score"].idxmax()]
BEST_MAE = float(BEST_ROW["MAE"])      # average error of the best model (in £)
BEST_RMSE = float(BEST_ROW["RMSE"])
BEST_R2 = float(BEST_ROW["R² Score"])  # R² of the best model

# Colours used in the charts (same blue as the app theme)
ENCODING_COLORS = {"One-Hot Encoding": "#1d4ed8", "Label Encoding": "#f59e0b"}
FS_COLORS = {WITHOUT_FS: "#1d4ed8", WITH_FS: "#ef4444"}

# If the dataset has fewer rows than this for a category, we show a warning.
RARE_CATEGORY_LIMIT = 30


# ----------------------------------------------------------------------
# 4. CUSTOM CSS  (light, clean, automotive look)
# ----------------------------------------------------------------------
def load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }

        /* Hero banner */
        .hero {
            background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
            border-radius: 18px; padding: 2.2rem 2.4rem; color: #ffffff;
            margin-bottom: 1.2rem;
        }
        .hero h1 { color: #ffffff; font-size: 2.3rem; font-weight: 800; margin: 0 0 .4rem 0; }
        .hero p  { color: #dbeafe; font-size: 1.05rem; margin: 0; line-height: 1.6; }
        .hero .tag {
            display: inline-block; background: rgba(255,255,255,.15); color: #ffffff;
            border-radius: 999px; padding: .2rem .8rem; font-size: .8rem;
            margin: 0 .4rem .8rem 0; font-weight: 600;
        }

        /* Small stat cards under the hero */
        .stat {
            background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
            padding: 1rem 1.2rem; text-align: center;
        }
        .stat .value { font-size: 1.5rem; font-weight: 700; color: #1d4ed8; }
        .stat .label { font-size: .85rem; color: #64748b; }

        /* Result card */
        .result-card {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 1px solid #6ee7b7; border-radius: 18px;
            padding: 1.8rem; text-align: center; margin-top: .5rem;
        }
        .result-card .small { color: #047857; font-weight: 600; letter-spacing: .06em; font-size: .85rem; }
        .result-card .price { font-size: 3rem; font-weight: 800; color: #065f46; margin: .2rem 0; }
        .result-card .range { color: #047857; font-size: 1rem; }

        /* Small insight box under charts */
        .insight {
            background: #eff6ff; border-left: 4px solid #1d4ed8; border-radius: 8px;
            padding: .8rem 1rem; color: #1e293b; font-size: .95rem; margin-top: .5rem;
        }

        /* Section headings */
        h2, h3 { color: #0f172a; font-weight: 700; }

        /* Predict button */
        div.stFormSubmitButton > button {
            background: #1d4ed8; color: #ffffff; border: none; border-radius: 10px;
            padding: .6rem 1.6rem; font-weight: 600; font-size: 1rem;
        }
        div.stFormSubmitButton > button:hover { background: #1e40af; color: #ffffff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# 5. LOADING THE DATA AND THE MODEL  (cached = loaded only once)
# ----------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Read ford.csv. Used only to fill the dropdowns / slider ranges."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    """Load the saved scikit-learn Pipeline (preprocessing + model together)."""
    return joblib.load(MODEL_PATH)


def build_input_options(df: pd.DataFrame) -> dict:
    """
    Read the dataset and prepare everything the input widgets need:
    dropdown lists, slider ranges and default values.
    """
    # IMPORTANT: in ford.csv the car-model names have a leading space
    # (e.g. ' Fiesta'). The model was trained on those exact strings, so we
    # must send them back unchanged. We show clean names to the user
    # ("Fiesta") and keep a dictionary that maps clean name -> original name.
    clean_to_raw = (
        df.groupby(df["model"].str.strip())["model"]
        .agg(lambda s: s.value_counts().index[0])   # most common spelling
        .to_dict()
    )

    # The dataset has one wrong row with year = 2060, so ignore future years.
    valid_years = df.loc[df["year"] <= datetime.now().year, "year"]

    return {
        "clean_to_raw": clean_to_raw,
        "models": sorted(clean_to_raw.keys()),
        "transmissions": sorted(df["transmission"].unique()),
        "fuel_types": sorted(df["fuelType"].unique()),
        "engine_sizes": sorted(s for s in df["engineSize"].unique() if s > 0),
        "year_min": int(valid_years.min()),
        "year_max": int(valid_years.max()),
        "mileage_max": int(df["mileage"].max()),
        "tax_max": int(df["tax"].max()),
        "mpg_min": float(df["mpg"].min()),
        "mpg_max": float(df["mpg"].max()),
        # Defaults = most common / median values of the dataset
        "default_model": df["model"].str.strip().mode()[0],
        "default_transmission": df["transmission"].mode()[0],
        "default_fuel": df["fuelType"].mode()[0],
        "default_year": int(valid_years.median()),
        "default_mileage": int(df["mileage"].median()),
        "default_tax": int(df["tax"].median()),
        "default_mpg": round(float(df["mpg"].median()), 1),
        "default_engine": float(df["engineSize"].median()),
    }


# ----------------------------------------------------------------------
# 6. PREDICTION LOGIC
# ----------------------------------------------------------------------
def make_input_dataframe(model, values: dict) -> pd.DataFrame:
    """
    Convert the user's inputs into a one-row DataFrame.
    Column names are taken from the model itself (model.feature_names_in_)
    so they always match what the Pipeline expects.
    """
    return pd.DataFrame([values])[list(model.feature_names_in_)]


def predict_price(model, input_df: pd.DataFrame) -> float:
    """The Pipeline does imputing + scaling + encoding + prediction by itself."""
    return float(model.predict(input_df)[0])


# ----------------------------------------------------------------------
# 7. UI SECTIONS
# ----------------------------------------------------------------------
def show_hero(df: pd.DataFrame) -> None:
    """Home section: title, short description and 3 quick stats."""
    st.markdown(
        """
        <div class="hero">
            <span class="tag">🤖 Machine Learning</span>
            <span class="tag">📈 Predictive Modelling</span>
            <span class="tag">£ GBP</span>
            <h1>🚗 Used Car Price Prediction</h1>
            <p>Enter the details of a used Ford car and our trained Machine Learning model
            will estimate its market price in British Pounds. The model learned the
            relationship between a car's features and its price from thousands of real
            used-car listings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    stats = [
        (f"{len(df):,}", "Cars in dataset"),
        ("Gradient Boosting", "Best model"),
        (f"{BEST_R2:.3f}", "R² score (test data)"),
    ]
    for col, (value, label) in zip((c1, c2, c3), stats):
        col.markdown(
            f'<div class="stat"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def show_sidebar(model) -> None:
    """Sidebar with project information."""
    with st.sidebar:
        st.title("🚗 Project Info")
        st.markdown(
            f"""
            **Used Car Price Prediction System**

            Predictive Modelling project  
            👤 {STUDENT_NAME}  
            🎓 {UNIVERSITY}
            """
        )
        st.divider()
        st.markdown(
            """
            **Dataset:** Ford Used Car Dataset (`ford.csv`)  
            **Target:** `price` (£)  
            **Model:** Gradient Boosting Regressor  
            **Encoding:** One-Hot Encoding
            """
        )
        st.divider()
        st.markdown("**How to use**")
        st.markdown(
            "1. Fill in the car details  \n"
            "2. Click **Predict Car Price**  \n"
            "3. Read the estimated price"
        )
        with st.expander("Technical info"):
            st.write(f"scikit-learn: `{sklearn.__version__}`")
            st.write("Pipeline steps:")
            st.write([f"{name} → {type(step).__name__}" for name, step in model.steps])


def show_prediction_form(model, df: pd.DataFrame, opt: dict) -> None:
    """Input widgets + Predict button + result card."""
    st.subheader("Enter Car Details")

    # st.form groups all inputs; the app re-runs only when the button is clicked.
    with st.form("car_form"):
        left, right = st.columns(2, gap="large")

        with left:
            car_model = st.selectbox(
                "Car Model", opt["models"], index=opt["models"].index(opt["default_model"])
            )
            transmission = st.selectbox(
                "Transmission",
                opt["transmissions"],
                index=opt["transmissions"].index(opt["default_transmission"]),
            )
            fuel_type = st.selectbox(
                "Fuel Type",
                opt["fuel_types"],
                index=opt["fuel_types"].index(opt["default_fuel"]),
            )
            year = st.slider(
                "Manufacturing Year",
                min_value=opt["year_min"],
                max_value=opt["year_max"],
                value=opt["default_year"],
            )

        with right:
            mileage = st.number_input(
                "Mileage (miles driven)",
                min_value=0,
                max_value=opt["mileage_max"],
                value=opt["default_mileage"],
                step=500,
            )
            tax = st.number_input(
                "Road Tax (£ per year)",
                min_value=0,
                max_value=opt["tax_max"],
                value=opt["default_tax"],
                step=5,
            )
            mpg = st.number_input(
                "Fuel Efficiency (MPG)",
                min_value=opt["mpg_min"],
                max_value=opt["mpg_max"],
                value=opt["default_mpg"],
                step=0.1,
            )
            engine_size = st.select_slider(
                "Engine Size (litres)",
                options=opt["engine_sizes"],
                value=opt["default_engine"] if opt["default_engine"] in opt["engine_sizes"] else opt["engine_sizes"][0],
            )

        submitted = st.form_submit_button("Predict Car Price")

    if not submitted:
        return

    # --- Build the model input -----------------------------------------
    # Names of the keys MUST match the training columns exactly.
    user_values = {
        "model": opt["clean_to_raw"][car_model],   # original name (with leading space)
        "year": int(year),
        "transmission": transmission,
        "mileage": int(mileage),
        "fuelType": fuel_type,
        "tax": int(tax),
        "mpg": float(mpg),
        "engineSize": float(engine_size),
    }
    input_df = make_input_dataframe(model, user_values)

    # --- Predict ---------------------------------------------------------
    price = predict_price(model, input_df)
    price = max(price, 0.0)                       # a price can't be negative
    low, high = max(price - BEST_MAE, 0.0), price + BEST_MAE

    # --- Show the result card ---------------------------------------------
    st.markdown(
        f"""
        <div class="result-card">
            <div class="small">ESTIMATED PRICE</div>
            <div class="price">£{price:,.0f}</div>
            <div class="range">Likely range: £{low:,.0f} – £{high:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "ℹ️ This is only an **estimate** made by a Machine Learning model. "
        f"On unseen test data the model's average error was about £{BEST_MAE:,.0f}. "
        "The real price can differ because of condition, location, car history, etc."
    )

    # --- Honest warnings for unusual inputs -----------------------------------
    fuel_count = int((df["fuelType"] == fuel_type).sum())
    if fuel_count < RARE_CATEGORY_LIMIT:
        st.warning(
            f"⚠️ Only {fuel_count} '{fuel_type}' car(s) exist in the dataset, "
            "so this prediction may be less reliable."
        )

    with st.expander("Show the data sent to the model"):
        st.dataframe(input_df, hide_index=True)


def show_model_info() -> None:
    """Simple explanation section (good for viva preparation)."""
    st.subheader("About the Model")

    st.markdown("#### 1. What is Predictive Modelling?")
    st.write(
        "Predictive Modelling means using **past data** to build a mathematical model "
        "that can **predict an unknown value** for new data. Here, we use data of old "
        "used-car sales to predict the price of a car the model has never seen."
    )

    st.markdown("#### 2. Dataset and Target Variable")
    st.write(
        "- **Dataset:** Ford Used Car Dataset (`ford.csv`) – 17,966 cars, 9 columns, no missing values.\n"
        "- **Input features (8):** model, year, transmission, mileage, fuelType, tax, mpg, engineSize.\n"
        "- **Target variable:** `price` (in British Pounds £) – this is what we predict.\n"
        "- **Problem type:** Regression (because the output is a continuous number)."
    )

    st.markdown("#### 3. How the data is processed (Pipeline)")
    st.write(
        "The data was split **80% training / 20% testing**. A scikit-learn **Pipeline** "
        "does all the steps automatically, in the same order, every time:\n"
        "- **Numerical columns** (year, mileage, tax, mpg, engineSize): "
        "missing values filled with median → Standard Scaling.\n"
        "- **Categorical columns** (model, transmission, fuelType): "
        "missing values filled with most frequent value → One-Hot Encoding.\n"
        "- Then the **Gradient Boosting Regressor** makes the prediction."
    )

    st.markdown("#### 4. Models evaluated")
    st.write("Three models were trained and compared (results on the test data):")
    st.dataframe(MODEL_RESULTS, hide_index=True)
    st.bar_chart(MODEL_RESULTS.set_index("Model")["R² Score"])

    st.markdown("#### 5. Best model")
    st.success(
        "🏆 **Gradient Boosting Regressor with One-Hot Encoding (without Feature Selection)** "
        "performed best with R² ≈ 0.917, MAE ≈ £963 and RMSE ≈ £1,364."
    )
    st.write(
        "Other experiments in the notebook: Label (Ordinal) Encoding gave a slightly lower "
        "score for Gradient Boosting (R² ≈ 0.915), and Feature Selection with SelectKBest "
        "also lowered it (R² ≈ 0.903). So the final saved model uses One-Hot Encoding "
        "without Feature Selection. Gradient Boosting builds many small decision trees one "
        "after another, where each new tree corrects the mistakes of the previous ones "
        "(this model: 200 trees, learning rate 0.05)."
    )

    st.markdown("#### 6. What do the metrics mean?")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("R² Score", f"{BEST_R2:.3f}")
        st.caption(
            "Shows how much of the variation in price the model can explain. "
            "1.0 = perfect, 0 = useless. 0.917 means about 91.7% is explained."
        )
    with m2:
        st.metric("MAE", f"£{BEST_MAE:,.0f}")
        st.caption(
            "Mean Absolute Error: the average difference between the actual price "
            "and the predicted price. Lower is better."
        )
    with m3:
        st.metric("RMSE", "£1,364")
        st.caption(
            "Root Mean Squared Error: like MAE but it punishes big mistakes more. "
            "Lower is better."
        )

    st.markdown("#### 7. Limitations")
    st.write(
        "- The model only knows used **Ford** cars from this dataset (mostly 2013–2020), prices in £.\n"
        "- It does not know the car's condition, accident history, colour or location.\n"
        "- Rare categories (e.g. Electric, Other fuel) have very few examples, so results are less reliable."
    )


# ----------------------------------------------------------------------
# 7b. MODEL PERFORMANCE DASHBOARD  (new tab)
# ----------------------------------------------------------------------
# Hover information shown on every chart (all 4 metrics of that bar)
HOVER_INFO = {
    "MAE": ":,.2f",
    "RMSE": ":,.2f",
    "R² Score": ":.4f",
    "Adjusted R²": ":.4f",
}


def make_bar_chart(data, y, color, color_map, title, y_title, text_format, y_range=None):
    """
    Build one grouped bar chart with Plotly Express.
      x     = model name
      y     = the metric we want to compare (R², MAE, ...)
      color = the group that makes the legend (Encoding or Feature Selection)
    """
    fig = px.bar(
        data,
        x="Model",
        y=y,
        color=color,
        barmode="group",                 # bars side by side
        text=y,                          # show the value on top of each bar
        color_discrete_map=color_map,
        hover_data=HOVER_INFO,           # extra info when mouse is over a bar
        title=title,
        template="plotly_white",         # clean light theme
    )
    fig.update_traces(texttemplate=text_format, textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=380,
        yaxis_title=y_title,
        xaxis_title=None,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(t=60, b=40, l=10, r=10),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    if y_range:
        fig.update_yaxes(range=y_range)
    else:
        fig.update_yaxes(rangemode="tozero")
    return fig


def show_metric_cards() -> None:
    """Four summary cards for the best model (re-uses the .stat CSS class)."""
    cards = [
        (BEST_ROW["Model"], "Best Model (One-Hot, no Feature Selection)"),
        (f"{BEST_R2:.4f}", "Best R² Score"),
        (f"£{BEST_MAE:,.2f}", "MAE (average error)"),
        (f"£{BEST_RMSE:,.2f}", "RMSE"),
    ]
    for col, (value, label) in zip(st.columns(4), cards):
        col.markdown(
            f'<div class="stat"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def show_dashboard() -> None:
    """Model Performance Dashboard: metric cards + 6 graphs + result tables."""
    st.subheader("Model Performance Dashboard")
    st.caption(
        "All values come from the notebook (evaluated on the 20% test data). "
        "Nothing is re-trained inside this app."
    )

    # Rows without feature selection = encoding comparison (3 models x 2 encodings)
    encoding_df = ALL_EXPERIMENTS[ALL_EXPERIMENTS["Feature Selection"] == WITHOUT_FS]
    # Rows with One-Hot Encoding = feature selection comparison (3 models x 2 settings)
    fs_df = ALL_EXPERIMENTS[ALL_EXPERIMENTS["Encoding"] == "One-Hot Encoding"]

    # ---- Summary cards ---------------------------------------------------
    show_metric_cards()
    st.write("")

    # ---- Graphs 1-4: model comparison across encodings ----------------------
    st.markdown("#### Model comparison: One-Hot vs Label Encoding")
    g1, g2 = st.columns(2)
    with g1:   # Graph 1
        st.plotly_chart(make_bar_chart(
            encoding_df, "R² Score", "Encoding", ENCODING_COLORS,
            "R² Score Comparison (higher is better)", "R² Score", "%{text:.4f}", y_range=[0, 1.05],
        ))
    with g2:   # Graph 2
        st.plotly_chart(make_bar_chart(
            encoding_df, "MAE", "Encoding", ENCODING_COLORS,
            "MAE Comparison (lower is better)", "MAE (£)", "£%{text:,.0f}",
        ))
    g3, g4 = st.columns(2)
    with g3:   # Graph 3
        st.plotly_chart(make_bar_chart(
            encoding_df, "RMSE", "Encoding", ENCODING_COLORS,
            "RMSE Comparison (lower is better)", "RMSE (£)", "£%{text:,.0f}",
        ))
    with g4:   # Graph 4
        st.plotly_chart(make_bar_chart(
            encoding_df, "Adjusted R²", "Encoding", ENCODING_COLORS,
            "Adjusted R² Comparison (higher is better)", "Adjusted R²", "%{text:.4f}", y_range=[0, 1.05],
        ))

    # ---- Graph 5: encoding technique comparison -----------------------------
    st.markdown("#### Encoding technique comparison")
    # One row per model with One-Hot and Label values side by side
    pivot = encoding_df.pivot(index="Model", columns="Encoding", values=["R² Score", "MAE"])
    gain = pd.DataFrame(
        {
            "Model": pivot.index,
            "R² gain": (pivot[("R² Score", "One-Hot Encoding")] - pivot[("R² Score", "Label Encoding")]).values,
            "R² One-Hot": pivot[("R² Score", "One-Hot Encoding")].values,
            "R² Label": pivot[("R² Score", "Label Encoding")].values,
            "MAE One-Hot (£)": pivot[("MAE", "One-Hot Encoding")].values,
            "MAE Label (£)": pivot[("MAE", "Label Encoding")].values,
        }
    )
    fig5 = px.bar(
        gain, x="Model", y="R² gain", text="R² gain",
        title="R² Gain of One-Hot Encoding over Label Encoding",
        hover_data={"R² gain": ":.4f", "R² One-Hot": ":.4f", "R² Label": ":.4f",
                    "MAE One-Hot (£)": ":,.2f", "MAE Label (£)": ":,.2f"},
        template="plotly_white",
    )
    fig5.update_traces(marker_color="#1d4ed8", texttemplate="%{text:+.4f}",
                       textposition="outside", cliponaxis=False)
    fig5.update_layout(height=380, yaxis_title="R² (One-Hot − Label)", xaxis_title=None,
                       margin=dict(t=60, b=40, l=10, r=10),
                       font=dict(family="Inter, Segoe UI, sans-serif"))
    c5, t5 = st.columns([2, 1], gap="large")
    with c5:
        st.plotly_chart(fig5)
    with t5:
        lr = gain.set_index("Model")
        st.markdown(
            f"""
            <div class="insight">
            <b>How to read this graph</b><br>
            A taller bar means One-Hot Encoding gave a better R² than Label Encoding.<br><br>
            • Linear Regression: <b>{lr.loc['Linear Regression', 'R² gain']:+.4f}</b><br>
            • Ridge Regression: <b>{lr.loc['Ridge Regression', 'R² gain']:+.4f}</b><br>
            • Gradient Boosting: <b>{lr.loc['Gradient Boosting', 'R² gain']:+.4f}</b><br><br>
            Linear models gain a lot from One-Hot Encoding, because Label Encoding gives
            car models an artificial order (0, 1, 2 ...). Gradient Boosting (tree based)
            is hardly affected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Graph 6: feature selection impact (One-Hot only) ---------------------
    st.markdown("#### Feature selection impact (One-Hot Encoding, SelectKBest with f_regression, k = 10)")
    f1, f2 = st.columns(2)
    with f1:
        st.plotly_chart(make_bar_chart(
            fs_df, "R² Score", "Feature Selection", FS_COLORS,
            "Feature Selection Impact: R² Score", "R² Score", "%{text:.4f}", y_range=[0, 1.05],
        ))
    with f2:
        st.plotly_chart(make_bar_chart(
            fs_df, "MAE", "Feature Selection", FS_COLORS,
            "Feature Selection Impact: MAE", "MAE (£)", "£%{text:,.0f}",
        ))
    gb = fs_df[fs_df["Model"] == "Gradient Boosting"].set_index("Feature Selection")
    st.markdown(
        f"""
        <div class="insight">
        <b>Conclusion:</b> Keeping only 10 features lowered the performance of every model.
        For Gradient Boosting, R² dropped from <b>{gb.loc[WITHOUT_FS, 'R² Score']:.4f}</b> to
        <b>{gb.loc[WITH_FS, 'R² Score']:.4f}</b> and MAE rose from
        <b>£{gb.loc[WITHOUT_FS, 'MAE']:,.2f}</b> to <b>£{gb.loc[WITH_FS, 'MAE']:,.2f}</b>.
        That is why the final saved model uses <b>all features</b> (no Feature Selection).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Full results table -----------------------------------------------------
    with st.expander("Show full results table (all experiments)"):
        st.dataframe(
            ALL_EXPERIMENTS.sort_values("R² Score", ascending=False),
            hide_index=True,
        )


# ----------------------------------------------------------------------
# 8. MAIN FUNCTION  (the app starts here)
# ----------------------------------------------------------------------
def main() -> None:
    load_css()

    # Load files, show a friendly error if something is wrong.
    try:
        df = load_data()
        model = load_model()
    except FileNotFoundError as err:
        st.error(f"File not found: {err}. Keep app.py, ford.csv and the .pkl file in the same folder.")
        st.stop()
    except Exception as err:  # e.g. scikit-learn version mismatch
        st.error(
            "Could not load the saved model. Install the exact scikit-learn version "
            "from requirements.txt (scikit-learn==1.9.0).\n\n"
            f"Details: {err}"
        )
        st.stop()

    options = build_input_options(df)

    show_sidebar(model)
    show_hero(df)
    st.write("")

    tab_predict, tab_info, tab_dashboard = st.tabs(
        ["🚗 Predict Price", "📊 Model Information", "📈 Model Performance Dashboard"]
    )
    with tab_predict:
        show_prediction_form(model, df, options)
    with tab_info:
        show_model_info()
    with tab_dashboard:
        show_dashboard()

    st.caption("Built with Streamlit & scikit-learn · Prices are in British Pounds (£) · Educational project")


if __name__ == "__main__":
    main()