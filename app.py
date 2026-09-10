import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="StudentIQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #172554 0%, #1d4ed8 100%);
        color: white;
        margin-bottom: 1.8rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0.7rem;
    }

    .hero-description {
        font-size: 0.95rem;
        opacity: 0.82;
        max-width: 900px;
    }

    /* Section headings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Insight cards */
    .insight {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.10);
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.7rem;
    }

    .success-insight {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.10);
        border-left: 4px solid #22c55e;
        margin-bottom: 0.7rem;
    }

    .warning-insight {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: rgba(245, 158, 11, 0.10);
        border-left: 4px solid #f59e0b;
        margin-bottom: 0.7rem;
    }

    .danger-insight {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: rgba(239, 68, 68, 0.10);
        border-left: 4px solid #ef4444;
        margin-bottom: 0.7rem;
    }

    /* Student profile */
    .profile-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }

    /* Small label */
    .small-label {
        font-size: 0.82rem;
        opacity: 0.7;
        margin-bottom: 0.2rem;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

SCORE_KEYWORDS = [
    "score",
    "marks",
    "mark",
    "percentage",
    "percent"
]

ID_KEYWORDS = [
    "student id",
    "student_id",
    "roll number",
    "roll no",
    "register number",
    "registration number",
    "student number"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column(column):
    """Normalize column names for matching."""
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(column).lower()
    ).strip()


def clean_column_names(df):
    """Clean whitespace around column names."""
    df = df.copy()
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_bytes(file_bytes, file_extension):
    """Load uploaded data once and cache it by file contents."""
    buffer = io.BytesIO(file_bytes)

    if file_extension == "csv":
        return pd.read_csv(buffer)

    if file_extension == "xlsx":
        return pd.read_excel(buffer)

    raise ValueError("Only CSV and XLSX files are supported.")


def load_uploaded_file(uploaded_file):
    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]
    return load_uploaded_bytes(uploaded_file.getvalue(), extension)


def clean_dataset(df):

    df = df.copy()

    # Remove completely empty rows/columns
    df = df.dropna(
        axis=0,
        how="all"
    )

    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove exact duplicates
    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates()

    return df, duplicate_count


def find_student_id(df):

    for column in df.columns:

        normalized = normalize_column(column)

        for keyword in ID_KEYWORDS:

            if keyword in normalized:

                return column

    return None


def detect_score_columns(df):

    candidates = []

    for column in df.columns:

        normalized = normalize_column(column)

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        if any(
            keyword in normalized
            for keyword in SCORE_KEYWORDS
        ):
            candidates.append(column)

    return candidates


def detect_named_subjects(df):

    """
    Detect common subject columns.
    """

    subject_map = {}

    for column in df.columns:

        normalized = normalize_column(column)

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        if "math" in normalized:
            subject_map["Mathematics"] = column

        elif "reading" in normalized:
            subject_map["Reading"] = column

        elif "writing" in normalized:
            subject_map["Writing"] = column

        elif "science" in normalized:
            subject_map["Science"] = column

        elif "english" in normalized:
            subject_map["English"] = column

    return subject_map


def create_student_ids(df, id_column):

    if id_column:

        return df[id_column].astype(str)

    return pd.Series(
        [
            f"ST-{i:04d}"
            for i in range(1, len(df) + 1)
        ],
        index=df.index
    )


def calculate_overall(df, score_columns):

    result = df.copy()

    result["Overall Performance"] = (
        result[score_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .mean(axis=1)
    )

    return result


def performance_status(score):

    if pd.isna(score):
        return "Unknown"

    if score < 50:
        return "High Priority"

    if score < 65:
        return "Needs Support"

    if score < 80:
        return "On Track"

    return "Excellent"


def attention_score(row, score_columns):

    score = row["Overall Performance"]

    if pd.isna(score):
        return 0

    attention = 0

    # Overall performance
    if score < 50:
        attention += 60

    elif score < 65:
        attention += 35

    elif score < 75:
        attention += 15

    # Weak subject
    subject_values = []

    for col in score_columns:

        value = pd.to_numeric(
            row.get(col),
            errors="coerce"
        )

        if not pd.isna(value):
            subject_values.append(value)

    if subject_values:

        weakest = min(subject_values)

        if weakest < 50:
            attention += 25

        elif weakest < 60:
            attention += 15

        elif weakest < 70:
            attention += 5

    # Attendance if available
    for column in row.index:

        normalized = normalize_column(column)

        if "attendance" in normalized:

            value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            if not pd.isna(value):

                if value < 75:
                    attention += 20

                elif value < 85:
                    attention += 8

    return min(attention, 100)


def get_risk_label(score):

    if score >= 60:
        return "High"

    if score >= 30:
        return "Moderate"

    return "Low"


def get_student_reasons(row, score_columns, class_average):

    reasons = []

    overall = row["Overall Performance"]

    if overall < 50:

        reasons.append(
            "Overall performance is below the passing range."
        )

    elif overall < 65:

        reasons.append(
            "Overall performance is below the class target."
        )

    # Weakest subject
    values = {}

    for column in score_columns:

        value = pd.to_numeric(
            row[column],
            errors="coerce"
        )

        if not pd.isna(value):
            values[column] = value

    if values:

        weakest_column = min(
            values,
            key=values.get
        )

        weakest_value = values[weakest_column]

        if weakest_value < 60:

            reasons.append(
                f"{weakest_column} is currently "
                f"{weakest_value:.0f}, indicating a clear area for improvement."
            )

    # Attendance
    for column in row.index:

        normalized = normalize_column(column)

        if "attendance" in normalized:

            value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            if not pd.isna(value) and value < 75:

                reasons.append(
                    f"Attendance is {value:.0f}%, "
                    "which may affect academic consistency."
                )

    # Class comparison
    if overall < class_average - 10:

        reasons.append(
            "Performance is significantly below the class average."
        )

    return reasons[:3]


def recommendations_for_student(
    row,
    score_columns,
    class_means
):

    recommendations = []

    # Weakest subject
    subject_values = {}

    for column in score_columns:

        value = pd.to_numeric(
            row[column],
            errors="coerce"
        )

        if not pd.isna(value):

            subject_values[column] = value

    if subject_values:

        weakest = min(
            subject_values,
            key=subject_values.get
        )

        weakest_value = subject_values[weakest]

        if weakest_value < 60:

            recommendations.append(
                f"Provide additional practice in {weakest}."
            )

    # Attendance
    for column in row.index:

        normalized = normalize_column(column)

        if "attendance" in normalized:

            value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            if not pd.isna(value) and value < 75:

                recommendations.append(
                    "Discuss attendance and identify barriers to regular participation."
                )

    # Preparation course
    for column in row.index:

        normalized = normalize_column(column)

        if "test preparation" in normalized:

            value = str(
                row[column]
            ).lower()

            if "none" in value:

                recommendations.append(
                    "Consider additional preparation or guided practice before assessments."
                )

    if not recommendations:

        recommendations.append(
            "Continue the current learning routine and monitor the next assessment."
        )

    return recommendations[:3]


def train_subject_model(X, y):
    """Train and cache one lightweight Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def build_subject_model(
    df,
    target_column,
    feature_columns
):
    model_df = df[
        feature_columns + [target_column]
    ].copy()

    model_df = model_df.dropna(
        subset=[target_column]
    )

    if len(model_df) < 50:
        return None, None, None

    X = model_df[feature_columns]
    y = pd.to_numeric(
        model_df[target_column],
        errors="coerce"
    )

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    numeric_features = [
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column])
    ]

    if not numeric_features:
        return None, None, None

    X = X[numeric_features].copy()
    X = X.fillna(X.median())

    if X.shape[1] == 0:
        return None, None, None

    model = train_subject_model(X, y)
    return model, X, y

def generate_expected_scores(df, score_columns):

    """
    Uses Random Forest to estimate expected subject
    performance from the other available subject scores.

    This avoids predicting an overall score from the
    same scores used to calculate it.
    """

    result = df.copy()

    model_results = {}

    for target in score_columns:

        feature_columns = [
            col
            for col in score_columns
            if col != target
        ]

        if not feature_columns:
            continue

        model, X, y = build_subject_model(
            result,
            target,
            feature_columns
        )

        if model is None:
            continue

        try:

            # Train once on the complete available
            # dataset for fast deployment inference.
            model.fit(X, y)

            prediction_input = (
                result[feature_columns]
                .copy()
            )

            prediction_input = prediction_input.fillna(
                X.median()
            )

            predictions = model.predict(
                prediction_input
            )

            expected_column = (
                f"Expected {target}"
            )

            result[expected_column] = np.clip(
                predictions,
                0,
                100
            )

            gap_column = (
                f"Gap {target}"
            )

            result[gap_column] = (
                result[target]
                - result[expected_column]
            )

            model_results[target] = model

        except Exception:
            continue

    return result, model_results


def identify_main_subject(score_columns, column):

    normalized = normalize_column(column)

    if "math" in normalized:
        return "Mathematics"

    if "reading" in normalized:
        return "Reading"

    if "writing" in normalized:
        return "Writing"

    if "science" in normalized:
        return "Science"

    if "english" in normalized:
        return "English"

    return column


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">
    <div class="hero-title">🎓 StudentIQ</div>
    <div class="hero-subtitle">
        Academic Performance & Early Intervention Platform
    </div>
    <div class="hero-description">
        Upload student records and get a clear picture of
        class performance, students who may need support,
        and practical actions for improvement.
    </div>
</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Student Records")

    uploaded_file = st.file_uploader(
        "Upload your student file",
        type=["csv", "xlsx"],
        help="Upload CSV or Excel student records."
    )

    st.divider()

    st.caption(
        "StudentIQ works best with academic records "
        "containing student scores or marks."
    )

    st.divider()

    st.caption(
        "🔒 Data is processed within this application session."
    )


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel file from the sidebar to begin."
    )

    st.markdown("### What StudentIQ helps with")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "**📊 Understand the class**\n\n"
            "See the overall academic picture without "
            "having to analyse spreadsheets manually."
        )

    with col2:
        st.markdown(
            "**🚨 Find students needing support**\n\n"
            "Identify students whose performance "
            "may require attention."
        )

    with col3:
        st.markdown(
            "**🎯 Take action**\n\n"
            "Get simple explanations and practical "
            "recommendations."
        )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_uploaded_file(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read the uploaded file: {e}"
    )

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_student_data(raw_df):
    """Prepare and analyze the uploaded dataset once per dataset."""
    df = clean_column_names(raw_df)

    df, duplicate_count = clean_dataset(df)

    if df.empty:

        raise ValueError(
            "The uploaded file does not contain usable records."
        )


    # ============================================================
    # REMOVE OLD GENERATED COLUMNS
    # ============================================================

    generated_columns = [
        "Overall Performance",
        "_Performance",
        "Risk Level",
        "Attention Score",
        "Support Level"
    ]

    df = df.drop(
        columns=[
            column
            for column in generated_columns
            if column in df.columns
        ],
        errors="ignore"
    )


    # ============================================================
    # IDENTIFY STUDENT ID
    # ============================================================

    student_id_column = find_student_id(df)

    df["_Student ID"] = create_student_ids(
        df,
        student_id_column
    )


    # ============================================================
    # IDENTIFY SCORE COLUMNS
    # ============================================================

    score_columns = detect_score_columns(df)

    named_subjects = detect_named_subjects(df)

    # Prefer the known subject structure when available
    if len(named_subjects) >= 2:

        score_columns = list(
            named_subjects.values()
        )

    if len(score_columns) < 2:

        raise ValueError(
            "StudentIQ needs at least two numeric academic score/mark columns "
            "to perform meaningful performance analysis. "
            f"Detected numeric columns: "
            f"{df.select_dtypes(include=np.number).columns.tolist()}"
        )


    # ============================================================
    # CONVERT SCORE COLUMNS
    # ============================================================

    for column in score_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # Remove rows where every score is missing
    df = df.dropna(
        subset=score_columns,
        how="all"
    )


    # ============================================================
    # OVERALL PERFORMANCE
    # ============================================================

    df = calculate_overall(
        df,
        score_columns
    )


    # ============================================================
    # STATUS + ATTENTION
    # ============================================================

    # Vectorized support-level classification.
    overall = df["Overall Performance"]

    df["Support Level"] = np.select(
        [
            overall < 50,
            overall < 65,
            overall < 80
        ],
        [
            "High Priority",
            "Needs Support",
            "On Track"
        ],
        default="Excellent"
    )

    # Vectorized attention calculation for large datasets.
    overall_values = df["Overall Performance"].fillna(0)

    attention = np.select(
        [
            overall_values < 50,
            overall_values < 65,
            overall_values < 75
        ],
        [
            60,
            35,
            15
        ],
        default=0
    ).astype(float)

    subject_min = df[score_columns].min(axis=1)

    attention += np.select(
        [
            subject_min < 50,
            subject_min < 60,
            subject_min < 70
        ],
        [
            25,
            15,
            5
        ],
        default=0
    ).astype(float)

    attendance_column = None
    for column in df.columns:
        if "attendance" in normalize_column(column):
            attendance_column = column
            break

    if attendance_column:
        attendance_values = pd.to_numeric(
            df[attendance_column],
            errors="coerce"
        )

        attention += np.select(
            [
                attendance_values < 75,
                attendance_values < 85
            ],
            [
                20,
                8
            ],
            default=0
        ).astype(float)

    df["Attention Score"] = np.minimum(attention, 100)

    # Vectorized risk classification.
    df["Risk Level"] = np.select(
        [
            df["Attention Score"] >= 60,
            df["Attention Score"] >= 30
        ],
        [
            "High",
            "Moderate"
        ],
        default="Low"
    )


    smart_df, subject_models = generate_expected_scores(
        df,
        score_columns
    )

    return df, smart_df, subject_models, score_columns, named_subjects, duplicate_count


try:
    (
        df,
        smart_df,
        subject_models,
        score_columns,
        named_subjects,
        duplicate_count
    ) = prepare_student_data(df)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unable to process the uploaded file: {e}")
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.success(
    f"{len(df):,} student records ready"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "👥 Students",
        "👤 Student Profile",
        "💡 Insights",
        "📄 Reports"
    ]
)


# ============================================================
# COMMON VALUES
# ============================================================

overall = df[
    "Overall Performance"
].dropna()

class_average = overall.mean()

high_priority_count = int(
    (df["Support Level"] == "High Priority").sum()
)

support_count = int(
    (df["Support Level"] == "Needs Support").sum()
)

on_track_count = int(
    (df["Support Level"] == "On Track").sum()
)

excellent_count = int(
    (df["Support Level"] == "Excellent").sum()
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Class Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Students",
        f"{len(df):,}"
    )

    c2.metric(
        "Average Performance",
        f"{class_average:.1f}%"
    )

    c3.metric(
        "On Track",
        f"{on_track_count + excellent_count:,}"
    )

    c4.metric(
        "Need Attention",
        f"{high_priority_count + support_count:,}"
    )

    st.markdown(
        '<div class="section-title">Class Health</div>',
        unsafe_allow_html=True
    )

    health_data = pd.DataFrame({
        "Status": [
            "Excellent",
            "On Track",
            "Needs Support",
            "High Priority"
        ],
        "Students": [
            excellent_count,
            on_track_count,
            support_count,
            high_priority_count
        ]
    })

    fig = px.bar(
        health_data,
        x="Status",
        y="Students",
        text="Students",
        title="How students are currently performing"
    )

    fig.update_layout(
        height=360,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Key insights
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">What stands out?</div>',
        unsafe_allow_html=True
    )

    # Weakest subject
    subject_means = (
        df[score_columns]
        .mean()
        .sort_values()
    )

    weakest_subject = subject_means.index[0]
    strongest_subject = subject_means.index[-1]

    weakest_value = subject_means.iloc[0]
    strongest_value = subject_means.iloc[-1]

    st.markdown(
        f"""
        <div class="warning-insight">
        <strong>📚 Area needing attention</strong><br>
        {weakest_subject} has the lowest average score
        at <strong>{weakest_value:.1f}%</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="success-insight">
        <strong>⭐ Strongest area</strong><br>
        {strongest_subject} has the highest average score
        at <strong>{strongest_value:.1f}%</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )

    if high_priority_count > 0:

        percentage = (
            high_priority_count
            / len(df)
            * 100
        )

        st.markdown(
            f"""
            <div class="danger-insight">
            <strong>🚨 Students requiring priority attention</strong><br>
            {high_priority_count} students
            ({percentage:.1f}% of the class)
            show indicators that may require immediate support.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="success-insight">
            <strong>✓ No high-priority group detected</strong><br>
            Continue monitoring student performance and
            address moderate concerns early.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Subject performance
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Subject Performance</div>',
        unsafe_allow_html=True
    )

    subject_data = pd.DataFrame({
        "Subject": [
            identify_main_subject(
                score_columns,
                col
            )
            for col in score_columns
        ],
        "Average": [
            df[col].mean()
            for col in score_columns
        ]
    })

    fig = px.bar(
        subject_data,
        x="Subject",
        y="Average",
        text_auto=".1f",
        title="Average performance by subject"
    )

    fig.update_layout(
        yaxis_title="Average score",
        xaxis_title="",
        height=350,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# STUDENTS
# ============================================================

elif page == "👥 Students":

    st.markdown(
        '<div class="section-title">Students Needing Support</div>',
        unsafe_allow_html=True
    )

    st.write(
        "These students have academic indicators that may "
        "benefit from additional attention."
    )

    filter_option = st.selectbox(
        "Show",
        [
            "All students",
            "High Priority",
            "Needs Support",
            "On Track",
            "Excellent"
        ]
    )

    if filter_option == "All students":

        display_df = df.copy()

    else:

        display_df = df[
            df["Support Level"]
            == filter_option
        ].copy()

    display_df = display_df.sort_values(
        by="Attention Score",
        ascending=False
    )

    display_columns = [
        "_Student ID",
        "Overall Performance",
        "Support Level",
        "Attention Score"
    ]

    display_columns += score_columns

    display_columns = [
        col
        for col in display_columns
        if col in display_df.columns
    ]

    final_display = display_df[
        display_columns
    ].copy()

    final_display = final_display.rename(
        columns={
            "_Student ID": "Student",
            "Overall Performance": "Overall",
            "Support Level": "Status",
            "Attention Score": "Attention"
        }
    )

    st.dataframe(
        final_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Overall": st.column_config.NumberColumn(
                "Overall",
                format="%.1f"
            ),
            "Attention": st.column_config.NumberColumn(
                "Attention",
                format="%.0f"
            )
        }
    )

    st.caption(
        "Higher attention scores indicate more indicators "
        "that may require follow-up."
    )


# ============================================================
# STUDENT PROFILE
# ============================================================

elif page == "👤 Student Profile":

    st.markdown(
        '<div class="section-title">Student Profile</div>',
        unsafe_allow_html=True
    )

    selected_id = st.selectbox(
        "Select a student",
        df["_Student ID"].astype(str).tolist()
    )

    student = df[
        df["_Student ID"].astype(str)
        == selected_id
    ].iloc[0]

    overall_score = student[
        "Overall Performance"
    ]

    status = student[
        "Support Level"
    ]

    # Header
    st.markdown(
        f"""
        <div class="profile-card">

        <h2>👤 {selected_id}</h2>

        <p>
        Current status:
        <strong>{status}</strong>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Overall Performance",
        f"{overall_score:.1f}%"
    )

    m2.metric(
        "Attention Level",
        student["Risk Level"]
    )

    m3.metric(
        "Attention Score",
        f"{student['Attention Score']:.0f}/100"
    )

    # --------------------------------------------------------
    # Subject scores
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Academic Performance</div>',
        unsafe_allow_html=True
    )

    subject_rows = []

    for column in score_columns:

        score = student[column]

        subject_rows.append({
            "Subject": identify_main_subject(
                score_columns,
                column
            ),
            "Score": score
        })

    subject_df = pd.DataFrame(
        subject_rows
    )

    fig = px.bar(
        subject_df,
        x="Subject",
        y="Score",
        text_auto=".0f",
        title="Student subject performance"
    )

    fig.update_layout(
        height=320,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Why flagged?
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Why this student may need attention</div>',
        unsafe_allow_html=True
    )

    reasons = get_student_reasons(
        student,
        score_columns,
        class_average
    )

    if reasons:

        for reason in reasons:

            st.markdown(
                f"""
                <div class="warning-insight">
                • {reason}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <div class="success-insight">
            ✓ No significant academic warning signs
            were identified in the available records.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Suggested Actions</div>',
        unsafe_allow_html=True
    )

    recommendations = recommendations_for_student(
        student,
        score_columns,
        df[score_columns].mean()
    )

    for recommendation in recommendations:

        st.write(
            f"✓ {recommendation}"
        )

    # --------------------------------------------------------
    # Smart academic signal
    # --------------------------------------------------------

    smart_columns = [
        col
        for col in smart_df.columns
        if col.startswith("Gap ")
    ]

    if smart_columns:

        st.markdown(
            '<div class="section-title">Smart Academic Signal</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "This compares the student's actual subject "
            "performance with an estimated expected score "
            "based on their other available academic scores."
        )

        for gap_column in smart_columns:

            target_column = gap_column.replace(
                "Gap ",
                ""
            )

            gap = student.get(
                gap_column,
                np.nan
            )

            if pd.isna(gap):
                continue

            if gap <= -8:

                st.markdown(
                    f"""
                    <div class="danger-insight">
                    <strong>⚠ {target_column}</strong><br>
                    Actual performance is about
                    <strong>{abs(gap):.1f} points below expected</strong>.
                    Consider additional support in this area.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif gap >= 8:

                st.markdown(
                    f"""
                    <div class="success-insight">
                    <strong>⭐ {target_column}</strong><br>
                    Performance is about
                    <strong>{gap:.1f} points above expected</strong>.
                    This is a relative strength.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# INSIGHTS
# ============================================================

elif page == "💡 Insights":

    st.markdown(
        '<div class="section-title">Academic Insights</div>',
        unsafe_allow_html=True
    )

    st.write(
        "StudentIQ converts the uploaded records into "
        "simple observations that can support academic decisions."
    )

    # --------------------------------------------------------
    # Subject ranking
    # --------------------------------------------------------

    subject_means = (
        df[score_columns]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.subheader("Subject strengths and gaps")

    for i, (column, value) in enumerate(
        subject_means.items()
    ):

        subject_name = identify_main_subject(
            score_columns,
            column
        )

        if i == 0:

            st.markdown(
                f"""
                <div class="success-insight">
                <strong>⭐ Strongest subject: {subject_name}</strong><br>
                Class average: {value:.1f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        elif i == len(subject_means) - 1:

            st.markdown(
                f"""
                <div class="warning-insight">
                <strong>📚 Subject needing attention: {subject_name}</strong><br>
                Class average: {value:.1f}%
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # Preparation course
    # --------------------------------------------------------

    prep_column = None

    for column in df.columns:

        normalized = normalize_column(column)

        if "test preparation" in normalized:

            prep_column = column
            break

    if prep_column:

        st.subheader(
            "Preparation and performance"
        )

        prep_analysis = (
            df.groupby(
                prep_column
            )["Overall Performance"]
            .mean()
            .reset_index()
        )

        prep_analysis.columns = [
            "Preparation",
            "Average"
        ]

        fig = px.bar(
            prep_analysis,
            x="Preparation",
            y="Average",
            text_auto=".1f",
            title="Average performance by test preparation"
        )

        fig.update_layout(
            height=330,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        best_group = (
            prep_analysis
            .sort_values(
                "Average",
                ascending=False
            )
            .iloc[0]
        )

        st.markdown(
            f"""
            <div class="insight">
            Students in the <strong>{best_group['Preparation']}</strong>
            preparation group have the highest average performance
            at <strong>{best_group['Average']:.1f}%</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    attendance_column = None

    for column in df.columns:

        if "attendance" in normalize_column(column):

            attendance_column = column
            break

    if attendance_column:

        st.subheader(
            "Attendance signal"
        )

        attendance = pd.to_numeric(
            df[attendance_column],
            errors="coerce"
        )

        valid = attendance.notna()

        if valid.any():

            low_attendance = (
                attendance < 75
            ).sum()

            st.markdown(
                f"""
                <div class="warning-insight">
                <strong>Attendance:</strong>
                {low_attendance} students have attendance
                below 75%.
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # Model signal
    # --------------------------------------------------------

    if len(subject_models) > 0:

        st.subheader(
            "Smart performance signals"
        )

        st.write(
            "The system also checks whether a student's "
            "performance in one subject is substantially "
            "different from what would be expected from "
            "their other academic results."
        )

        gap_columns = [
            col
            for col in smart_df.columns
            if col.startswith("Gap ")
        ]

        for gap_column in gap_columns:

            target = gap_column.replace(
                "Gap ",
                ""
            )

            negative_count = int(
                (
                    smart_df[gap_column]
                    < -8
                ).sum()
            )

            if negative_count > 0:

                st.markdown(
                    f"""
                    <div class="warning-insight">
                    <strong>{target}</strong>:
                    {negative_count} students show a
                    noticeable performance gap in this area.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# REPORTS
# ============================================================

elif page == "📄 Reports":

    st.markdown(
        '<div class="section-title">Reports</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Download a clean academic summary for further "
        "review or sharing with staff."
    )

    report = df.copy()

    report = report.rename(
        columns={
            "_Student ID": "Student ID"
        }
    )

    # Remove internal technical columns
    technical_columns = [
        "Attention Score"
    ]

    report = report.drop(
        columns=technical_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Summary sheet
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "Metric": [
            "Total Students",
            "Average Performance",
            "Excellent",
            "On Track",
            "Needs Support",
            "High Priority"
        ],
        "Value": [
            len(df),
            round(class_average, 2),
            excellent_count,
            on_track_count,
            support_count,
            high_priority_count
        ]
    })

    # --------------------------------------------------------
    # Priority students
    # --------------------------------------------------------

    priority = df[
        df["Support Level"].isin(
            [
                "High Priority",
                "Needs Support"
            ]
        )
    ].copy()

    priority = priority.sort_values(
        "Attention Score",
        ascending=False
    )

    priority = priority.rename(
        columns={
            "_Student ID": "Student ID"
        }
    )

    priority = priority.drop(
        columns=[
            "Attention Score"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Create Excel report
    # --------------------------------------------------------

    @st.cache_data(show_spinner=False)
    def create_excel_report(summary_df, priority_df, report_df):
        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:
            summary_df.to_excel(
                writer,
                index=False,
                sheet_name="Class Summary"
            )
            priority_df.to_excel(
                writer,
                index=False,
                sheet_name="Needs Support"
            )
            report_df.to_excel(
                writer,
                index=False,
                sheet_name="Student Records"
            )

        return output.getvalue()

    report_bytes = create_excel_report(
        summary,
        priority,
        report
    )

    st.download_button(
        label="📥 Download Academic Report",
        data=report_bytes,
        file_name="StudentIQ_Academic_Report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )

    st.success(
        "Your academic report is ready."
    )

    # --------------------------------------------------------
    # Technical information
    # --------------------------------------------------------

    with st.expander(
        "Technical information"
    ):

        st.write(
            "This section is provided for technical review "
            "and is not required for normal staff usage."
        )

        st.write(
            "**Data processing:** Pandas / NumPy"
        )

        st.write(
            "**Visualization:** Plotly"
        )

        st.write(
            "**Predictive component:** Random Forest"
        )

        st.write(
            "**Application:** Streamlit"
        )

        st.write(
            f"**Records processed:** {len(df):,}"
        )

        st.write(
            f"**Academic indicators:** {len(score_columns)}"
        )

        st.write(
            f"**Smart subject models:** {len(subject_models)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "StudentIQ • Academic Performance & Early Intervention Platform"
)