# ============================================================
# AI-POWERED UNIVERSITY ADMISSIONS SYSTEM
# ============================================================
#
# TECHNOLOGIES:
#   Python
#   Flask
#   Flask-SQLAlchemy
#   SQLite
#   HTML5
#   CSS3
#   JavaScript
#   Scikit-learn
#
# ADMISSION SCORING:
#
# WAEC:
#   A1 = 10
#   B2 = 9
#   B3 = 8
#   C4 = 7
#   C5 = 6
#   C6 = 5
#
# WAEC Aggregate =
#   (Sum of 5 WAEC points / 50) * 40
#
# JAMB Aggregate =
#   (JAMB score / 400) * 60
#
# Total Score =
#   WAEC Aggregate + JAMB Aggregate
#
# Maximum Total Score = 100
#
# The admission decision is based on Total Score.
# ============================================================


from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    flash
)

from flask_sqlalchemy import SQLAlchemy

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///admissions.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.secret_key = "university-admission-secret-key"

db = SQLAlchemy(app)


# ============================================================
# 2. ADMISSION CONFIGURATION
# ============================================================

# IMPORTANT:
# Change this value to the official university
# admission cutoff when deploying the system.

ADMISSION_CUTOFF = 50.0

MAX_JAMB_SCORE = 400

MAX_WAEC_POINTS = 50

MAX_WAEC_AGGREGATE = 40

MAX_JAMB_AGGREGATE = 60

MAX_TOTAL_SCORE = 100


# ============================================================
# 3. WAEC GRADING SYSTEM
# ============================================================

WAEC_POINTS = {

    "A1": 10,

    "B2": 9,

    "B3": 8,

    "C4": 7,

    "C5": 6,

    "C6": 5
}


# ============================================================
# 4. UNIVERSITY COLLEGES AND COURSES
# ============================================================

COLLEGES = {

    "College of Humanity and Management Sciences": [

        "B.A. English",

        "B.A. Fine and Applied Arts",

        "B.A. Music",

        "B.A. Religious Studies",

        "B.Sc. Accounting",

        "B.Sc. Business Administration",

        "B.Sc. Economics",

        "B.Sc. Finance",

        "B.Sc. Industrial Relations & Personal Management",

        "B.Sc. Mass Communication",

        "B.Sc. Public Administration",

        "B.Sc. Securities and Investment"
    ],


    "College of Basic and Applied Sciences": [

        "B.Sc. Applied Geophysics",

        "B.Sc. Biochemistry",

        "B.Sc. Biology",

        "B.Sc. Biotechnology",

        "B.Sc. Chemistry",

        "B.Sc. Computer Science",

        "B.Sc. Software Engineering",

        "B.Sc. Cyber Security",

        "B.Sc. Mathematics",

        "B.Sc. Microbiology",

        "B.Sc. Physics",

        "B.Sc. Physics with Electronics",

        "B.Sc. Geology",

        "B.Sc. Industrial Chemistry",

        "B.Sc. Food Science and Technology"
    ],


    "College of Applied Health Sciences": [

        "B.Sc. Nursing Science",

        "B.Sc. Medical Laboratory Science",

        "B.Sc. Public Health",

        "B.Sc. Community Health",

        "B.Sc. Biomedical Laboratory Science",

        "B.Sc. Nutrition and Dietetics"
    ]
}


# ============================================================
# 5. AI PROGRAMME DESCRIPTIONS
# ============================================================
#
# This is used by the AI recommendation component.
#
# The final admission decision does NOT depend on this
# recommendation. The Total Score determines admission.

PROGRAM_DESCRIPTIONS = {

    "B.A. English":
        """
        English literature language writing communication
        grammar reading journalism education linguistics
        """,

    "B.A. Fine and Applied Arts":
        """
        art painting drawing design sculpture creativity
        visual arts graphics illustration
        """,

    "B.A. Music":
        """
        music singing instruments composition performance
        sound musical arts creativity
        """,

    "B.A. Religious Studies":
        """
        religion theology philosophy ethics faith culture
        religious education history
        """,

    "B.Sc. Accounting":
        """
        accounting finance auditing taxation bookkeeping
        financial reporting business numbers
        """,

    "B.Sc. Business Administration":
        """
        business management entrepreneurship leadership
        marketing organization administration
        """,

    "B.Sc. Economics":
        """
        economics finance markets statistics development
        business policy economic analysis
        """,

    "B.Sc. Finance":
        """
        finance investment banking money markets
        financial management economics
        """,

    "B.Sc. Industrial Relations & Personal Management":
        """
        human resources employment labour industrial
        relations management workers organization
        """,

    "B.Sc. Mass Communication":
        """
        journalism media broadcasting communication
        television radio writing public relations
        """,

    "B.Sc. Public Administration":
        """
        government administration public policy
        leadership governance management
        """,

    "B.Sc. Securities and Investment":
        """
        securities investment stocks bonds finance
        capital markets portfolio investment
        """,

    "B.Sc. Applied Geophysics":
        """
        physics earth science geology geophysics
        exploration minerals petroleum environment
        """,

    "B.Sc. Biochemistry":
        """
        biology chemistry medicine molecules laboratory
        biotechnology biological chemistry
        """,

    "B.Sc. Biology":
        """
        biology organisms life science animals plants
        genetics ecology evolution
        """,

    "B.Sc. Biotechnology":
        """
        biotechnology biology genetics molecular science
        laboratory medicine agriculture
        """,

    "B.Sc. Chemistry":
        """
        chemistry laboratory chemical reactions
        molecules materials science
        """,

    "B.Sc. Computer Science":
        """
        computer programming software algorithms
        artificial intelligence data science computing
        technology
        """,

    "B.Sc. Software Engineering":
        """
        software programming applications systems
        web development engineering testing
        """,

    "B.Sc. Cyber Security":
        """
        cybersecurity computer security networks
        hacking digital forensics information security
        """,

    "B.Sc. Mathematics":
        """
        mathematics algebra calculus statistics
        numerical analysis problem solving
        """,

    "B.Sc. Microbiology":
        """
        microbiology bacteria microorganisms laboratory
        medicine disease biotechnology
        """,

    "B.Sc. Physics":
        """
        physics energy matter mechanics electricity
        quantum science mathematics
        """,

    "B.Sc. Physics with Electronics":
        """
        physics electronics circuits electricity
        communication technology instrumentation
        """,

    "B.Sc. Geology":
        """
        geology earth rocks minerals environment
        geoscience exploration
        """,

    "B.Sc. Industrial Chemistry":
        """
        industrial chemistry chemicals manufacturing
        laboratory materials production
        """,

    "B.Sc. Food Science and Technology":
        """
        food science nutrition chemistry processing
        food production technology safety
        """,

    "B.Sc. Nursing Science":
        """
        nursing healthcare medicine patient care
        clinical health science hospital
        """,

    "B.Sc. Medical Laboratory Science":
        """
        medical laboratory medicine diagnostics
        blood disease clinical science laboratory
        """,

    "B.Sc. Public Health":
        """
        public health community medicine disease
        prevention epidemiology health promotion
        """,

    "B.Sc. Community Health":
        """
        community health healthcare public health
        prevention primary healthcare
        """,

    "B.Sc. Biomedical Laboratory Science":
        """
        biomedical laboratory medicine diagnostics
        biology chemistry clinical research
        """,

    "B.Sc. Nutrition and Dietetics":
        """
        nutrition diet food health medicine
        human nutrition wellness disease
        """
}


# ============================================================
# 6. CREATE AI PROGRAMME RECOMMENDER
# ============================================================

PROGRAM_NAMES = list(
    PROGRAM_DESCRIPTIONS.keys()
)


PROGRAM_TEXTS = [

    PROGRAM_DESCRIPTIONS[
        program
    ]

    for program in PROGRAM_NAMES
]


vectorizer = TfidfVectorizer(
    stop_words="english"
)


PROGRAM_MATRIX = vectorizer.fit_transform(
    PROGRAM_TEXTS
)


def recommend_program(
    applicant_interest
):

    """
    AI-assisted programme recommendation
    using TF-IDF and cosine similarity.
    """

    if not applicant_interest:

        return (
            None,
            0.0
        )


    applicant_vector = vectorizer.transform(
        [applicant_interest]
    )


    similarity_scores = cosine_similarity(
        applicant_vector,
        PROGRAM_MATRIX
    )[0]


    best_index = similarity_scores.argmax()


    recommended_program = (
        PROGRAM_NAMES[best_index]
    )


    confidence = (
        similarity_scores[best_index]
    )


    return (
        recommended_program,
        float(confidence)
    )


# ============================================================
# 7. DATABASE MODEL
# ============================================================

class Applicant(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )


    # --------------------------------------------
    # Personal information
    # --------------------------------------------

    first_name = db.Column(
        db.String(100),
        nullable=False
    )


    last_name = db.Column(
        db.String(100),
        nullable=False
    )


    email = db.Column(
        db.String(150),
        nullable=False
    )


    phone = db.Column(
        db.String(50)
    )


    # --------------------------------------------
    # College and programme
    # --------------------------------------------

    college = db.Column(
        db.String(200),
        nullable=False
    )


    course = db.Column(
        db.String(200),
        nullable=False
    )


    # --------------------------------------------
    # JAMB
    # --------------------------------------------

    jamb_score = db.Column(
        db.Integer,
        nullable=False
    )


    jamb_aggregate = db.Column(
        db.Float,
        nullable=False
    )


    # --------------------------------------------
    # Five WAEC grades
    # --------------------------------------------

    waec_subject1 = db.Column(
        db.String(10),
        nullable=False
    )


    waec_subject2 = db.Column(
        db.String(10),
        nullable=False
    )


    waec_subject3 = db.Column(
        db.String(10),
        nullable=False
    )


    waec_subject4 = db.Column(
        db.String(10),
        nullable=False
    )


    waec_subject5 = db.Column(
        db.String(10),
        nullable=False
    )


    # --------------------------------------------
    # WAEC points
    # --------------------------------------------

    waec_points1 = db.Column(
        db.Integer,
        nullable=False
    )


    waec_points2 = db.Column(
        db.Integer,
        nullable=False
    )


    waec_points3 = db.Column(
        db.Integer,
        nullable=False
    )


    waec_points4 = db.Column(
        db.Integer,
        nullable=False
    )


    waec_points5 = db.Column(
        db.Integer,
        nullable=False
    )


    waec_total_points = db.Column(
        db.Integer,
        nullable=False
    )


    waec_aggregate = db.Column(
        db.Float,
        nullable=False
    )


    # --------------------------------------------
    # Total score
    # --------------------------------------------

    total_score = db.Column(
        db.Float,
        nullable=False
    )


    # --------------------------------------------
    # Admission decision
    # --------------------------------------------

    decision = db.Column(
        db.String(50),
        nullable=False
    )


    # --------------------------------------------
    # AI recommendation
    # --------------------------------------------

    applicant_interest = db.Column(
        db.Text
    )


    ai_recommended_course = db.Column(
        db.String(200)
    )


    ai_confidence = db.Column(
        db.Float
    )


# ============================================================
# 8. SCORING FUNCTIONS
# ============================================================

def calculate_waec_points(
    grades
):

    """
    Convert five WAEC grades into points.
    """

    if len(grades) != 5:

        raise ValueError(
            "Exactly five WAEC subjects are required."
        )


    points = []


    for grade in grades:

        grade = grade.strip().upper()


        if grade not in WAEC_POINTS:

            raise ValueError(
                f"Invalid WAEC grade: {grade}"
            )


        points.append(
            WAEC_POINTS[grade]
        )


    return points


# ============================================================
# 9. WAEC AGGREGATE
# ============================================================

def calculate_waec_aggregate(
    waec_points
):

    """
    WAEC Aggregate =
    (Sum of WAEC points / 50) * 40
    """

    total_points = sum(
        waec_points
    )


    waec_aggregate = (
        total_points
        / MAX_WAEC_POINTS
    ) * MAX_WAEC_AGGREGATE


    return round(
        waec_aggregate,
        2
    )


# ============================================================
# 10. JAMB AGGREGATE
# ============================================================

def calculate_jamb_aggregate(
    jamb_score
):

    """
    JAMB Aggregate =
    (JAMB / 400) * 60
    """

    if (
        jamb_score < 0
        or jamb_score > MAX_JAMB_SCORE
    ):

        raise ValueError(
            "JAMB score must be between "
            "0 and 400."
        )


    jamb_aggregate = (
        jamb_score
        / MAX_JAMB_SCORE
    ) * MAX_JAMB_AGGREGATE


    return round(
        jamb_aggregate,
        2
    )


# ============================================================
# 11. TOTAL SCORE
# ============================================================

def calculate_total_score(
    jamb_score,
    waec_points
):

    """
    Total Score =
    WAEC Aggregate + JAMB Aggregate
    """

    waec_aggregate = (
        calculate_waec_aggregate(
            waec_points
        )
    )


    jamb_aggregate = (
        calculate_jamb_aggregate(
            jamb_score
        )
    )


    total_score = (
        waec_aggregate
        + jamb_aggregate
    )


    return (
        waec_aggregate,
        jamb_aggregate,
        round(
            total_score,
            2
        )
    )


# ============================================================
# 12. ADMISSION DECISION
# ============================================================

def determine_admission(
    total_score
):

    """
    The Total Score determines the admission decision.
    """

    if (
        total_score
        >= ADMISSION_CUTOFF
    ):

        return "ADMITTED"

    else:

        return "NOT ADMITTED"


# ============================================================
# 13. BASE HTML + CSS
# ============================================================

BASE_HTML = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0">

<title>
AI University Admissions System
</title>


<style>

/* ==========================================================
   GLOBAL CSS
   ========================================================== */

* {

    box-sizing: border-box;

}


body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        #f3f6fb;

    color:
        #1f2937;

    line-height:
        1.6;

}


/* ==========================================================
   NAVIGATION
   ========================================================== */

.navbar {

    background:
        #0f2a5f;

    color:
        white;

    padding:
        18px 5%;

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    flex-wrap:
        wrap;

}


.navbar .brand {

    font-size:
        21px;

    font-weight:
        bold;

}


.navbar a {

    color:
        white;

    text-decoration:
        none;

    margin-left:
        22px;

    font-weight:
        600;

}


.navbar a:hover {

    color:
        #93c5fd;

}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.container {

    width:
        92%;

    max-width:
        1200px;

    margin:
        35px auto;

}


/* ==========================================================
   CARDS
   ========================================================== */

.card {

    background:
        white;

    padding:
        28px;

    margin-bottom:
        25px;

    border-radius:
        14px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.07);

}


/* ==========================================================
   HEADINGS
   ========================================================== */

h1 {

    color:
        #0f2a5f;

    margin-bottom:
        25px;

}


h2 {

    color:
        #173f8a;

}


/* ==========================================================
   FORMS
   ========================================================== */

.form-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        20px;

}


.form-group {

    display:
        flex;

    flex-direction:
        column;

}


.form-group.full {

    grid-column:
        1 / -1;

}


label {

    font-weight:
        bold;

    margin-bottom:
        6px;

}


input,
select,
textarea {

    width:
        100%;

    padding:
        12px;

    border:
        1px solid #cbd5e1;

    border-radius:
        7px;

    font-size:
        15px;

    background:
        white;

}


input:focus,
select:focus,
textarea:focus {

    outline:
        none;

    border-color:
        #2563eb;

    box-shadow:
        0 0 0 3px
        rgba(37,99,235,0.12);

}


/* ==========================================================
   BUTTONS
   ========================================================== */

.btn {

    display:
        inline-block;

    background:
        #0f2a5f;

    color:
        white;

    padding:
        12px 22px;

    border:
        none;

    border-radius:
        7px;

    text-decoration:
        none;

    cursor:
        pointer;

    font-weight:
        bold;

}


.btn:hover {

    background:
        #173f8a;

}


.btn-success {

    background:
        #15803d;

}


/* ==========================================================
   STATISTICS
   ========================================================== */

.stats {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        20px;

}


.stat {

    background:
        white;

    padding:
        25px;

    border-radius:
        12px;

    text-align:
        center;

    box-shadow:
        0 4px 15px
        rgba(0,0,0,0.06);

}


.stat-number {

    font-size:
        35px;

    font-weight:
        bold;

    color:
        #0f2a5f;

}


/* ==========================================================
   SCORE DISPLAY
   ========================================================== */

.score-box {

    display:
        grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap:
        20px;

}


.score-item {

    background:
        #eff6ff;

    padding:
        25px;

    text-align:
        center;

    border-radius:
        10px;

}


.score-value {

    font-size:
        32px;

    font-weight:
        bold;

    color:
        #0f2a5f;

}


/* ==========================================================
   ADMISSION STATUS
   ========================================================== */

.admitted {

    color:
        #15803d;

    font-weight:
        bold;

}


.not-admitted {

    color:
        #b91c1c;

    font-weight:
        bold;

}


.decision-box {

    padding:
        25px;

    border-radius:
        12px;

    text-align:
        center;

    background:
        #f8fafc;

}


.decision {

    font-size:
        32px;

    font-weight:
        bold;

}


/* ==========================================================
   TABLE
   ========================================================== */

.table-container {

    overflow-x:
        auto;

}


table {

    width:
        100%;

    border-collapse:
        collapse;

}


th,
td {

    padding:
        13px;

    border-bottom:
        1px solid #e2e8f0;

    text-align:
        left;

}


th {

    background:
        #eaf1ff;

    color:
        #0f2a5f;

}


/* ==========================================================
   INFO BOX
   ========================================================== */

.info-box {

    background:
        #eff6ff;

    border-left:
        5px solid #2563eb;

    padding:
        18px;

    margin-bottom:
        20px;

    border-radius:
        5px;

}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {

    text-align:
        center;

    padding:
        30px;

    color:
        #64748b;

}


/* ==========================================================
   RESPONSIVE DESIGN
   ========================================================== */

@media(max-width: 900px) {

    .stats {

        grid-template-columns:
            repeat(2, 1fr);

    }

    .score-box {

        grid-template-columns:
            1fr;

    }

}


@media(max-width: 700px) {

    .form-grid {

        grid-template-columns:
            1fr;

    }

    .stats {

        grid-template-columns:
            1fr;

    }

    .navbar {

        flex-direction:
            column;

        align-items:
            flex-start;

    }

    .navbar a {

        margin-left:
            0;

        margin-right:
            15px;

    }

}

</style>

</head>


<body>


<nav class="navbar">

<div class="brand">

AI University Admissions

</div>


<div>

<a href="/">
Dashboard
</a>

<a href="/apply">
Apply
</a>

<a href="/applicants">
Applicants
</a>

</div>

</nav>


<div class="container">


{% with messages =
get_flashed_messages()
%}

{% for message in messages %}

<div class="info-box">

{{ message }}

</div>

{% endfor %}

{% endwith %}


{{ content|safe }}


</div>


<div class="footer">

AI-Powered University Admissions System

</div>


</body>

</html>

"""


# ============================================================
# 14. DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    applicants = Applicant.query.all()


    total = len(
        applicants
    )


    admitted = sum(

        1

        for applicant
        in applicants

        if applicant.decision
        == "ADMITTED"

    )


    not_admitted = sum(

        1

        for applicant
        in applicants

        if applicant.decision
        == "NOT ADMITTED"

    )


    average_score = (

        sum(
            applicant.total_score
            for applicant
            in applicants
        ) / total

        if total > 0

        else 0

    )


    content = f"""

<h1>
AI University Admissions Dashboard
</h1>


<div class="card">

<h2>
Admission Scoring Model
</h2>


<div class="info-box">

<strong>
WAEC Aggregate:
</strong>

(Sum of five WAEC points / 50)
× 40

<br><br>

<strong>
JAMB Aggregate:
</strong>

(JAMB / 400) × 60

<br><br>

<strong>
Total Score:
</strong>

WAEC Aggregate + JAMB Aggregate

<br><br>

<strong>
Maximum Total Score:
</strong>

100

<br><br>

<strong>
Admission Cutoff:
</strong>

{ADMISSION_CUTOFF}

</div>

</div>


<div class="stats">


<div class="stat">

<div class="stat-number">
{total}
</div>

Applications

</div>


<div class="stat">

<div class="stat-number">
{admitted}
</div>

Admitted

</div>


<div class="stat">

<div class="stat-number">
{not_admitted}
</div>

Not Admitted

</div>


<div class="stat">

<div class="stat-number">
{average_score:.2f}
</div>

Average Score

</div>


</div>


<br>


<div class="card">

<h2>
WAEC Grading System
</h2>


<div class="table-container">

<table>

<tr>

<th>
WAEC Grade
</th>

<th>
Points
</th>

</tr>


<tr>
<td>A1</td>
<td>10</td>
</tr>


<tr>
<td>B2</td>
<td>9</td>
</tr>


<tr>
<td>B3</td>
<td>8</td>
</tr>


<tr>
<td>C4</td>
<td>7</td>
</tr>


<tr>
<td>C5</td>
<td>6</td>
</tr>


<tr>
<td>C6</td>
<td>5</td>
</tr>

</table>

</div>

</div>


<div class="card">

<h2>
University Colleges
</h2>

<ul>

"""

    for college, courses in COLLEGES.items():

        content += f"""

<li>

<strong>
{college}
</strong>

({len(courses)} programmes)

</li>

"""


    content += """

</ul>


<a
class="btn"
href="/apply"
>

Start Application

</a>


</div>

"""


    return render_template_string(

        BASE_HTML,

        content=content

    )


# ============================================================
# 15. APPLICATION FORM
# ============================================================

@app.route(
    "/apply",
    methods=[
        "GET",
        "POST"
    ]
)
def apply():

    if request.method == "POST":

        try:

            # --------------------------------------------
            # Personal information
            # --------------------------------------------

            first_name = request.form[
                "first_name"
            ].strip()


            last_name = request.form[
                "last_name"
            ].strip()


            email = request.form[
                "email"
            ].strip()


            phone = request.form[
                "phone"
            ].strip()


            # --------------------------------------------
            # College/course
            # --------------------------------------------

            college = request.form[
                "college"
            ]


            course = request.form[
                "course"
            ]


            # Validate college

            if college not in COLLEGES:

                raise ValueError(
                    "Invalid college selected."
                )


            # Validate course

            if course not in COLLEGES[
                college
            ]:

                raise ValueError(
                    "The selected course does not "
                    "belong to the selected college."
                )


            # --------------------------------------------
            # JAMB
            # --------------------------------------------

            jamb_score = int(
                request.form[
                    "jamb_score"
                ]
            )


            if (
                jamb_score < 0
                or jamb_score > 400
            ):

                raise ValueError(
                    "JAMB score must be between "
                    "0 and 400."
                )


            # --------------------------------------------
            # Five WAEC subjects
            # --------------------------------------------

            grades = [

                request.form[
                    "waec_subject1"
                ].strip().upper(),

                request.form[
                    "waec_subject2"
                ].strip().upper(),

                request.form[
                    "waec_subject3"
                ].strip().upper(),

                request.form[
                    "waec_subject4"
                ].strip().upper(),

                request.form[
                    "waec_subject5"
                ].strip().upper()

            ]


            # --------------------------------------------
            # Convert WAEC grades to points
            # --------------------------------------------

            waec_points = (
                calculate_waec_points(
                    grades
                )
            )


            # --------------------------------------------
            # Calculate all scores
            # --------------------------------------------

            (
                waec_aggregate,
                jamb_aggregate,
                total_score
            ) = calculate_total_score(

                jamb_score,

                waec_points

            )


            # --------------------------------------------
            # Admission decision
            # --------------------------------------------

            decision = (
                determine_admission(
                    total_score
                )
            )


            # --------------------------------------------
            # AI programme recommendation
            # --------------------------------------------

            applicant_interest = request.form.get(
                "applicant_interest",
                ""
            ).strip()


            (
                ai_recommended_course,
                ai_confidence
            ) = recommend_program(
                applicant_interest
            )


            # --------------------------------------------
            # Save applicant
            # --------------------------------------------

            applicant = Applicant(

                first_name=first_name,

                last_name=last_name,

                email=email,

                phone=phone,

                college=college,

                course=course,

                jamb_score=jamb_score,

                jamb_aggregate=jamb_aggregate,

                waec_subject1=grades[0],

                waec_subject2=grades[1],

                waec_subject3=grades[2],

                waec_subject4=grades[3],

                waec_subject5=grades[4],

                waec_points1=waec_points[0],

                waec_points2=waec_points[1],

                waec_points3=waec_points[2],

                waec_points4=waec_points[3],

                waec_points5=waec_points[4],

                waec_total_points=sum(
                    waec_points
                ),

                waec_aggregate=waec_aggregate,

                total_score=total_score,

                decision=decision,

                applicant_interest=(
                    applicant_interest
                ),

                ai_recommended_course=(
                    ai_recommended_course
                ),

                ai_confidence=(
                    ai_confidence
                )

            )


            db.session.add(
                applicant
            )


            db.session.commit()


            return redirect(

                url_for(

                    "application_result",

                    applicant_id=applicant.id

                )

            )


        except ValueError as error:

            flash(
                str(error)
            )

            return redirect(
                url_for("apply")
            )


    # ========================================================
    # COLLEGE DATA FOR JAVASCRIPT
    # ========================================================

    college_data = repr(
        COLLEGES
    )


    content = """

<h1>
University Admission Application
</h1>


<div class="card">


<div class="info-box">

<strong>
Admission Calculation
</strong>

<br><br>

WAEC Aggregate =
(Sum of five WAEC points / 50) × 40

<br>

JAMB Aggregate =
(JAMB / 400) × 60

<br>

Total Score =
WAEC Aggregate + JAMB Aggregate

<br><br>

Maximum Total Score = 100

</div>


<form method="POST">


<h2>
1. Applicant Information
</h2>


<div class="form-grid">


<div class="form-group">

<label>
First Name
</label>

<input
type="text"
name="first_name"
required
>

</div>


<div class="form-group">

<label>
Last Name
</label>

<input
type="text"
name="last_name"
required
>

</div>


<div class="form-group">

<label>
Email
</label>

<input
type="email"
name="email"
required
>

</div>


<div class="form-group">

<label>
Phone Number
</label>

<input
type="text"
name="phone"
>

</div>


</div>


<h2>
2. College and Course
</h2>


<div class="form-grid">


<div class="form-group">

<label>
Select College
</label>

<select
id="college"
name="college"
required
>

<option value="">
-- Select College --
</option>

<option value="College of Humanity and Management Sciences">

College of Humanity and Management Sciences

</option>

<option value="College of Basic and Applied Sciences">

College of Basic and Applied Sciences

</option>

<option value="College of Applied Health Sciences">

College of Applied Health Sciences

</option>

</select>

</div>


<div class="form-group">

<label>
Select Course
</label>

<select
id="course"
name="course"
required
>

<option value="">
-- Select College First --
</option>

</select>

</div>


</div>


<h2>
3. JAMB Score
</h2>


<div class="form-group">

<label>
JAMB Score (0 - 400)
</label>

<input
type="number"
name="jamb_score"
min="0"
max="400"
required
>

</div>


<h2>
4. Five WAEC Subjects
</h2>


<p>
Select the WAEC grade obtained in each
of the five subjects.
</p>


<div class="form-grid">


<div class="form-group">

<label>
WAEC Subject 1
</label>

<select
name="waec_subject1"
required
>

<option value="">
Select Grade
</option>

<option value="A1">
A1 - 10 Points
</option>

<option value="B2">
B2 - 9 Points
</option>

<option value="B3">
B3 - 8 Points
</option>

<option value="C4">
C4 - 7 Points
</option>

<option value="C5">
C5 - 6 Points
</option>

<option value="C6">
C6 - 5 Points
</option>

</select>

</div>


<div class="form-group">

<label>
WAEC Subject 2
</label>

<select
name="waec_subject2"
required
>

<option value="">
Select Grade
</option>

<option value="A1">
A1 - 10 Points
</option>

<option value="B2">
B2 - 9 Points
</option>

<option value="B3">
B3 - 8 Points
</option>

<option value="C4">
C4 - 7 Points
</option>

<option value="C5">
C5 - 6 Points
</option>

<option value="C6">
C6 - 5 Points
</option>

</select>

</div>


<div class="form-group">

<label>
WAEC Subject 3
</label>

<select
name="waec_subject3"
required
>

<option value="">
Select Grade
</option>

<option value="A1">
A1 - 10 Points
</option>

<option value="B2">
B2 - 9 Points
</option>

<option value="B3">
B3 - 8 Points
</option>

<option value="C4">
C4 - 7 Points
</option>

<option value="C5">
C5 - 6 Points
</option>

<option value="C6">
C6 - 5 Points
</option>

</select>

</div>


<div class="form-group">

<label>
WAEC Subject 4
</label>

<select
name="waec_subject4"
required
>

<option value="">
Select Grade
</option>

<option value="A1">
A1 - 10 Points
</option>

<option value="B2">
B2 - 9 Points
</option>

<option value="B3">
B3 - 8 Points
</option>

<option value="C4">
C4 - 7 Points
</option>

<option value="C5">
C5 - 6 Points
</option>

<option value="C6">
C6 - 5 Points
</option>

</select>

</div>


<div class="form-group">

<label>
WAEC Subject 5
</label>

<select
name="waec_subject5"
required
>

<option value="">
Select Grade
</option>

<option value="A1">
A1 - 10 Points
</option>

<option value="B2">
B2 - 9 Points
</option>

<option value="B3">
B3 - 8 Points
</option>

<option value="C4">
C4 - 7 Points
</option>

<option value="C5">
C5 - 6 Points
</option>

<option value="C6">
C6 - 5 Points
</option>

</select>

</div>


</div>


<h2>
5. AI Programme Recommendation
</h2>


<div class="form-group full">

<label>
Describe your academic interests and
career goals
</label>

<textarea
name="applicant_interest"
rows="6"
placeholder="Example: I am interested in programming, artificial intelligence, software development and cybersecurity."
></textarea>

</div>


<br>


<button
type="submit"
class="btn"
>

Calculate Admission Score

</button>


</form>


</div>


<script>

/* ==========================================================
   CASCADING COLLEGE -> COURSE SELECTION
   ========================================================== */


const collegeCourses = """

    # Insert Python dictionary as JSON-like JavaScript
    import json

    content += json.dumps(
        COLLEGES
    )

    content += """

;


const collegeSelect =
    document.getElementById(
        "college"
    );


const courseSelect =
    document.getElementById(
        "course"
    );


collegeSelect.addEventListener(
    "change",
    function() {

        const selectedCollege =
            this.value;


        courseSelect.innerHTML =
            "";


        if (
            selectedCollege === ""
        ) {

            const option =
                document.createElement(
                    "option"
                );

            option.value = "";

            option.textContent =
                "-- Select College First --";

            courseSelect.appendChild(
                option
            );

            return;

        }


        const defaultOption =
            document.createElement(
                "option"
            );

        defaultOption.value = "";

        defaultOption.textContent =
            "-- Select Course --";

        courseSelect.appendChild(
            defaultOption
        );


        collegeCourses[
            selectedCollege
        ].forEach(
            function(course) {

                const option =
                    document.createElement(
                        "option"
                    );

                option.value =
                    course;

                option.textContent =
                    course;

                courseSelect.appendChild(
                    option
                );

            }
        );

    }
);

</script>

"""


    return render_template_string(

        BASE_HTML,

        content=content

    )


# ============================================================
# 16. APPLICATION RESULT
# ============================================================

@app.route(
    "/application/<int:applicant_id>"
)
def application_result(
    applicant_id
):

    applicant = Applicant.query.get_or_404(
        applicant_id
    )


    if applicant.decision == "ADMITTED":

        status_class = "admitted"

    else:

        status_class = "not-admitted"


    # Calculate percentage representation
    score_percentage = (
        applicant.total_score
    )


    content = f"""

<h1>
Admission Assessment Result
</h1>


<div class="card">


<h2>
Applicant
</h2>


<p>

<strong>
Application ID:
</strong>

{applicant.id}

</p>


<p>

<strong>
Name:
</strong>

{applicant.first_name}
{applicant.last_name}

</p>


<p>

<strong>
Email:
</strong>

{applicant.email}

</p>


<p>

<strong>
College:
</strong>

{applicant.college}

</p>


<p>

<strong>
Selected Course:
</strong>

{applicant.course}

</p>


</div>


<div class="card">


<h2>
JAMB Assessment
</h2>


<div class="score-box">


<div class="score-item">

<div class="score-value">

{applicant.jamb_score}

</div>

JAMB Score / 400

</div>


<div class="score-item">

<div class="score-value">

{applicant.jamb_aggregate:.2f}

</div>

JAMB Aggregate / 60

</div>


</div>


</div>


<div class="card">


<h2>
WAEC Assessment
</h2>


<div class="table-container">

<table>


<tr>

<th>
Subject
</th>

<th>
Grade
</th>

<th>
Points
</th>

</tr>


<tr>

<td>
Subject 1
</td>

<td>
{applicant.waec_subject1}
</td>

<td>
{applicant.waec_points1}
</td>

</tr>


<tr>

<td>
Subject 2
</td>

<td>
{applicant.waec_subject2}
</td>

<td>
{applicant.waec_points2}
</td>

</tr>


<tr>

<td>
Subject 3
</td>

<td>
{applicant.waec_subject3}
</td>

<td>
{applicant.waec_points3}
</td>

</tr>


<tr>

<td>
Subject 4
</td>

<td>
{applicant.waec_subject4}
</td>

<td>
{applicant.waec_points4}
</td>

</tr>


<tr>

<td>
Subject 5
</td>

<td>
{applicant.waec_subject5}
</td>

<td>
{applicant.waec_points5}
</td>

</tr>


<tr>

<th colspan="2">
Total WAEC Points
</th>

<th>
{applicant.waec_total_points}
/
50
</th>

</tr>


</table>

</div>


<br>


<div class="score-box">


<div class="score-item">

<div class="score-value">

{applicant.waec_aggregate:.2f}

</div>

WAEC Aggregate / 40

</div>


</div>


</div>


<div class="card">


<h2>
Total Admission Score
</h2>


<div class="score-box">


<div class="score-item">

<div class="score-value">

{applicant.waec_aggregate:.2f}

</div>

WAEC / 40

</div>


<div class="score-item">

<div class="score-value">

{applicant.jamb_aggregate:.2f}

</div>

JAMB / 60

</div>


<div class="score-item">

<div class="score-value">

{applicant.total_score:.2f}

</div>

TOTAL / 100

</div>


</div>


<br>


<div class="info-box">

<strong>
Calculation:
</strong>

<br><br>

WAEC Aggregate:

{applicant.waec_total_points}
/
50
× 40
=
{applicant.waec_aggregate:.2f}


<br><br>


JAMB Aggregate:

{applicant.jamb_score}
/
400
× 60
=
{applicant.jamb_aggregate:.2f}


<br><br>


Total Score:

{applicant.waec_aggregate:.2f}
+
{applicant.jamb_aggregate:.2f}
=
<strong>
{applicant.total_score:.2f}
</strong>

</div>


</div>


<div class="card">


<div class="decision-box">


<h2>
Admission Decision
</h2>


<div class="decision {status_class}">

{applicant.decision}

</div>


<p>

Admission Cutoff:

<strong>
{ADMISSION_CUTOFF}
/
100
</strong>

</p>


<p>

The admission decision is based on
the applicant's Total Score.

</p>


</div>


</div>


<div class="card">


<h2>
AI Programme Recommendation
</h2>


<p>

<strong>
Selected Course:
</strong>

{applicant.course}

</p>


<p>

<strong>
AI Recommended Course:
</strong>

{applicant.ai_recommended_course or "Not available"}

</p>


<p>

<strong>
AI Similarity Score:
</strong>

{applicant.ai_confidence:.2f}

</p>


<div class="info-box">

The AI recommendation is an
advisory programme-matching component.
The admission decision is determined
by the Total Score.

</div>


</div>

"""


    return render_template_string(

        BASE_HTML,

        content=content

    )


# ============================================================
# 17. APPLICANTS PAGE
# ============================================================

@app.route(
    "/applicants"
)
def applicants():

    records = Applicant.query.order_by(

        Applicant.total_score.desc()

    ).all()


    rows = ""


    position = 0


    for applicant in records:

        position += 1


        if applicant.decision == "ADMITTED":

            status_class = "admitted"

        else:

            status_class = "not-admitted"


        rows += f"""

<tr>

<td>
{position}
</td>


<td>
{applicant.id}
</td>


<td>
{applicant.first_name}
{applicant.last_name}
</td>


<td>
{applicant.college}
</td>


<td>
{applicant.course}
</td>


<td>
{applicant.jamb_score}
</td>


<td>
{applicant.waec_total_points}
</td>


<td>
<strong>
{applicant.total_score:.2f}
</strong>
</td>


<td class="{status_class}">
{applicant.decision}
</td>


<td>

<a
href="/application/{applicant.id}"
>

View

</a>

</td>


</tr>

"""


    content = f"""

<h1>
Applicants and Admission Ranking
</h1>


<div class="card">


<p>

Applicants are displayed in descending
order of Total Score.

</p>


<div class="table-container">

<table>


<thead>

<tr>

<th>
Rank
</th>

<th>
ID
</th>

<th>
Applicant
</th>

<th>
College
</th>

<th>
Course
</th>

<th>
JAMB
</th>

<th>
WAEC
</th>

<th>
Total
</th>

<th>
Decision
</th>

<th>
View
</th>

</tr>

</thead>


<tbody>

{rows}

</tbody>


</table>

</div>


</div>

"""


    return render_template_string(

        BASE_HTML,

        content=content

    )


# ============================================================
# 18. INITIALIZE DATABASE
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# 19. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=============================================="
    )

    print(
        " AI-POWERED UNIVERSITY ADMISSIONS SYSTEM"
    )

    print(
        "=============================================="
    )

    print(
        f"Admission Cutoff: "
        f"{ADMISSION_CUTOFF}/100"
    )

    print(
        "Maximum JAMB: 400"
    )

    print(
        "Maximum WAEC Points: 50"
    )

    print(
        "Maximum WAEC Aggregate: 40"
    )

    print(
        "Maximum JAMB Aggregate: 60"
    )

    print(
        "Maximum Total Score: 100"
    )

    print(
        "=============================================="
    )

    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "=============================================="
    )

    print()


    app.run(
        debug=True
    )