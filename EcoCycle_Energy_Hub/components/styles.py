
import streamlit as st


def inject_styles():
    """Apply the EcoCycle Energy Hub global responsive UI styles."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

        :root {
            --forest: #176344;
            --deep: #104b34;
            --mint: #eaf4ee;
            --ink: #17251e;
            --muted: #6b7b70;
            --line: #e2eae4;
            --canvas: #f5f8f5;
            --white: #ffffff;
        }

        html, body, [class*="css"], .stApp {
            font-family: 'DM Sans', sans-serif;
            color: var(--ink);
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: var(--canvas);
        }

        .block-container {
            max-width: 1320px;
            padding: 1.2rem clamp(.8rem, 3vw, 2.4rem) 3rem;
        }

        header,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        footer {
            visibility: hidden;
        }

        h1, h2, h3, h4 {
            font-family: 'Manrope', sans-serif !important;
            letter-spacing: -0.04em !important;
            color: var(--ink) !important;
        }

        p {
            line-height: 1.65;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 12px;
            border: 1px solid var(--forest);
            background: var(--forest);
            color: white;
            font-weight: 700;
            min-height: 2.65rem;
            padding: .55rem 1rem;
            transition: all .2s ease;
        }

        .stButton > button:hover {
            background: var(--deep);
            border-color: var(--deep);
            transform: translateY(-1px);
            color: white;
        }

        .stButton > button p {
            color: inherit;
        }

        /* Upload area */
        [data-testid="stFileUploader"] section {
            border: 1.5px dashed #9eb9a7;
            border-radius: 18px;
            background: #fbfdfb;
        }

        /* Metrics and tables */
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 4px 18px #163c2410;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            overflow: hidden;
        }

        /* Top navigation */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 13px 18px;
            margin-bottom: 12px;
            box-shadow: 0 5px 22px #163c2408;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font: 800 1.02rem 'Manrope', sans-serif;
            letter-spacing: -.03em;
        }

        .brand-mark {
            height: 42px;
            width: 42px;
            border-radius: 13px;
            background: linear-gradient(145deg, #176344, #16877a);
            display: grid;
            place-items: center;
            color: white;
            font-weight: 800;
            font-size: 1.05rem;
        }

        .brand small {
            display: block;
            color: var(--muted);
            font: 500 .72rem 'DM Sans', sans-serif;
            letter-spacing: 0;
            margin-top: 2px;
        }

        .status-pill {
            border: 1px solid #d4e7d9;
            background: #edf7f0;
            color: var(--forest);
            border-radius: 999px;
            padding: 7px 12px;
            font-size: .76rem;
            font-weight: 700;
        }

        /* Navigation bar */
        .nav-wrap {
            position: sticky;
            top: 0;
            z-index: 99;
            background: #f5f8f5ed;
            backdrop-filter: blur(14px);
            padding: 7px 0 12px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 18px;
        }

        .nav-wrap [data-testid="stHorizontalBlock"] {
            gap: 5px;
        }

        .nav-wrap .stButton > button {
            background: transparent;
            border-color: transparent;
            color: #64766a;
            min-height: 2.35rem;
            padding: .4rem .72rem;
            font-size: .86rem;
        }

        .nav-wrap .stButton > button:hover {
            background: var(--mint);
            color: var(--deep);
            border-color: #d6e8dc;
            transform: none;
        }

        /* Homepage hero */
        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 25px;
            padding: clamp(28px, 5vw, 58px);
            background:
                radial-gradient(circle at 84% 15%, #ffffff24 0 2px, transparent 3px),
                linear-gradient(115deg, #103e2c 0%, #176344 55%, #16877a 100%);
            color: white;
            display: flex;
            align-items: center;
            gap: 24px;
            min-height: 280px;
            box-shadow: 0 18px 44px #103e2c22;
        }

        .hero:after {
            content: '';
            position: absolute;
            width: 330px;
            height: 330px;
            border: 1px solid #ffffff26;
            border-radius: 50%;
            right: 8%;
            top: -42%;
            box-shadow: 0 0 0 35px #ffffff0b,
                        0 0 0 72px #ffffff08;
        }

        .hero-copy {
            position: relative;
            z-index: 2;
            max-width: 730px;
        }

        .eyebrow {
            color: #bfe8d0;
            text-transform: uppercase;
            letter-spacing: .16em;
            font-size: .72rem;
            font-weight: 800;
            margin-bottom: 13px;
        }

        .hero h1 {
            color: white !important;
            font-size: clamp(2.1rem, 5vw, 3.7rem);
            line-height: 1.04;
            margin: 0 0 15px;
        }

        .hero p {
            color: #e0f0e6;
            font-size: 1.03rem;
            max-width: 650px;
            margin: 0;
        }

        .hero-aside {
            position: relative;
            z-index: 2;
            margin-left: auto;
            min-width: 170px;
            text-align: center;
        }

        .orbit {
            height: 150px;
            width: 150px;
            border: 1px solid #ffffff66;
            border-radius: 50%;
            display: grid;
            place-items: center;
            box-shadow: 0 0 0 17px #ffffff0c,
                        0 0 0 36px #ffffff08;
        }

        .orbit span {
            width: 82px;
            height: 82px;
            border-radius: 25px;
            background: #ffffff20;
            border: 1px solid #ffffff55;
            display: grid;
            place-items: center;
            font: 800 1.25rem 'Manrope', sans-serif;
            color: white;
        }

        /* Page headings */
        .section-kicker {
            color: var(--forest);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin: 23px 0 7px;
        }

        .page-title {
            font: 800 clamp(1.7rem, 3vw, 2.25rem) 'Manrope', sans-serif;
            letter-spacing: -.05em;
            margin: 4px 0;
        }

        .page-subtitle {
            color: var(--muted);
            margin: 0 0 20px;
        }

        /* Content cards */
        .panel {
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 5px 22px #163c2407;
        }

        .feature-panel {
            background: linear-gradient(135deg, #eaf4ee, #f7fbf8);
            border: 1px solid #d8e9dd;
            border-radius: 18px;
            padding: 22px;
        }

        .feature-panel h3 {
            margin-top: 0;
        }

        .mini-label {
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .1em;
            color: var(--muted);
            font-weight: 800;
        }

        .big-stat {
            font: 800 2rem 'Manrope', sans-serif;
            letter-spacing: -.05em;
            color: var(--forest);
        }

        .soft-note {
            background: #fff8e8;
            border: 1px solid #f0e2bd;
            border-radius: 13px;
            padding: 13px 16px;
            color: #725b25;
            font-size: .88rem;
            line-height: 1.55;
        }

        /* Images */
        [data-testid="stImage"] img {
            border-radius: 16px;
            object-fit: cover;
        }

        /* Footer */
        .footer {
            margin-top: 42px;
            padding: 18px 0;
            border-top: 1px solid var(--line);
            text-align: center;
            color: #829087;
            font-size: .78rem;
        }

        /* Responsive layout */
        @media (max-width: 760px) {
            .block-container {
                padding: .65rem .8rem 2rem;
            }

            .hero {
                padding: 28px 22px;
                min-height: 0;
                border-radius: 19px;
            }

            .hero-aside {
                display: none;
            }

            .nav-wrap {
                position: relative;
            }

            .nav-wrap [data-testid="stHorizontalBlock"] {
                flex-wrap: nowrap;
                overflow-x: auto;
            }

            .nav-wrap .stButton {
                min-width: max-content;
            }

            .nav-wrap .stButton > button {
                white-space: nowrap;
                padding: .4rem .55rem;
            }

            .panel {
                padding: 16px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Backwards-compatible alias if another module uses this name.
def apply_styles():
    inject_styles()
