import streamlit as st
from recom_v3 import MangaRecommender
import math
import random
import json
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =============================================================================
# Load and Cache the Recommender
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_recommender():
    """
    Initialize and return the MangaRecommender with custom weights.
    """
    recommender = MangaRecommender(
        synopsis_weight=0.4,
        genre_weight=0.35,
        theme_weight=0.15,
        numerical_weight=0.1,
        diversity_threshold=0.65
    )
    recommender.fit("cleaned_manga_data.json")
    return recommender

recommender = load_recommender()

# =============================================================================
# Pre-widget Reset Logic for Filters
# =============================================================================
if st.session_state.get('_reset_filters', False):
    st.session_state.selected_genres = []
    st.session_state.genre_logic = "AND"
    st.session_state.selected_themes = []
    st.session_state.theme_logic = "AND"
    st.session_state.selected_demographics = []
    st.session_state.demo_logic = "AND"
    st.session_state.score_filter = 0.0
    st.session_state.items_per_page = 10
    st.session_state.apply_browse_filters = False
    st.session_state._reset_filters = False

# =============================================================================
# Initialize Session State
# =============================================================================
if "detail_for" not in st.session_state:
    st.session_state.detail_for = recommender.data[0]['Title']

if "apply_browse_filters" not in st.session_state:
    st.session_state.apply_browse_filters = False

if "filter_version" not in st.session_state:
    st.session_state.filter_version = 0

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("Manga Recommender")

# --- Manga Selection ---
st.sidebar.markdown("### Select Manga")
manga_titles = sorted([m['Title'] for m in recommender.data])
selected_title = st.sidebar.selectbox(
    "Search or select a title",
    manga_titles,
    index=manga_titles.index(st.session_state.detail_for),
    key="selected_title"
)
st.session_state.detail_for = selected_title

# --- Browse Filters (affect 'Browse Manga' tab) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Browse Manga Filters")
filter_ver = st.session_state.filter_version

# Genre Filter
all_genres = sorted({g for m in recommender.data for g in m.get('Genres', [])})
selected_genres = st.sidebar.multiselect("Genres", all_genres, key=f"selected_genres_{filter_ver}")
genre_logic = st.sidebar.radio("Genre Filter Mode", ["AND", "OR", "WITHOUT"], horizontal=True, key=f"genre_logic_{filter_ver}")

# Theme Filter
all_themes = sorted({t for m in recommender.data for t in m.get('Themes', [])})
selected_themes = st.sidebar.multiselect("Themes", all_themes, key=f"selected_themes_{filter_ver}")
theme_logic = st.sidebar.radio("Theme Filter Mode", ["AND", "OR", "WITHOUT"], horizontal=True, key=f"theme_logic_{filter_ver}")

# Demographic Filter
all_demographics = sorted({m.get("Demographic", "unknown") for m in recommender.data})
selected_demographics = st.sidebar.multiselect("Demographics", all_demographics, key=f"selected_demographics_{filter_ver}")
demo_logic = st.sidebar.radio("Demographic Filter Mode", ["AND", "OR", "WITHOUT"], horizontal=True, key=f"demo_logic_{filter_ver}")

# Score & Pagination
score_filter = st.sidebar.slider("Minimum Score", 0.0, 10.0, 0.0, 0.1, key=f"score_filter_{filter_ver}")
items_per_page = st.sidebar.slider("Items per Page", 5, 30, 10, 5, key=f"items_per_page_{filter_ver}")

# --- Filter Control Buttons ---
if st.sidebar.button("🔄 Reset Filters"):
    st.session_state._reset_filters = True
    st.rerun()  # If st.rerun() is supported; otherwise, instruct the user to refresh.

if st.sidebar.button("✅ Apply Filters"):
    st.session_state.apply_browse_filters = True

# --- Surprise Me Button ---
st.sidebar.markdown("---")
if st.sidebar.button("🎲 Surprise Me!"):
    st.session_state.detail_for = random.choice(manga_titles)

# =============================================================================
# Helper: Show Manga Details
# =============================================================================
def show_manga_details(title):
    """
    Display the details of the selected manga.
    """
    details = recommender.get_manga_details(title)
    if not details:
        st.error("Details not found!")
        return

    st.title(details['Title'])
    col1, col2 = st.columns([1, 2])
    with col1:
        if details.get("Image URL"):
            st.image(details["Image URL"], use_container_width=True)
    with col2:
        st.markdown(f"**Score:** {details.get('Score', 'N/A')}")
        st.markdown(f"**Genres:** {', '.join(details.get('Genres', []))}")
        st.markdown(f"**Themes:** {', '.join(details.get('Themes', []))}")
        st.markdown(f"**Demographic:** {details.get('Demographic', 'N/A').title()}")
    st.markdown("### Synopsis")
    st.write(details.get('Synopsis', ''))

# --- Show Selected Manga Details ---
show_manga_details(st.session_state.detail_for)

# =============================================================================
# Save Feedback to Google Sheets
# =============================================================================
def save_feedback_to_gsheet(entry):
    """
    Saves a feedback entry to your Google Sheet "Manga_Feedback" in the worksheet "Feedback".
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    sheet = client.open("Manga_Feedback").worksheet("Feedback")
    sheet.append_row([
        entry["timestamp"],
        entry["query_title"],
        entry["recommended_title"],
        entry["feedback"],
        entry["similarity_score"],
    ])

# =============================================================================
# Tabs Layout
# =============================================================================
tabs = st.tabs(["Similar Recommendations", "Genre Suggestions", "Browse Manga"])

# --- Tab 1: Similar Recommendations ---
with tabs[0]:
    st.header("Similar Manga Recommendations")
    num_recs = st.slider("Number of Recommendations", 1, 20, 10)
    recs = recommender.recommend_by_title(st.session_state.detail_for, top_n=num_recs)
    for rec in recs:
        st.markdown("---")
        st.markdown(f"### {rec['Title']}  (Score: {rec['Score']:.2f})")
        col1, col2 = st.columns([1, 3])
        with col1:
            if rec.get("Image URL"):
                st.image(rec["Image URL"], use_container_width=True)
        with col2:
            similarity = f"{rec.get('similarity_score', 'N/A'):.2f}" if 'similarity_score' in rec else "N/A"
            st.markdown(f"**Similarity Score:** {similarity}")
            st.markdown(f"**Genres:** {', '.join(rec.get('Genres', []))}")
            st.markdown(f"**Themes:** {', '.join(rec.get('Themes', []))}")
            st.markdown(f"**Demographic:** {rec.get('Demographic', 'N/A').title()}")
            st.markdown(f"**Synopsis:** {rec.get('Synopsis', '')[:200]}...")

            # Feedback Buttons with Tooltips
            fb_col1, fb_col2 = st.columns([1, 1])
            with fb_col1:
                if st.button("👍", key=f"up_{rec['Title']}"):
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "query_title": st.session_state.detail_for,
                        "recommended_title": rec["Title"],
                        "feedback": "upvote",
                        "similarity_score": rec.get("similarity_score"),
                    }
                    save_feedback_to_gsheet(entry)
                    st.toast("Good recommendation recorded!", icon="✅")
                st.caption("Good recommendation")
            with fb_col2:
                if st.button("👎", key=f"down_{rec['Title']}"):
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "query_title": st.session_state.detail_for,
                        "recommended_title": rec["Title"],
                        "feedback": "downvote",
                        "similarity_score": rec.get("similarity_score"),
                    }
                    save_feedback_to_gsheet(entry)
                    st.toast("Bad recommendation recorded!", icon="⚠️")
                st.caption("Bad recommendation")

# --- Tab 2: Genre Suggestions ---
with tabs[1]:
    st.header("Similar Genre Suggestions")
    sim_genres = recommender.get_similar_genres(st.session_state.detail_for)
    if sim_genres:
        st.markdown(f"Genres similar to **{st.session_state.detail_for}**:")
        for genre in sim_genres:
            st.subheader(genre.title())
            # Display sample manga for each genre (no browse button here)
            genre_manga = [m for m in recommender.data if genre in m.get("Genres", [])]
            samples = random.sample(genre_manga, min(3, len(genre_manga)))
            cols = st.columns(len(samples))
            for col, manga in zip(cols, samples):
                with col:
                    st.image(manga.get("Image URL"), use_container_width=True, caption=manga["Title"])
            st.markdown("---")
    else:
        st.info("No similar genre suggestions found.")

# --- Tab 3: Browse Manga ---
with tabs[2]:
    st.header("Browse Manga")

    def filter_manga(data, genres, genre_mode, themes, theme_mode, demos, demo_mode, min_score):
        """
        Filters the manga data based on:
          - Minimum Score
          - Genre filtering (AND/OR/WITHOUT)
          - Theme filtering (AND/OR/WITHOUT)
          - Demographic filtering (AND/OR/WITHOUT)
        Returns the filtered list.
        """
        filtered = []
        for m in data:
            if m.get('Score', 0) < min_score:
                continue

            g = set(m.get("Genres", []))
            t = set(m.get("Themes", []))
            d = m.get("Demographic", "")

            # Apply Genre logic
            if genres:
                if genre_mode == "AND" and not all(gx in g for gx in genres):
                    continue
                elif genre_mode == "OR" and not any(gx in g for gx in genres):
                    continue
                elif genre_mode == "WITHOUT" and any(gx in g for gx in genres):
                    continue

            # Apply Theme logic
            if themes:
                if theme_mode == "AND" and not all(tx in t for tx in themes):
                    continue
                elif theme_mode == "OR" and not any(tx in t for tx in themes):
                    continue
                elif theme_mode == "WITHOUT" and any(tx in t for tx in themes):
                    continue

            # Apply Demographic logic
            if demos:
                if demo_mode == "AND" and not all(dx == d for dx in demos):
                    continue
                elif demo_mode == "OR" and not any(dx == d for dx in demos):
                    continue
                elif demo_mode == "WITHOUT" and any(dx == d for dx in demos):
                    continue

            filtered.append(m)
        return filtered

    # Retrieve filter values using keys that include the filter version
    fv = st.session_state.filter_version
    selected_genres_value = st.session_state.get(f"selected_genres_{fv}", [])
    genre_logic_value = st.session_state.get(f"genre_logic_{fv}", "AND")
    selected_themes_value = st.session_state.get(f"selected_themes_{fv}", [])
    theme_logic_value = st.session_state.get(f"theme_logic_{fv}", "AND")
    selected_demographics_value = st.session_state.get(f"selected_demographics_{fv}", [])
    demo_logic_value = st.session_state.get(f"demo_logic_{fv}", "AND")
    score_filter_value = st.session_state.get(f"score_filter_{fv}", 0.0)
    items_per_page_value = st.session_state.get(f"items_per_page_{fv}", 10)

    if not st.session_state.apply_browse_filters:
        st.info("Use the filters in the sidebar and click 'Apply Filters' to browse manga.")
    else:
        filtered = filter_manga(
            recommender.data,
            selected_genres_value, genre_logic_value,
            selected_themes_value, theme_logic_value,
            selected_demographics_value, demo_logic_value,
            score_filter_value
        )

        # Sort filtered manga by Score (descending order)
        filtered = sorted(filtered, key=lambda m: m.get("Score", 0), reverse=True)

        total_pages = math.ceil(len(filtered) / items_per_page_value)
        page = st.number_input("Page", 1, max(1, total_pages), 1)
        start = (page - 1) * items_per_page_value
        end = start + items_per_page_value

        for manga in filtered[start:end]:
            st.markdown("---")
            st.markdown(f"### {manga['Title']}  (Score: {manga['Score']:.2f})")
            col1, col2 = st.columns([1, 3])
            with col1:
                if manga.get("Image URL"):
                    st.image(manga["Image URL"], use_container_width=True)
            with col2:
                st.markdown(f"**Genres:** {', '.join(manga.get('Genres', []))}")
                st.markdown(f"**Themes:** {', '.join(manga.get('Themes', []))}")
                st.markdown(f"**Demographic:** {manga.get('Demographic', 'N/A').title()}")
                st.markdown(f"**Synopsis:** {manga.get('Synopsis', '')[:200]}...")
        st.write(f"Page {page} of {total_pages}")
