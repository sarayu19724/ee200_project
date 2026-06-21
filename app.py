import streamlit as st
import pickle
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import collections
import os

from fingerprint import fingerprint, identify, compute_spectrogram, get_peaks

st.set_page_config(
    page_title="Sonic Signatures",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics

st.markdown("""
    <style>
    .metric-box {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #00d4ff;
        margin: 10px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_database():
    """Load the pre-built fingerprint database"""
    db_path = os.path.join(os.path.dirname(__file__), "database.pkl")
    if not os.path.exists(db_path):
        st.error("⚠️ Database not found! Please run `python build_database.py` first.")
        st.stop()

    with open(db_path, "rb") as f:
        db = pickle.load(f)
    return db


db = load_database()

# =====================================================
# HEADER
# =====================================================

col1,col2 = st.columns([1, 4])
with col1:

    st.write('')

with col2:
    st.title("🎵 Sonic Signatures")
    st.write("**Shazam-style Audio Fingerprinting** ")

st.divider()

# =====================================================
# NAVIGATION
# =====================================================

mode = st.sidebar.radio(
    "🎯 Choose Mode",
    ["📚 Library", "🔍 Identify", "📊 Batch"],
    key="mode_selector"
)

# =====================================================
# LIBRARY MODE
# =====================================================

if mode == "📚 Library":
    st.subheader("📚 Song Library")
    st.write("Fingerprints currently indexed in the database")

    # Extract song list from database
    songs_in_db = {}
    for hash_val, song_list in db.items():
        for song_name, _ in song_list:
            if song_name not in songs_in_db:
                songs_in_db[song_name] = 0
            songs_in_db[song_name] += 1

    if songs_in_db:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📖 Songs Indexed", len(songs_in_db))
        with col2:
            st.metric("🔗 Total Hashes", sum(len(v) for v in db.values()))
        with col3:
            avg_hashes = sum(songs_in_db.values()) / len(songs_in_db) if songs_in_db else 0
            st.metric("📌 Avg Hashes/Song", f"{int(avg_hashes)}")

        st.write("**Songs in database:**")
        df_songs = pd.DataFrame([
            {"Song": name, "Hash Count": count}
            for name, count in sorted(songs_in_db.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(df_songs, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No songs in database. Run build_database.py first.")

# =====================================================
# IDENTIFY MODE
# =====================================================

elif mode == "🔍 Identify":
    st.subheader("🔍 Identify a Clip")
    st.write("Upload an audio file to identify which song it belongs to")

    uploaded = st.file_uploader(
        "📤 Upload audio clip",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="single_upload"
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        try:
            from fingerprint import load_audio

            y = load_audio(path)

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(
                ["🎙️ Spectrogram", "⭐ Constellation Map", "🎯 Results", "📊 Offset Analysis"]
            )

            # ==================== SPECTROGRAM ====================
            with tab1:
                st.subheader("Spectrogram Analysis")
                S_db, freqs, times = compute_spectrogram(y)

                fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
                im = ax.pcolormesh(
                    times,
                    freqs[:len(freqs) // 4],
                    S_db[:len(freqs) // 4],
                    shading='gouraud',
                    cmap='magma'
                )
                ax.set_xlabel("Time (s)", fontsize=11)
                ax.set_ylabel("Frequency (Hz)", fontsize=11)
                ax.set_title("Time-Frequency Spectrogram", fontsize=13, fontweight='bold')
                plt.colorbar(im, ax=ax, label="Magnitude (dB)")
                plt.tight_layout()
                st.pyplot(fig)

            # ==================== CONSTELLATION ====================
            with tab2:
                st.subheader("Constellation Map (Peak Detection)")
                S_db, freqs, times = compute_spectrogram(y)
                peaks = get_peaks(S_db)

                fig2, ax2 = plt.subplots(figsize=(12, 5), dpi=100)
                ax2.imshow(S_db, origin="lower", aspect="auto", cmap='magma')

                if peaks:
                    pt = [t for f, t in peaks]
                    pf = [f for f, t in peaks]
                    ax2.scatter(pt, pf, s=6, c="cyan", alpha=0.7, edgecolors='white', linewidth=0.5)

                ax2.set_xlabel("Time Frames", fontsize=11)
                ax2.set_ylabel("Frequency Bins", fontsize=11)
                ax2.set_title(f"Peak Constellation ({len(peaks)} peaks detected)", fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2)

            # ==================== RESULTS ====================
            with tab3:
                st.subheader("🎯 Recognition Results")

                result = identify(y, db, top_n=5)

                if len(result):
                    best_song, best_score = result[0]

                    # Success message
                    st.success(f"✅ **Recognized Song**: {best_song}")
                    st.metric("Cluster Score", best_score)

                    # Top candidates
                    st.write("**Top Candidates:**")
                    candidates = pd.DataFrame([
                        {"Rank": i + 1, "Song": song, "Score": score}
                        for i, (song, score) in enumerate(result)
                    ])
                    st.dataframe(candidates, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ No match found. The clip may not be in the database.")

            # ==================== OFFSET HISTOGRAM ====================
            with tab4:
                st.subheader("📊 Offset Histogram")

                if len(result):
                    best_song = result[0][0]

                    from fingerprint import get_offset_histogram

                    counter = get_offset_histogram(y, db, best_song)

                    if counter:
                        fig3, ax3 = plt.subplots(figsize=(12, 5), dpi=100)
                        offsets = list(counter.keys())
                        matches = list(counter.values())

                        bars = ax3.bar(offsets, matches, color='cyan', alpha=0.7, edgecolor='white', linewidth=1)
                        ax3.set_xlabel("Time Offset (frames)", fontsize=11)
                        ax3.set_ylabel("Number of Matches", fontsize=11)
                        ax3.set_title(f"Hash Alignment for '{best_song}'", fontsize=13, fontweight='bold')
                        ax3.grid(axis='y', alpha=0.3)

                        # Highlight the peak
                        if matches:
                            peak_idx = matches.index(max(matches))
                            bars[peak_idx].set_color('lime')

                        plt.tight_layout()
                        st.pyplot(fig3)
                    else:
                        st.info("No offset data to display")

        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
        finally:
            if os.path.exists(path):
                os.remove(path)

# =====================================================
# BATCH MODE
# =====================================================

elif mode == "📊 Batch":
    st.subheader("📊 Batch Identification")
    st.write("Upload multiple clips at once and get a CSV with predictions")

    files = st.file_uploader(
        "📤 Upload multiple audio clips",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if files:
        if st.button("▶️ Run Batch", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            rows = []

            for idx, file in enumerate(files):
                status_text.text(f"Processing {idx + 1}/{len(files)}: {file.name}")

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(file.read())
                        path = tmp.name

                    from fingerprint import load_audio

                    y = load_audio(path)
                    result = identify(y, db, top_n=1)

                    prediction = result[0][0] if result else "Unknown"
                    rows.append({"filename": file.name, "prediction": prediction})

                except Exception as e:
                    rows.append({"filename": file.name, "prediction": f"Error: {str(e)}"})
                finally:
                    if os.path.exists(path):
                        os.remove(path)

                progress_bar.progress((idx + 1) / len(files))

            status_text.empty()
            progress_bar.empty()

            # Display results
            df = pd.DataFrame(rows)
            st.subheader("Results")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download button
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="⬇️ Download results.csv",
                data=csv,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success(f"✅ Processed {len(files)} files successfully!")
