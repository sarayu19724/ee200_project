#!/usr/bin/env python3
"""
Build the fingerprint database from the EE200 song collection.

This script indexes all songs from Sarayu's song database and creates
a database.pkl file for the Streamlit app.

Usage:
    python db.py
"""

import pickle
import os
from pathlib import Path
from fingerprint import build_database

# Sarayu's Windows path (from Downloads folder)
SONG_PATH = r"C:\Users\Shreya\Downloads\ee200_project\EE200_Project_Song_Database\EE200 Project Song Database"

print("=" * 70)
print("🎵 SONIC SIGNATURES - DATABASE BUILDER")
print("=" * 70)

# Check if the path exists
if not os.path.isdir(SONG_PATH):
    print(f"\n❌ Error: Song directory not found!")
    print(f"   Expected: {SONG_PATH}")
    print("\n💡 Make sure the folder exists at the correct location.")
    exit(1)

# Count songs
audio_files = set((
        list(Path(SONG_PATH).glob("*.mp3")) +
        list(Path(SONG_PATH).glob("*.MP3")) +
        list(Path(SONG_PATH).glob("*.wav")) +
        list(Path(SONG_PATH).glob("*.WAV")) +
        list(Path(SONG_PATH).glob("*.flac")) +
        list(Path(SONG_PATH).glob("*.FLAC")) +
        list(Path(SONG_PATH).glob("*.ogg")) +
        list(Path(SONG_PATH).glob("*.OGG"))
))

if not audio_files:
    print(f"\n❌ Error: No audio files found in:")
    print(f"   {SONG_PATH}")
    exit(1)

print(f"\n✅ Found {len(audio_files)} songs")
print(f"📂 Location: {SONG_PATH}")
print(f"\n⏱️  Building database... (this takes 5-10 minutes)")
print("=" * 70 + "\n")
print("Total files:", len(audio_files))
print("Unique files:", len(set(audio_files)))
try:
    # Build the database
    db, songs = build_database(SONG_PATH)

    # Calculate stats
    total_hashes = len(db)
    total_entries = sum(len(v) for v in db.values())
    avg_hashes = total_entries / len(songs) if songs else 0

    # Save to pickle file
    with open("database.pkl", "wb") as f:
        pickle.dump(db, f)

    file_size = os.path.getsize('database.pkl') / (1024 * 1024)

    # Print results
    print("\n" + "=" * 70)
    print("✅ DATABASE BUILT SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📊 Statistics:")
    print(f"   • Songs indexed: {len(songs)}")
    print(f"   • Unique hashes: {total_hashes:,}")
    print(f"   • Total hash entries: {total_entries:,}")
    print(f"   • Average hashes per song: {int(avg_hashes):,}")
    print(f"   • Database file size: {file_size:.2f} MB")
    print(f"\n💾 Saved to: database.pkl")
    print(f"\n🚀 Next step: streamlit run app.py")
    print("=" * 70 + "\n")

except Exception as e:
    print(f"\n❌ Error building database:")
    print(f"   {str(e)}")
    import traceback

    traceback.print_exc()
    exit(1)