#!/usr/bin/env python3
"""
Bulk-download YouTube clips of Shaheds, FPVs and other UAVs in flight.

Install:  pip install -U yt-dlp   (and have ffmpeg on PATH)
Run:      python grab_drone_videos.py --per-query 30 --out ./drone_videos
          python grab_drone_videos.py --min-height 1080   # keep only true 1080p+

Output:   <out>/<class>/<video_id>_<height>p.mp4  +  .info.json (source URL, title, uploader)
          <out>/archive.txt prevents re-downloading the same video across runs.
Note: footage is not open-licensed; keep info.json to track provenance.
"""
import argparse
from pathlib import Path
import yt_dlp

# class -> search queries (UA / RU / EN). Tuned toward ground-view "just flying" clips.
QUERIES = {
    "shahed": [
        "шахед летить", "шахед над містом", "шахед пролітає", "шахед вдень летить",
        "шахед летит", "шахед пролетел над", "герань-2 летит", "герань полет видео",
        "Shahed drone flying over", "Shahed 136 flying low", "Geran-2 in flight",
    ],
    "jet_shahed": [
        "реактивний шахед летить", "реактивний шахед над містом", "герань-3 летит",
        "jet Shahed flying", "Shahed 238 flying",
    ],
    "gerbera": ["гербера дрон летить", "гербера бпла летит", "Gerbera drone flying"],
    "fpv": [
        "FPV дрон летить", "FPV дрон в небі", "FPV дрон летит на камеру",
        "FPV drone flying in sky Ukraine", "fiber optic FPV drone flying",
    ],
    "lancet": ["ланцет летит", "Lancet drone flying", "ланцет бпла полет"],
    "orlan_zala": ["орлан-10 летить", "Orlan-10 flying", "ZALA drone flying", "зала бпла летить"],
    "quadcopter_recon": ["мавік летить в небі", "Mavic drone in the sky Ukraine", "розвідувальний дрон в небі"],
    # generic UAV sightings (mixed types, useful for a catch-all "uav" class)
    "bpla_generic": [
        "БпЛА летить", "БпЛА над містом", "БпЛА в небі відео", "дрон летить в небі",
        "БПЛА летит", "беспилотник летит над", "беспилотник в небе видео",
        "UAV flying over Ukraine", "drone spotted in sky Ukraine",
    ],
    "molniya": ["молнія дрон летить", "молния бпла летит", "молния-2 дрон", "Molniya drone flying"],
    "kub_italmas": ["куб бпла летит", "KUB-BLA flying", "італмас дрон", "Italmas drone flying"],
    "decoy": ["пародія дрон", "пародия бпла", "дрон-приманка летить", "Parodiya decoy drone"],
    "supercam_merlin": ["суперкам бпла", "Supercam drone flying", "мерлін бпла", "Merlin-VR drone"],
    "heavy_bomber": ["баба яга дрон летить", "вампір дрон летить", "Baba Yaga drone flying", "heavy hexacopter bomber drone night"],
    "ua_fixed_wing": ["лелека бпла політ", "фурія бпла", "Leleka-100 flying", "Shark UAV flying", "лютий дрон летить"],
    "large_uav": ["форпост бпла", "Orion drone Russia flying", "Mohajer-6 flying", "Bayraktar TB2 flying"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./drone_videos")
    ap.add_argument("--per-query", type=int, default=25, help="search results per query")
    ap.add_argument("--max-duration", type=int, default=300, help="seconds; skips long compilations")
    ap.add_argument("--min-height", type=int, default=720, help="drop videos below this resolution")
    ap.add_argument("--classes", nargs="*", default=list(QUERIES), help="subset of classes")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for cls in args.classes:
        opts = {
            "format": f"bv*[height>={args.min_height}][height<=1080]+ba/b[height>={args.min_height}][height<=1080]",
            "merge_output_format": "mp4",
            "outtmpl": str(out / cls / "%(id)s_%(height)sp.%(ext)s"),
            "download_archive": str(out / "archive.txt"),   # global dedupe across classes
            "match_filter": yt_dlp.utils.match_filter_func(
                f"duration < {args.max_duration} & !is_live"),
            "writeinfojson": True,
            "ignoreerrors": True,
            "noplaylist": True,
            "quiet": False,
            "sleep_interval": 2, "max_sleep_interval": 5,     # be polite, avoid throttling
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            for q in QUERIES[cls]:
                print(f"\n=== [{cls}] {q}")
                ydl.download([f"ytsearch{args.per_query}:{q}"])

if __name__ == "__main__":
    main()