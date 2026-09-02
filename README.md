# ☕ Coffee Content Pipeline

### Automated coffee content generation and publishing with GitHub Actions.

---

## What It Does

1. **Generates a prompt** with Groq.
2. **Generates an image** with FLUX.1-schnell via Hugging Face.
3. **Publishes the image** to Bluesky and Tumblr.
4. **Creates a vertical video** using MiDaS/OpenCV or ffmpeg.
5. **Adds music** from `assets/music/` when available.
6. **Uploads the video** to YouTube Shorts.
7. **Logs each run** to `logs/posts.jsonl`.
8. **Cleans up** generated files after publishing.

## Tech Stack

| Purpose    | Tool                          |
| ---------- | ----------------------------- |
| Compute    | GitHub Actions                |
| Prompt     | Groq                          |
| Image      | Hugging Face — FLUX.1-schnell |
| Video      | MiDaS + OpenCV / ffmpeg       |
| Publishing | Bluesky + Tumblr + YouTube    |

## Setup

Store API credentials in **GitHub Secrets**. Required services are Groq, Hugging Face, Bluesky, Tumblr, and YouTube. `GH_PAT` is used to update the Tumblr refresh token when it rotates.

Add `.mp3` files to `assets/music/` for background music.

## Notes

* YouTube OAuth must use **Production** mode for long-term scheduled uploads.
* MiDaS runs on CPU.
* Pinterest support is implemented but disabled.
* Failed runs keep generated files for debugging.
