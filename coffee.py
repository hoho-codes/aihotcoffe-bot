import os
import time
import random
import subprocess
import sys
from huggingface_hub import InferenceClient
import requests
import glob
from nacl import encoding, public
from datetime import datetime, timezone

# --- Config from environment/secrets ---
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "stabilityai/stable-video-diffusion-img2vid-xt")
HF_VIDEO_PROVIDER = os.environ.get("HF_VIDEO_PROVIDER", "hf-inference")
# Kept short on purpose: shorter clips are cheaper/faster per generation
CLIP_DURATION_SECONDS = int(os.environ.get("CLIP_DURATION_SECONDS", "5"))
#PINTEREST_TOKEN = os.environ["PINTEREST_TOKEN"]
#BOARD_ID = os.environ["PINTEREST_BOARD_ID"]
GITHUB_REPO = os.environ["GITHUB_REPOSITORY"]  # auto-set by GitHub Actions, e.g. "user/repo"
GITHUB_BRANCH = os.environ.get("GITHUB_REF_NAME", "coffee")
IMAGE_FILENAME = "assets/generated_image.png"
VIDEO_FILENAME = "assets/generated_video.mp4"

TUMBLR_CONSUMER_KEY = os.environ["TUMBLR_CONSUMER_KEY"]
TUMBLR_CONSUMER_SECRET = os.environ["TUMBLR_CONSUMER_SECRET"]
TUMBLR_REFRESH_TOKEN = os.environ["TUMBLR_REFRESH_TOKEN"]
TUMBLR_BLOG_NAME = os.environ["TUMBLR_BLOG_NAME"]

BSKY_HANDLE = os.environ["BSKY_HANDLE"]
BSKY_APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]

GH_PAT = os.environ["GH_PAT"]  # Personal Access Token with repo scope, for updating secrets
GITHUB_REPO = os.environ["GITHUB_REPOSITORY"]
GITHUB_BRANCH = os.environ.get("GITHUB_REF_NAME", "main")

YT_CLIENT_ID = os.environ["YT_CLIENT_ID"]
YT_CLIENT_SECRET = os.environ["YT_CLIENT_SECRET"]
YT_REFRESH_TOKEN = os.environ["YT_REFRESH_TOKEN"]
YT_PRIVACY_STATUS = os.environ.get("YT_PRIVACY_STATUS", "unlisted")

FALLBACK_PROMPTS = [
    "a steaming latte on a rustic wooden cafe table, morning sunlight, cozy atmosphere",
    "a cappuccino with latte art next to an open book, cozy cafe interior, soft light",
    "an iced coffee on a marble table, city cafe window in background, bright daylight",
    "a pour-over coffee setup on a cafe counter, warm afternoon light, minimalist",
    "a flat white on a cafe table with a croissant, natural window light",
    "a coffee cup on an outdoor cafe table, european street in background, golden hour",
]

EFFECTS = [
    "zoompan_in",
    "zoompan_out",
    "pan_horizontal",
    "diagonal_zoom",
    "breathing_zoom",
    "vignette_zoom",
    "color_drift",
    "grain_zoom",
]

# --- Content variety pools ---

STYLE_MODIFIERS = [
    "warm film photography style",
    "soft watercolor illustration style",
    "moody noir black and white style",
    "minimalist line art style",
    "vintage polaroid aesthetic",
    "cinematic golden-hour photography",
    "cozy hand-drawn illustration style",
    "high-contrast editorial photography",
]

SUBJECT_VARIANTS = [
    "a steaming cup of coffee",
    "a rustic bag of whole coffee beans",
    "a charming cafe exterior with an outdoor table",
    "a barista's hands pouring latte art",
    "a coffee cup next to an open journal",
    "a French press and cup on a windowsill",
    "an espresso machine with a fresh shot pulling",
    "a coffee cup with steam rising, seen from above",
]

CAPTION_INTROS = [
    "Coffee time ☕",
    "Morning fuel ☕",
    "A little cafe moment ☕",
    "Brewed just for you ☕",
    "Slow morning vibes ☕",
    "Cafe daydreaming ☕",
    "Warm cup, warm thoughts ☕",
    "Today's coffee ritual ☕",
]

def generate_prompt():
    print("Generating prompt with Groq...")

    style = random.choice(STYLE_MODIFIERS)
    subject = random.choice(SUBJECT_VARIANTS)

    system_instruction = (
        "You are a creative assistant that writes short, vivid prompts for an "
        "AI image generator. Each prompt must feature this subject: "
        f"{subject}. Render it in this style: {style}. "
        "Describe lighting, setting, and mood, under 25 words total. "
        "Do not repeat common phrasing. Return ONLY the prompt text, nothing else."
    )

    last_err = None
    for attempt in range(3):
        try:
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": "Generate one new coffee-themed image prompt."},
                    ],
                    "max_tokens": 300,
                    "temperature": 1.0,
                    "reasoning_effort": "low",
                },
            )
            prompt = res.json()["choices"][0]["message"]["content"].strip().strip('"')
            if not prompt:
                raise ValueError("Groq returned an empty prompt")
            print(f"Generated prompt: {prompt}")
            return prompt
        except Exception as e:
            last_err = e
            print(f"generate_prompt attempt {attempt + 1} failed ({e}); retrying...")
            time.sleep(5)

    print(f"Groq prompt generation failed after retries ({last_err}); using fallback prompt.")
    return random.choice(FALLBACK_PROMPTS)

def generate_image():
    print("Generating image...")
    #prompt = random.choice(PROMPTS)
    prompt = generate_prompt()
    client = InferenceClient(token=HF_TOKEN)
    client.headers["x-use-cache"] = "0"
# model choices:
# - "black-forest-labs/FLUX.1-schnell" (State-of-the-art high quality)
# - "stabilityai/stable-diffusion-xl-base-1.0"
    model_id = "black-forest-labs/FLUX.1-schnell"
    image = client.text_to_image(prompt=prompt,model=model_id)
    image.save(IMAGE_FILENAME)
    print(f"Saved image for prompt: {prompt}")
    return prompt
    

def add_background_music(video_path: str, output_path: str, duration: int) -> str:
    music_files = glob.glob("assets/music/*.mp3")
    print(f"Found {len(music_files)} music file(s) in assets/music/")
    if not music_files:
        print("No music files found — skipping audio overlay.")
        return video_path

    music_path = random.choice(music_files)
    print(f"Adding background music: {music_path}")

    fade_start = max(duration - 1, 0)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-filter_complex",
        f"[1:a]atrim=0:{duration},afade=t=out:st={fade_start}:d=1,volume=0.8[aud]",
        "-map", "0:v",
        "-map", "[aud]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg audio overlay failed:\n{result.stderr}")
        raise RuntimeError("ffmpeg audio overlay failed")
    return output_path
def git_run(*args):
    result = subprocess.run(["git"] + list(args), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def commit_image():
    print("Committing image to repo...")
    git_run("config", "user.name", "coffee-bot")
    git_run("config", "user.email", "coffee-bot@users.noreply.github.com")
    git_run("add", IMAGE_FILENAME)
    commit_result = git_run("commit", "-m", "Daily coffee image")
    if commit_result.returncode != 0:
        print("Nothing to commit or commit failed — aborting.")
        sys.exit(1)
    push_result = git_run("push")
    if push_result.returncode != 0:
        print("Push failed — aborting.")
        sys.exit(1)

    image_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{IMAGE_FILENAME}"
    print(f"Image will be publicly available at: {image_url}")
    return image_url


def remove_image():
    print("Removing image from repo...")
    git_run("rm", IMAGE_FILENAME)
    git_run("commit", "-m", "Remove published image")
    git_run("push")

def remove_video():
    print("Removing video from repo...")
    git_run("rm", VIDEO_FILENAME)
    git_run("commit", "-m", "Remove published video")
    git_run("push")

# ---------- GitHub secret update ----------

def update_github_secret(secret_name, secret_value):
    headers = {"Authorization": f"token {GH_PAT}", "Accept": "application/vnd.github+json"}

    key_res = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/public-key",
        headers=headers,
    )
    key_res.raise_for_status()
    key_data = key_res.json()

    public_key = public.PublicKey(key_data["key"].encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    encrypted_b64 = encoding.Base64Encoder().encode(encrypted).decode("utf-8")

    put_res = requests.put(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/{secret_name}",
        headers=headers,
        json={"encrypted_value": encrypted_b64, "key_id": key_data["key_id"]},
    )
    put_res.raise_for_status()
    print(f"Updated secret {secret_name} in {GITHUB_REPO}")

#def publish_to_pinterest(image_url, caption):
#    print("Publishing to Pinterest...")
#    res = requests.post(
#        "https://api.pinterest.com/v5/pins",
#        headers={"Authorization": f"Bearer {PINTEREST_TOKEN}"},
#        json={
#            "board_id": BOARD_ID,
#            "media_source": {
#                "source_type": "image_url",
#                "url": image_url,
#            },
#            "title": "Coffee Moments ☕",
#            "description": f"{caption} #coffee #cafe #coffeetime",
#        },
#        timeout=30,
#    )
#    return res

# ---------- Tumblr ----------

def refresh_tumblr_token():
    print("Refreshing Tumblr access token...")
    res = requests.post(
        "https://api.tumblr.com/v2/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": TUMBLR_REFRESH_TOKEN,
            "client_id": TUMBLR_CONSUMER_KEY,
            "client_secret": TUMBLR_CONSUMER_SECRET,
        },
        timeout=30,
    )
    if res.status_code != 200:
        print(f"Tumblr refresh error body: {res.text}")
    res.raise_for_status()
    data = res.json()
    print("Tumblr token refreshed.")

    new_refresh_token = data.get("refresh_token")
    if new_refresh_token and new_refresh_token != TUMBLR_REFRESH_TOKEN:
        print("Tumblr issued a new refresh token — updating GitHub secret...")
        try:
            update_github_secret("TUMBLR_REFRESH_TOKEN", new_refresh_token)
        except Exception as e:
            print(f"WARNING: failed to update TUMBLR_REFRESH_TOKEN secret: {e}")
            print("Next run may fail with invalid_grant unless updated manually.")

    return data["access_token"]


def publish_to_tumblr(access_token, image_url, caption):
    print("Publishing to Tumblr...")
    res = requests.post(
        f"https://api.tumblr.com/v2/blog/{TUMBLR_BLOG_NAME}/posts",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "content": [
                {"type": "image", "media": [{"url": image_url}]},
                {"type": "text", "text": caption},
            ],
            "tags": "coffee,cafe,coffeetime",
        },
        timeout=30,
    )
    return res

# ---------- BlSky ----------

def bsky_login():
    print("Logging into Bluesky...")
    res = requests.post(
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        json={"identifier": BSKY_HANDLE, "password": BSKY_APP_PASSWORD},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    return data["accessJwt"], data["did"]


def bsky_upload_image(access_jwt, image_path):
    print("Uploading image to Bluesky...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    res = requests.post(
        "https://bsky.social/xrpc/com.atproto.repo.uploadBlob",
        headers={
            "Authorization": f"Bearer {access_jwt}",
            "Content-Type": "image/png",
        },
        data=image_bytes,
        timeout=60,
    )
    res.raise_for_status()
    return res.json()["blob"]


def publish_to_bluesky(image_path, caption):
    try:
        access_jwt, did = bsky_login()
        blob = bsky_upload_image(access_jwt, image_path)

        post_record = {
            "collection": "app.bsky.feed.post",
            "repo": did,
            "record": {
                "$type": "app.bsky.feed.post",
                "text": caption,
                "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "embed": {
                    "$type": "app.bsky.embed.images",
                    "images": [
                        {"alt": "Coffee at a cafe", "image": blob}
                    ],
                },
            },
        }

        res = requests.post(
            "https://bsky.social/xrpc/com.atproto.repo.createRecord",
            headers={"Authorization": f"Bearer {access_jwt}"},
            json=post_record,
            timeout=30,
        )
        return res
    except Exception as e:
        print(f"Bluesky error: {e}")
        return None

# ----- YouTube --------

def generate_video_from_image(image_path: str, motion_prompt: str, out_path: str) -> str:
    """
    Animate the already-generated FLUX coffee image into a short clip
    and write raw bytes to out_path. Uses huggingface_hub's
    image_to_video, which is the image-conditioned counterpart to the
    text_to_video call and reuses the same HF_TOKEN.
 
    `motion_prompt` should describe the *motion* to add (steam rising,
    slow push-in, light flicker) -- the model keeps the input image's
    subject and composition, it isn't regenerating the scene from text.
    Ignored entirely by SVD, which doesn't take a text prompt -- kept as
    a param so this drops in cleanly if you switch to a prompt-aware
    model like Wan2.2-I2V later.
 
    On the free hf-inference tier the model may need to "wake up" and
    return 503 while it loads into the shared queue; retry a few times
    with a short backoff rather than treating that as a hard failure.
    """
    import time
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError
 
    client = InferenceClient(provider=HF_VIDEO_PROVIDER, api_key=HF_TOKEN)
 
    with open(image_path, "rb") as f:
        input_image = f.read()
 
    kwargs = {"model": HF_VIDEO_MODEL}
    if HF_VIDEO_PROVIDER != "hf-inference":
        # SVD ignores prompt/duration; only pass these to prompt-aware,
        # duration-aware routed models like Wan2.2-I2V.
        kwargs["prompt"] = motion_prompt
        kwargs["duration"] = CLIP_DURATION_SECONDS
 
    last_err = None
    for attempt in range(5):
        try:
            video_bytes = client.image_to_video(input_image, **kwargs)
            with open(out_path, "wb") as f:
                f.write(video_bytes)
            return out_path
        except HfHubHTTPError as e:
            last_err = e
            print(f"image_to_video attempt {attempt + 1} failed ({e}); retrying...")
            time.sleep(15)
    raise last_err
 
 
def to_vertical_short(input_path: str, output_path: str) -> str:
    """
    Force the clip into YouTube Shorts' required shape: 1080x1920 (9:16),
    H.264 + AAC, exactly CLIP_DURATION_SECONDS long.
 
    Most open video models generate landscape or square by default, so
    this pads to vertical with a blurred background copy rather than
    cropping out the subject. SVD's free-tier output also tends to run
    shorter than CLIP_DURATION_SECONDS (fixed ~25 frames), so rather than
    just padding with a frozen last frame, this slows playback (tpad
    only clones the last frame; setpts stretches evenly) to fill the
    target length smoothly. Requires ffmpeg, which ubuntu-latest ships
    with already.
    """
    filter_complex = (
        "[0:v]scale=1080:-2,boxblur=20:5[bg];"
        "[0:v]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        "crop=1080:1920:(iw-1080)/2:(ih-1920)/2,"
        f"tpad=stop_mode=clone:stop_duration={CLIP_DURATION_SECONDS}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(CLIP_DURATION_SECONDS),  # hard cap in case source ran long
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
 
 
# ---------------------------------------------------------------------------
# 2. YouTube upload (OAuth2 refresh token, resumable upload)
# ---------------------------------------------------------------------------
def yt_refresh_access_token() -> str:
    """
    Exchange the long-lived refresh token for a fresh ~1hr access token.
    Unlike Tumblr, YouTube/Google does NOT rotate the refresh token on
    each use once your OAuth consent screen is in "Production" status --
    the same refresh token keeps working indefinitely (until unused for
    6 months, revoked, or your client secret is rotated). Store it once
    as a repo secret and forget about it.
 
    If your consent screen is still in "Testing" status, Google expires
    refresh tokens after 7 days regardless of use -- that will silently
    break an unattended daily cron, so this is worth resolving (submit
    for verification / publish to production) before relying on this.
    """
    res = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": YT_CLIENT_ID,
            "client_secret": YT_CLIENT_SECRET,
            "refresh_token": YT_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    res.raise_for_status()
    return res.json()["access_token"]
 
 
def publish_to_youtube(video_path: str, title: str, description: str, tags=None):
    try:
        access_token = yt_refresh_access_token()

        metadata = {
            "snippet": {
                "title": title[:100],
                "description": description,
                "tags": tags or ["coffee", "cafe", "shorts"],
                "categoryId": "22",
            },
            "status": {
                "privacyStatus": YT_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
            },
        }

        init_res = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos"
            "?uploadType=resumable&part=snippet,status",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
            },
            json=metadata,
            timeout=30,
        )
        if not init_res.ok:
            print(f"YouTube init error body: {init_res.text}")  # <-- add this
        init_res.raise_for_status()
        upload_url = init_res.headers["Location"]

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        upload_res = requests.put(
            upload_url,
            headers={"Content-Type": "video/mp4"},
            data=video_bytes,
            timeout=180,
        )
        if not upload_res.ok:
            print(f"YouTube upload error body: {upload_res.text}")  # <-- and this
        upload_res.raise_for_status()
        return upload_res
    except Exception as e:
        print(f"YouTube error: {e}")
        return None
        

def build_filter(effect_name: str, duration: int, fps: int = 30) -> str:
    total_frames = duration * fps
    filters = {
        "zoompan_in": f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.0007,1.3)':d={total_frames}:s=1080x1920:fps={fps}",
        "zoompan_out": f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='if(eq(on,1),1.3,max(1.001,zoom-0.0007))':d={total_frames}:s=1080x1920:fps={fps}",
        "pan_horizontal": f"scale=1600:1920:force_original_aspect_ratio=increase,crop=1080:1920:x='min(t*40,iw-1080)':y=0",
        "diagonal_zoom": f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.0005,1.2)':x='iw/2-(iw/zoom/2)+t*10':y='ih/2-(ih/zoom/2)+t*5':d={total_frames}:s=1080x1920:fps={fps}",
        "breathing_zoom": f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='1.1+0.05*sin(on/10)':d={total_frames}:s=1080x1920:fps={fps}",
        "vignette_zoom": f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.0007,1.3)':d={total_frames}:s=1080x1920:fps={fps},vignette=PI/4",
        "color_drift": f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,eq=saturation=1.1,zoompan=z='min(zoom+0.0006,1.25)':d={total_frames}:s=1080x1920:fps={fps}",
        "grain_zoom": f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,zoompan=z='min(zoom+0.0007,1.3)':d={total_frames}:s=1080x1920:fps={fps},noise=alls=8:allf=t+u",
    }
    base = filters.get(effect_name, filters["zoompan_in"])
    fade_out_start = max(duration - 1, 0)
    return f"{base},fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out_start}:d=0.5"


def image_to_motion_clip(image_path: str, output_path: str, duration: int = None) -> str:
    duration = duration or CLIP_DURATION_SECONDS
    fps = 30
    effect = random.choice(EFFECTS)
    print(f"Using effect: {effect}")
    vf = build_filter(effect, duration, fps)

    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Fallback motion clip created at {output_path} using '{effect}'")
    return output_path 
# ---------------------------------------------------------------------------
# Glue -- call this with the same image_path and caption your existing
# post_coffee.py already generated and used for Pinterest/Tumblr/Bluesky,
# before that image gets deleted from the repo.
# ---------------------------------------------------------------------------
def run_youtube_short_step(image_path: str, motion_prompt: str, caption: str, all_ok):
    raw_path = "assets/raw_video_temp.mp4"
    vertical_path = VIDEO_FILENAME
    final_path = "assets/generated_video_with_audio.mp4"

    # try:
    #     generate_video_from_image(image_path, motion_prompt, raw_path)
    #     to_vertical_short(raw_path, vertical_path)
    #     print("Used AI-generated motion clip.")
    # except Exception as e:
    #     print(f"AI video generation failed ({e}); falling back to ffmpeg zoompan.")
    #     try:
    #         image_to_motion_clip(image_path, vertical_path, duration=CLIP_DURATION_SECONDS)
    #     except Exception as fallback_err:
    #         print(f"Fallback clip generation also failed: {fallback_err}")
    #         return False

    try:
        image_to_motion_clip(image_path, vertical_path, duration=CLIP_DURATION_SECONDS)
    except Exception as fallback_err:
        print(f"Fallback clip generation failed: {fallback_err}")
        return False

    if os.path.exists(raw_path):
        os.remove(raw_path)

    # Add background music
    try:
        add_background_music(vertical_path, final_path, CLIP_DURATION_SECONDS)
        os.replace(final_path, vertical_path)  # overwrite vertical_path with the audio version
    except Exception as e:
        print(f"Music overlay failed ({e}); uploading silent clip instead.")
        
    # Commit the final video to the repo, same pattern as the image
    print("Committing video to repo...")
    git_run("add", vertical_path)
    commit_result = git_run("commit", "-m", "Daily coffee video")
    if commit_result.returncode == 0:
        push_result = git_run("push")
        if push_result.returncode != 0:
            print("Video push failed — continuing anyway, upload uses local file.")
    else:
        print("Nothing to commit for video (may already match), continuing.")

    title = caption[:95] + " #Shorts"
    res = publish_to_youtube(vertical_path, title, caption)

    if res is not None and res.ok:
        print(f"Uploaded Short: {res.json().get('id')}")
        return all_ok
    else:
        print("YouTube Short upload failed; see error above.")
        return False

def main():
    prompt = generate_image()
    image_url = commit_image()

    time.sleep(30)

    all_ok = True
    caption_intro = random.choice(CAPTION_INTROS)

        # --- Pinterest ---
 #   pin_res = publish_to_pinterest(image_url, f"{caption_intro} {prompt}")
 #   if pin_res.status_code == 201:
 #       print("Pinterest: published successfully:", pin_res.json())
 #   else:
 #       print(f"Pinterest: publish failed ({pin_res.status_code}): {pin_res.text}")
 #       all_ok = False

    # --- Bluesky ---
    bsky_res = publish_to_bluesky(IMAGE_FILENAME, f"{caption_intro} {prompt}")
    if bsky_res is not None and bsky_res.status_code == 200:
        print("Bluesky: published successfully:", bsky_res.json())
    else:
        print(f"Bluesky: publish failed: {bsky_res.text if bsky_res else 'exception before request'}")
        all_ok = False

    # --- Tumblr ---
    try:
        tumblr_access_token = refresh_tumblr_token()
        tumblr_res = publish_to_tumblr(tumblr_access_token, image_url, f"{caption_intro} \n\n{prompt}")
        if tumblr_res.status_code in (200, 201):
            print("Tumblr: published successfully:", tumblr_res.json())
        else:
            print(f"Tumblr: publish failed ({tumblr_res.status_code}): {tumblr_res.text}")
            all_ok = False
    except Exception as e:
        print(f"Tumblr: error during refresh/publish: {e}")
        all_ok = False

    motion_prompt = "steam gently rising from the cup, soft ambient light flicker"
    all_ok = run_youtube_short_step(IMAGE_FILENAME, motion_prompt, f"{caption_intro} #Shorts\n\n{prompt}", all_ok)

    if not all_ok:
        print("At least one platform failed — leaving image in repo for debugging.")
        sys.exit(1)

if __name__ == "__main__":
    main()
