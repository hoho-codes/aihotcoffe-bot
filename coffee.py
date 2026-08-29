import os
import time
import random
import subprocess
import sys
from huggingface_hub import InferenceClient
import requests
from nacl import encoding, public

# --- Config from environment/secrets ---
HF_TOKEN = os.environ["HF_TOKEN"]
#PINTEREST_TOKEN = os.environ["PINTEREST_TOKEN"]
#BOARD_ID = os.environ["PINTEREST_BOARD_ID"]
GITHUB_REPO = os.environ["GITHUB_REPOSITORY"]  # auto-set by GitHub Actions, e.g. "user/repo"
GITHUB_BRANCH = os.environ.get("GITHUB_REF_NAME", "coffee")
IMAGE_FILENAME = "images/generated_image.png"

TUMBLR_CONSUMER_KEY = os.environ["TUMBLR_CONSUMER_KEY"]
TUMBLR_CONSUMER_SECRET = os.environ["TUMBLR_CONSUMER_SECRET"]
TUMBLR_REFRESH_TOKEN = os.environ["TUMBLR_REFRESH_TOKEN"]
TUMBLR_BLOG_NAME = os.environ["TUMBLR_BLOG_NAME"]

BSKY_HANDLE = os.environ["BSKY_HANDLE"]
BSKY_APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]

GH_PAT = os.environ["GH_PAT"]  # Personal Access Token with repo scope, for updating secrets
GITHUB_REPO = os.environ["GITHUB_REPOSITORY"]
GITHUB_BRANCH = os.environ.get("GITHUB_REF_NAME", "main")

PROMPTS = [
    "a steaming latte on a rustic wooden cafe table, morning sunlight, cozy atmosphere",
    "a cappuccino with latte art next to an open book, cozy cafe interior, soft light",
    "an iced coffee on a marble table, city cafe window in background, bright daylight",
    "a pour-over coffee setup on a cafe counter, warm afternoon light, minimalist",
    "a flat white on a cafe table with a croissant, natural window light",
    "a coffee cup on an outdoor cafe table, european street in background, golden hour",
]

def generate_prompt():
    print("Generating prompt with AI...")
    client = InferenceClient(token=HF_TOKEN)
    
    system_instruction = (
        "You are a creative assistant that writes short, vivid prompts for an "
        "AI image generator. Each prompt must be a single coffee/cafe themed "
        "scene, under 25 words, describing lighting, setting, and mood. "
        "Must include a coffee cup/glass in the image. Do not repeat common phrasing. "
        "Return ONLY the prompt text, nothing else."
    )
    
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "Generate one new coffee-themed image prompt."},
        ],
        model="meta-llama/Llama-3.1-8B-Instruct",  # or another instruct model available on HF Inference
        max_tokens=60,
        temperature=1.0,
    )
    
    prompt = response.choices[0].message.content.strip().strip('"')
    print(f"Generated prompt: {prompt}")
    return prompt

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

def main():
    prompt = generate_image()
    image_url = commit_image()

    # Give the CDN a moment to catch up before Pinterest fetches it
    time.sleep(30)

    all_ok = True

    # --- Pinterest ---
 #   pin_res = publish_to_pinterest(image_url, prompt)
 #   if pin_res.status_code == 201:
 #       print("Pinterest: published successfully:", pin_res.json())
 #   else:
 #       print(f"Pinterest: publish failed ({pin_res.status_code}): {pin_res.text}")
 #       all_ok = False

    # --- Bluesky ---
    bsky_res = publish_to_bluesky(IMAGE_FILENAME, f"Coffee time ☕ {prompt}")
    if bsky_res is not None and bsky_res.status_code == 200:
        print("Bluesky: published successfully:", bsky_res.json())
    else:
        print(f"Bluesky: publish failed: {bsky_res.text if bsky_res else 'exception before request'}")
        all_ok = False

    # --- Tumblr ---
    try:
        tumblr_access_token = refresh_tumblr_token()
        tumblr_res = publish_to_tumblr(tumblr_access_token, image_url, f"Coffee time ☕ \n\n{prompt}")
        if tumblr_res.status_code in (200, 201):
            print("Tumblr: published successfully:", tumblr_res.json())
        else:
            print(f"Tumblr: publish failed ({tumblr_res.status_code}): {tumblr_res.text}")
            all_ok = False
    except Exception as e:
        print(f"Tumblr: error during refresh/publish: {e}")
        all_ok = False

    if not all_ok:
        print("At least one platform failed — leaving image in repo for debugging.")
        sys.exit(1)

if __name__ == "__main__":
    main()
