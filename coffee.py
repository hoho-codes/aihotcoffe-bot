import os
import time
import random
import subprocess
import sys
from huggingface_hub import InferenceClient
import requests

# --- Config from environment/secrets ---
HF_TOKEN = os.environ["HF_TOKEN"]
#PINTEREST_TOKEN = os.environ["PINTEREST_TOKEN"]
#BOARD_ID = os.environ["PINTEREST_BOARD_ID"]
GITHUB_REPO = os.environ["GITHUB_REPOSITORY"]  # auto-set by GitHub Actions, e.g. "user/repo"
GITHUB_BRANCH = os.environ.get("GITHUB_REF_NAME", "main")
IMAGE_FILENAME = "generated_image.png"

PROMPTS = [
    "a steaming latte on a rustic wooden cafe table, morning sunlight, cozy atmosphere",
    "a cappuccino with latte art next to an open book, cozy cafe interior, soft light",
    "an iced coffee on a marble table, city cafe window in background, bright daylight",
    "a pour-over coffee setup on a cafe counter, warm afternoon light, minimalist",
    "a flat white on a cafe table with a croissant, natural window light",
    "a coffee cup on an outdoor cafe table, european street in background, golden hour",
]


def generate_image():
    print("Generating image...")
    prompt = random.choice(PROMPTS)
    client = InferenceClient(token=HF_TOKEN)
    client.headers["x-use-cache"] = "0"
# model choices:
# - "black-forest-labs/FLUX.1-schnell" (State-of-the-art high quality)
# - "stabilityai/stable-diffusion-xl-base-1.0"
    model_id = "black-forest-labs/FLUX.1-schnell"
    image = client.text_to_image(prompt=prompt,)
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


def main():
    prompt = generate_image()
    image_url = commit_image()

    # Give the CDN a moment to catch up before Pinterest fetches it
    time.sleep(5)

#    res = publish_to_pinterest(image_url, prompt)
#
#    if res.status_code == 201:
#        print("Pin published successfully:", res.json())
#        remove_image()
#    else:
#        print(f"Pinterest publish failed ({res.status_code}): {res.text}")
#        print("Leaving image in repo for debugging — remove manually if needed.")
#        sys.exit(1)


if __name__ == "__main__":
    main()
