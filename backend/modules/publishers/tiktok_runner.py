import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import time
import traceback
from pathlib import Path


def patch_tiktok_uploader():
    try:
        from playwright.sync_api import Page
        from tiktok_uploader import config
        from tiktok_uploader.auth import AuthBackend
        import tiktok_uploader.upload as tu_upload

        orig_resolve = AuthBackend._resolve_cookies

        def patched_resolve_cookies(self):
            cookies = orig_resolve(self)
            for c in cookies:
                if "domain" not in c and "url" not in c:
                    c["domain"] = ".tiktok.com"
                    c["path"] = "/"
            return cookies

        AuthBackend._resolve_cookies = patched_resolve_cookies

        def robust_go_to_upload(page: Page) -> None:
            upload_url = str(config.paths.upload)
            if page.url != upload_url:
                page.goto(upload_url)
            else:
                page.reload()
            page.on("dialog", lambda dialog: dialog.accept())
            try:
                page.wait_for_selector("#root, #app, input[type='file'], .upload-container", timeout=45000)
            except Exception:
                pass

        tu_upload._go_to_upload = robust_go_to_upload

        def robust_set_description(page: Page, description: str) -> None:
            if not description:
                return
            try:
                desc_locator = page.locator(f"xpath={config.selectors.upload.description}")
                desc_locator.wait_for(state="visible", timeout=15000)
                desc_locator.click()
                page.keyboard.press("Control+a")
                page.keyboard.press("Backspace")
                time.sleep(0.3)
                # Ensure Draft.js/contenteditable clears any pre-filled filename completely
                try:
                    desc_locator.fill("")
                except Exception:
                    page.keyboard.press("Control+a")
                    page.keyboard.press("Delete")
                time.sleep(0.3)

                words = [w for w in description.split() if w]
                for word in words:
                    if word.startswith("#"):
                        desc_locator.press_sequentially(word, delay=40)
                        time.sleep(0.3)
                        try:
                            mention_box = page.locator(f"xpath={config.selectors.upload.mention_box}")
                            mention_box.wait_for(state="visible", timeout=2000)
                            desc_locator.press("Enter")
                        except Exception:
                            pass
                    else:
                        desc_locator.press_sequentially(word + " ", delay=20)
            except Exception:
                try:
                    desc_locator = page.locator(f"xpath={config.selectors.upload.description}")
                    desc_locator.fill(description)
                except Exception:
                    pass

        tu_upload._set_description = robust_set_description

        def robust_post_video(page: Page) -> None:
            post_btn = None
            selectors = [
                f"xpath={config.selectors.upload.post}",
                "button[data-e2e='post_video_button']",
                "button:has-text('Post')",
                ".TUXButton--primary:has-text('Post')",
            ]
            for sel in selectors:
                try:
                    candidate = page.locator(sel).first
                    if candidate.is_visible(timeout=3000):
                        post_btn = candidate
                        break
                except Exception:
                    pass

            if not post_btn:
                post_btn = page.locator(f"xpath={config.selectors.upload.post}")

            for _ in range(60):
                try:
                    if post_btn.get_attribute("data-disabled") != "true" and not post_btn.is_disabled():
                        break
                except Exception:
                    pass
                time.sleep(2)

            try:
                post_btn.scroll_into_view_if_needed()
                post_btn.click()
            except Exception:
                try:
                    page.evaluate('document.querySelector("button[data-e2e=\'post_video_button\']").click()')
                except Exception:
                    pass

            time.sleep(3)
            for _ in range(3):
                for modal_text in ["Post now", "Post anyway", "Continue", "Confirm"]:
                    try:
                        modal_btn = page.locator(f"//button[contains(., '{modal_text}')]").first
                        if modal_btn.is_visible(timeout=1500):
                            modal_btn.click()
                            time.sleep(2)
                    except Exception:
                        pass

            root_dir = Path(__file__).resolve().parents[3]
            screenshot_dir = root_dir / "storage" / "data"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / "tiktok_last_post_screen.png"

            confirmed = False
            success_texts = [
                "Your video has been uploaded",
                "Your video is uploaded to TikTok",
                "Your video is uploaded",
                "Manage your posts",
                "Upload another video",
                "Upload another",
                "Video published",
                "Your video is being processed",
                "View profile",
            ]

            for _ in range(35):
                current_url = page.url.lower()
                if any(part in current_url for part in ["creator-center/content", "tiktokstudio/content", "/profile", "manage"]):
                    confirmed = True
                    break
                for st in success_texts:
                    try:
                        loc = page.locator(f"//*[contains(text(), '{st}')]").first
                        if loc.is_visible(timeout=300):
                            confirmed = True
                            break
                    except Exception:
                        pass
                if confirmed:
                    break
                time.sleep(2)

            try:
                page.screenshot(path=str(screenshot_path))
            except Exception:
                pass

            if not confirmed:
                try:
                    if not post_btn.is_visible():
                        confirmed = True
                except Exception:
                    pass

            if not confirmed:
                raise RuntimeError("TikTok upload confirmation timed out.")

        tu_upload._post_video = robust_post_video
    except Exception as e:
        print(f"[TikTokRunner] Monkeypatch warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Isolated TikTok session uploader")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--caption", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--cookies-path", default="")
    parser.add_argument("--browser", default="edge")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(json.dumps({"success": False, "error": f"Video file not found: {args.video_path}"}))
        sys.exit(1)

    sess = args.session_id.strip() or None
    cookies = args.cookies_path.strip() if args.cookies_path and os.path.isfile(args.cookies_path) else ""
    if not sess and not cookies:
        print(json.dumps({"success": False, "error": "No TikTok session_id or cookies file provided."}))
        sys.exit(1)

    try:
        patch_tiktok_uploader()
        from tiktok_uploader.upload import TikTokUploader

        uploader = TikTokUploader(
            sessionid=sess,
            cookies=cookies,
            browser=args.browser,
            headless=args.headless,
        )
        ok = uploader.upload_video(filename=args.video_path, description=args.caption)
        if ok:
            print(json.dumps({"success": True, "browser": args.browser, "caption": args.caption}))
            sys.exit(0)
        print(json.dumps({"success": False, "error": "TikTok upload returned false."}))
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc), "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
