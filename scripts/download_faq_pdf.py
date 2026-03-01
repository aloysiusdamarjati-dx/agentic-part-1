"""Download FAQ Dexa Medica PDF from upstream repository."""
import os
import urllib.request

FAQ_URL = "https://github.com/hizkiafebianto/revou-gen-ai-tutorial/raw/main/docs/FAQ%20Dexa%20Medica.pdf"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
OUTPUT_PATH = os.path.join(DOCS_DIR, "FAQ Dexa Medica.pdf")


def download_faq_pdf():
    """Download the FAQ PDF to docs folder."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(FAQ_URL, OUTPUT_PATH)
        print(f"Downloaded FAQ PDF to {OUTPUT_PATH}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        print(f"Please manually download from: {FAQ_URL}")
        print(f"and save to: {os.path.abspath(OUTPUT_PATH)}")
        return False


if __name__ == "__main__":
    download_faq_pdf()
