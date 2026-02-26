from dotenv import load_dotenv
import os
import sys

# Ensure app directory is in path
sys.path.append(os.getcwd())

load_dotenv()

from app.config import get_settings

settings = get_settings()

print(f"ENABLE_CHATGPT (env): {os.getenv('ENABLE_CHATGPT')}")
print(f"settings.enable_chatgpt: {settings.enable_chatgpt}")
print(f"scraper_pipelines: {settings.scraper_pipelines}")
print(f"scraper_url_chatgpt: {settings.scraper_url_chatgpt}")
