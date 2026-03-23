import yaml
import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def main():
    base_dir = Path(__file__).parent
    yaml_path = base_dir / "target_web.yaml"
    downloads_dir = base_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    web_targets = config.get("web", {})

    # Maximum articles to fetch per site initially
    MAX_ARTICLES = 20

    for name, base_url in web_targets.items():
        print(f"[{name}] Starting extraction from homepage: {base_url}")

        # 1. Fetch homepage
        try:
            # Setting a User-Agent to help avoid basic blocks
            headers = {"User-Agent": "Mozilla/5.0"}
            home_resp = requests.get(base_url, headers=headers, timeout=10)
            home_resp.raise_for_status()
        except Exception as e:
            print(f"[{name}] Failed to fetch homepage: {e}")
            continue

        soup = BeautifulSoup(home_resp.text, 'html.parser')

        # 2. Extract links
        domain = urlparse(base_url).netloc
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            # Filter valid HTTP links within the same domain
            parsed_url = urlparse(full_url)
            if parsed_url.scheme in ('http', 'https') and domain in parsed_url.netloc:
                # Remove URL fragments to avoid duplicates
                clean_url = full_url.split('#')[0]
                # Filter out obvious non-html resources and the homepage itself
                if clean_url != base_url and not any(clean_url.lower().endswith(ext) for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.svg', '.zip', '.csv', '.gif']):
                    links.add(clean_url)

        # 3. Download and extract text
        articles_data = {}
        links_list = list(links)[:MAX_ARTICLES]

        print(
            f"[{name}] Found {len(links)} unique links on homepage. Fetching top {len(links_list)}...")

        for url in links_list:
            print(f"  Fetching {url}...")
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()

                article_soup = BeautifulSoup(resp.text, 'html.parser')
                import re
                
                # 1. Try to find the main content area to avoid nav/footers
                main_content = article_soup.find('main') or article_soup.find('article') or article_soup.find('div', class_=re.compile('(content|article|body|post)', re.I))
                
                if main_content:
                    paragraphs = main_content.find_all('p')
                    if not paragraphs:
                        # Fallback for sites like TPM that don't use <p> tags
                        raw_texts = main_content.get_text(separator='\n\n', strip=True).split('\n\n')
                        from bs4 import Tag
                        paragraphs = []
                        for t in raw_texts:
                            new_p = soup.new_tag("p")
                            new_p.string = t
                            paragraphs.append(new_p)
                else:
                    paragraphs = article_soup.find_all('p')
                
                cleaned_paragraphs = []
                for p in paragraphs:
                    p_text = p.get_text(separator=' ', strip=True)
                    
                    # Filter out very short paragraphs (likely UI elements, links, or dates)
                    if len(p_text) < 40:
                        continue
                        
                    # Filter out common boilerplate phrases
                    lower_text = p_text.lower()
                    if any(bad in lower_text for bad in [
                        "cookie", "copyright", "all rights reserved", 
                        "javascript is disabled", "please enable javascript",
                        "subscribe to our newsletter", "follow us on"
                    ]):
                        continue
                        
                    cleaned_paragraphs.append(p_text)

                text = "\n\n".join(cleaned_paragraphs)
                
                # Further ensure the article is actually useful (contains relevant keywords)
                if text and ("gold" in text.lower() or "silver" in text.lower() or "metal" in text.lower() or "bullion" in text.lower() or "coin" in text.lower()):
                    articles_data[url] = text
                else:
                    print(f"  [WARN] No relevant gold/silver text found for {url}")
            except Exception as e:
                print(f"  [ERROR] Failed {url}: {e}")

        # 4. Save to JSON
        json_path = downloads_dir / f"{name}_articles.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=2)

        print(f"[{name}] Saved {len(articles_data)} articles to {json_path}\n")


if __name__ == "__main__":
    main()
