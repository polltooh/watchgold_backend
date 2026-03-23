from rag_site.azure_openai import AzureOpenAIChat
from rag_site.config import Settings
import json
import os
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import sys
from pathlib import Path

# Provide access to sibling modules since we are inside chatgold/feeds
sys.path.append(str(Path(__file__).parent.parent.parent))


# Standard Gold / Silver Feeds
DEFAULT_FEEDS = [
    "https://www.kitco.com/news/category/mining/rss",
    "https://www.fxstreet.com/rss/news",
    "https://www.mining.com/commodity/gold/feed/",
    "https://goldbroker.com/news/rss-feed-40",
]


def fetch_rss_feeds(urls=DEFAULT_FEEDS) -> list[dict]:
    all_articles = []

    import xml.etree.ElementTree as ET
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    for url in urls:
        try:
            print(f"Fetching RSS from {url}...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"Error fetching {url}: HTTP {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, features="html.parser")
            items = soup.find_all("item")

            for item in items:
                title = item.find("title").text if item.find(
                    "title") is not None else "No Title"

                link_node = item.find("link")
                link = "No Link"
                if link_node is not None:
                    # In html.parser, <link> is void, so its text is pushed to next_sibling
                    if link_node.get("href"):
                        link = link_node.get("href")
                    elif link_node.next_sibling:
                        link = str(link_node.next_sibling).strip()

                description = item.find("description").text if item.find(
                    "description") is not None else ""

                # HTML parser lowercase tags usually
                pubDate_node = item.find("pubdate") or item.find("pubDate")
                pubDate = pubDate_node.text if pubDate_node is not None else ""

                # Basic cleaning of description (remove HTML tags)
                clean_desc = BeautifulSoup(description, "html.parser").get_text(
                    separator=" ", strip=True) if description else ""

                all_articles.append({
                    "title": title.strip(),
                    "link": link.strip(),
                    "description": clean_desc.strip(),
                    "pubDate": pubDate.strip(),
                    "source": url
                })
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    return all_articles


def summarize_with_ai(articles: list[dict]) -> list[dict]:
    print("Summarizing feeds with AI...")
    settings = Settings.from_env()
    chat_model = AzureOpenAIChat(
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        chat_model=settings.azure_openai_chat_model,
    )

    articles_text = ""
    valid_articles = [art for art in articles if art.get(
        'link') and art['link'].strip() and art['link'] != "No Link"]
    # Limit to first 100 to avoid token limits
    for idx, art in enumerate(valid_articles[:100]):
        articles_text += f"\n--- Article {idx+1} ---\nTitle: {art['title']}\nLink: {art['link']}\nDescription: {art['description']}\n"

    prompt = f"""
    You are an expert financial analyst. I have gathered the latest RSS feed news articles regarding the gold and silver markets today.
    Your task is to group all related news articles together into the TOP 3 most important news events or themes spanning the market today.
    
    CRITICAL INSTRUCTION: Make absolutely sure that the 3 events are DISTINCT and cover completely DIFFERENT aspects of the gold/silver markets. 
    For example, you must forcefully diversify them (e.g. one event on Macro/Geopolitics, one event on Physical Demand/Retail like India/China, and one event on Mining Companies or Digital/Crypto innovations). DO NOT choose three events that are just different angles of the same macro geopolitical sell-off.
    
    For each of the 3 events, provide:
    1. A VERY CONCISE title (e.g. "Gold Plunges on War Fears")
    2. A ONE SENTENCE summary summarizing the combined related news for that event.
    3. A JSON array containing ALL the original links from the articles that contributed to this event.
    
    You MUST output your response strictly as a JSON array of objects, with no markdown formatting backticks, using this exact schema:
    [
      {{
        "title": "Very concise title",
        "links": ["url1", "url2"],
        "summary": "Your one-sentence summary combining the related news"
      }}
    ]
    
    News Articles:
    {articles_text}
    """

    answer = chat_model.generate_answer(prompt)

    # Try to parse the JSON
    try:
        # Strip potential markdown code blocks
        clean_ans = answer.strip()
        if clean_ans.startswith("```json"):
            clean_ans = clean_ans[7:]
        if clean_ans.startswith("```"):
            clean_ans = clean_ans[3:]
        if clean_ans.endswith("```"):
            clean_ans = clean_ans[:-3]

        ans_json = json.loads(clean_ans.strip())
        if isinstance(ans_json, list) and len(ans_json) > 3:
            ans_json = ans_json[:3]
        return ans_json
    except json.JSONDecodeError as e:
        print(
            f"Failed to parse AI JSON response: {e}\nResponse was:\n{answer}")
        return []


def raw_crawl_article(url: str) -> dict:
    import requests
    from bs4 import BeautifulSoup
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return {"image": "", "text": ""}
        soup = BeautifulSoup(resp.content, "html.parser")

        # Get og:image
        og_image = soup.find("meta", property="og:image")
        image_url = og_image["content"] if og_image and og_image.get(
            "content") else ""

        # Get paragraphs
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text(separator=" ", strip=True)
                         for p in paragraphs])

        return {"image": image_url, "text": text}
    except Exception as e:
        print(f"Crawler error on {url}: {e}")
        return {"image": "", "text": ""}


def generate_detailed_markdown(item: dict, chat_model) -> str:
    print(f"Generating detailed markdown for: {item.get('title')}")
    # Grab up to first 3 links to avoid huge context or slow crawling
    links = item.get("links", [])[:3]

    extracted_data = []
    main_image = ""
    for link in links:
        data = raw_crawl_article(link)
        if data["image"] and not main_image:
            main_image = data["image"]
        extracted_data.append(
            f"Source: {link}\nContent:\n{data['text'][:2000]}")

    combined_text = "\n\n".join(extracted_data)

    img_instruction = f"- If a main image was found, you MUST start your response with: ![]({main_image})\\n" if main_image else ""

    prompt = f"""
    You are an expert financial journalist. Write a comprehensive, detailed markdown news report about the following market event.
    
    Event Title: {item.get('title')}
    Highlights: {item.get('summary')}
    
    Here is the scraped raw content from the source articles:
    {combined_text}
    
    REQUIREMENTS:
    - Write a highly professional, detailed blog post in Markdown format.
    {img_instruction}
    - Structure the report with an introduction, a detailed analysis section using H2/H3 tags, and a conclusion.
    - Omit any website boilerplate, ads code, or navigation text that might have been scraped. Focus strictly on the news.
    - Give both bullish and bearish arguments in the end to replace conclusion section. They should be subtitled with 
      ## Bullish and ## Bearish.
    - Add a "### Sources" section at the end listing the exact links.
    - ONLY output the raw markdown text. Do NOT wrap it in ```markdown blocks.
    """
    answer = chat_model.generate_answer(prompt)
    clean_ans = answer.strip()
    if clean_ans.startswith("```markdown"):
        clean_ans = clean_ans[11:]
    if clean_ans.startswith("```"):
        clean_ans = clean_ans[3:]
    if clean_ans.endswith("```"):
        clean_ans = clean_ans[:-3]

    return clean_ans.strip()


def get_daily_summary() -> list[dict]:
    today_str = datetime.now().strftime("%Y%m%d")

    # Ensure downloads directory exists
    downloads_dir = os.path.join(os.path.dirname(
        __file__), "..", "crawler", "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    summary_file = os.path.join(downloads_dir, f"rss_summary_{today_str}.json")
    raw_file = os.path.join(downloads_dir, f"rss_raw_{today_str}.json")

    # 1. Check if summary already exists for today
    if os.path.exists(summary_file):
        print("Returning cached summary for today...")
        with open(summary_file, "r") as f:
            return json.load(f)

    # 2. Fetch raw feeds
    articles = fetch_rss_feeds()

    # Save raw feeds just in case
    with open(raw_file, "w") as f:
        json.dump(articles, f, indent=2)

    if not articles:
        return []

    # 3. Summarize with AI
    top_summary = summarize_with_ai(articles)

    if isinstance(top_summary, list):
        settings = Settings.from_env()
        chat_model = AzureOpenAIChat(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            chat_model=settings.azure_openai_chat_model,
        )
        for item in top_summary:
            item["detailed_markdown"] = generate_detailed_markdown(
                item, chat_model)

    # 4. Save summary
    if top_summary:
        with open(summary_file, "w") as f:
            json.dump(top_summary, f, indent=2)

    return top_summary


if __name__ == "__main__":
    # Standard test execution
    res = get_daily_summary()
    print(json.dumps(res, indent=2))
