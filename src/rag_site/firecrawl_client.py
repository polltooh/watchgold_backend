from firecrawl import Firecrawl


class FirecrawlCrawler:
    def __init__(self, api_key: str) -> None:
        self.client = Firecrawl(api_key=api_key)

    @staticmethod
    def _normalize_pages(result) -> list[dict]:
        if isinstance(result, list):
            return [item if isinstance(item, dict) else item.__dict__ for item in result]

        if isinstance(result, dict):
            if isinstance(result.get("data"), list):
                return [
                    item if isinstance(item, dict) else item.__dict__
                    for item in result["data"]
                ]
            return [result]

        if hasattr(result, "__dict__"):
            obj = result.__dict__
            if isinstance(obj.get("data"), list):
                return [
                    item if isinstance(item, dict) else item.__dict__
                    for item in obj["data"]
                ]
            return [obj]

        return []

    def crawl_site(self, start_url: str, limit: int = 20) -> list[dict]:
        result = self.client.crawl(
            start_url,
            limit=limit,
            scrape_options={"formats": ["markdown"]},
        )
        return self._normalize_pages(result)
