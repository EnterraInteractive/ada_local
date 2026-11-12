import json
from core.inference import runtime as inference_runtime
import datetime
from duckduckgo_search import DDGS

class NewsManager:
    """Manages fetching and curating news for the Briefing dashboard."""

    def __init__(self):
        self.ddgs = DDGS()
        # Simple in-memory cache: {category: {"timestamp": dt, "data": []}}
        self.cache = {}
        self.cache_duration = datetime.timedelta(minutes=15)

    def get_briefing(self, status_callback=None, use_ai: bool = True) -> list:
        """
        Get a curated briefing.
        Fetches 'top' and 'technology' news, then asks AI to pick the best ones.
        """
        # 1. Check cache first
        if status_callback: status_callback("Checking local cache...")
        cache_key = "briefing_ai" if use_ai else "briefing_raw"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # 2. Fetch raw news
        raw_news = []
        try:
            if status_callback: status_callback("Scanning global headlines...")
            # Fetch generic top news
            for r in self.ddgs.news("top news", max_results=5):
                r['category'] = 'Top Stories'
                raw_news.append(r)
            
            if status_callback: status_callback("Retrieving technology sector updates...")
            # Fetch Tech news
            for r in self.ddgs.news("technology news", max_results=5):
                r['category'] = 'Technology'
                raw_news.append(r)
                
            # Fetch Science news
            for r in self.ddgs.news("science breakthrough", max_results=3):
                r['category'] = 'Science'
                raw_news.append(r)

        except Exception as e:
            print(f"Error fetching news from DDGS: {e}")
            # Rate limit or connection error - return empty to trigger fallback
            # In a production app, we might retry with backoff, but for now we fail gracefully.
            return []

        # 3. AI Curation
        curated_news = None
        if use_ai:
            if status_callback: status_callback("AI is reading and curating stories...")
            curated_news = self._curate_with_ai(raw_news)
        
        # 4. Fallback if AI fails: just return raw news formatted
        if not curated_news:
            curated_news = self._format_raw_fallback(raw_news)

        # 5. Save to cache
        self.cache[cache_key] = {
            "timestamp": datetime.datetime.now(),
            "data": curated_news
        }
        
        return curated_news

    def _get_from_cache(self, key: str):
        if key in self.cache:
            entry = self.cache[key]
            if datetime.datetime.now() - entry["timestamp"] < self.cache_duration:
                return entry["data"]
        return None

    def _format_raw_fallback(self, raw_news):
        """Fallback formatting if AI fails."""
        formatted = []
        seen_titles = set()
        
        for item in raw_news:
            if item['title'] in seen_titles:
                continue
            seen_titles.add(item['title'])
            
            formatted.append({
                "title": item.get('title'),
                "source": item.get('source'),
                "date": item.get('date'),
                "category": item.get('category', 'General'),
                "url": item.get('url'),
                "image": item.get('image') # DDGS might return 'image'
            })
        return formatted[:8] # Return top 8

    def _curate_with_ai(self, raw_news):
        """Send raw news to LLM to select and strictly format."""
        
        # Minify valid data for prompt
        # We only need title, source, category to make a decision
        news_input = [
            {"id": i, "title": n.get('title'), "source": n.get('source'), "category": n.get('category')} 
            for i, n in enumerate(raw_news)
        ]

        prompt = f"""
You are an expert News Editor.
Here is a list of raw news articles:
{json.dumps(news_input, indent=2)}

Task:
1. Select the 6 most important and diverse stories.
2. Rewrite the titles to be punchy and short (under 10 words).
3. Assign a category: 'Technology', 'Science', 'Markets', 'Culture', or 'Top Stories'.
4. Return ONLY a JSON array of objects.
   Format: [{{"id": <original_id>, "title": "<new_title>", "category": "<category>"}}]

Do NOT add any markdown or text. Just the JSON array.
"""
        
        try:
            response_parts = inference_runtime.stream_chat(
                [{"role": "user", "content": prompt}],
                max_tokens=900,
                temperature=0.3,
                top_p=0.9,
            )
            content = "".join(response_parts).strip()

            fence = chr(96) * 3
            if content.startswith(fence):
                content = content[len(fence):].strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()
                if content.endswith(fence):
                    content = content[:-len(fence)].strip()

            selected = json.loads(content)
            if not isinstance(selected, list):
                raise ValueError("News curation response was not a JSON array.")

            final_list = []
            for item in selected:
                if not isinstance(item, dict):
                    continue
                original_id = item.get("id")
                if not isinstance(original_id, int) or not (0 <= original_id < len(raw_news)):
                    continue
                original = raw_news[original_id]
                final_list.append({
                    "title": str(item.get("title") or original.get("title") or ""),
                    "source": original.get("source"),
                    "date": original.get("date"),
                    "category": str(item.get("category") or original.get("category") or "General"),
                    "url": original.get("url"),
                    "image": original.get("image"),
                    "body": original.get("body"),
                })
            return final_list or None

        except Exception as e:
            print(f"AI Curation failed: {e}")
            return None

        return None

news_manager = NewsManager()
