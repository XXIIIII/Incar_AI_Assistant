"""
Web Search Module for Incar AI Assistant

Provides web search functionality to enhance travel recommendations with
real-time information from Google Search API.

Note: Currently disabled in main application for stability,
but can be enabled by uncommenting relevant sections in tutor.py
"""

import os
import requests
import re
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool


def initialize_search_api():
    """
    Initialize Google Search API with environment variables.
    
    Returns:
        GoogleSearchAPIWrapper: Configured search wrapper
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Try to get from environment variables first
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Fallback to development keys (should be removed for production)
    if not cse_id:
        print("GOOGLE_CSE_ID not found in environment, using development key")
        os.environ["GOOGLE_CSE_ID"] = DEVELOPMENT_CSE_ID
    
    if not api_key:
        print("GOOGLE_API_KEY not found in environment, using development key")
        os.environ["GOOGLE_API_KEY"] = DEVELOPMENT_API_KEY
    
    return GoogleSearchAPIWrapper()


def get_top_search_results(query, num_results=3):
    """
    Get top search results for a given query.
    
    Args:
        query (str): Search query string
        num_results (int): Number of results to return (default: 3)
        
    Returns:
        list: List of search result dictionaries with title, link, snippet
    """
    try:
        search = initialize_search_api()
        results = search.results(query, num_results)
        print(f"Found {len(results)} search results for: '{query}'")
        return results
    except Exception as e:
        print(f"Search API error: {e}")
        return []


def extract_web_content(url, timeout=10):
    """
    Extract text content from a webpage URL.
    
    Args:
        url (str): Target webpage URL
        timeout (int): Request timeout in seconds
        
    Returns:
        str: Extracted text content or None if failed
        
    Note: BeautifulSoup import is commented out - uncomment if needed
    """
    try:
        # Send GET request with timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            # Note: Uncomment the following lines if BeautifulSoup is available
            # from bs4 import BeautifulSoup
            # soup = BeautifulSoup(response.content, 'html.parser')
            # text = soup.get_text()
            # return text
            
            # Simple text extraction without BeautifulSoup
            return response.text
        else:
            print(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"Timeout while fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return None


def extract_meaningful_text(text):
    """
    Extract meaningful words from text using regex patterns.
    
    Supports both English and Chinese text extraction.
    
    Args:
        text (str): Input text to process
        
    Returns:
        list: List of meaningful words/characters
    """
    # Pattern for English words, Chinese characters, and numbers
    pattern = r'(?:[a-zA-Z]+|[\u4E00-\u9FFF]+|\d+)'
    words = re.findall(pattern, text)
    return words


def web_search(user_query):
    """
    Perform web search and return concatenated snippets.
    
    This is the main function used by the AI tutor to get additional
    context for travel recommendations.
    
    Args:
        user_query (str): User's travel query
        
    Returns:
        str: Concatenated search result snippets
    """
    try:
        # Create search tool
        search_tool = Tool(
            name="Google Search Snippets",
            description="Search Google for recent travel information.",
            func=get_top_search_results,
        )
        
        # Execute search
        raw_results = search_tool.run(user_query)
        
        # Concatenate snippets
        combined_text = ''
        for result in raw_results:
            if isinstance(result, dict) and 'snippet' in result:
                combined_text += result['snippet'] + '\n'
        
        print(f"Extracted {len(combined_text)} characters from search results")
        return combined_text
        
    except Exception as e:
        print(f"Web search error: {e}")
        return ""


def search_travel_info(destination, query_type="attractions"):
    """
    Specialized search function for travel-related queries.
    
    Args:
        destination (str): Travel destination
        query_type (str): Type of search ('attractions', 'restaurants', 'hotels', etc.)
        
    Returns:
        str: Formatted search results
    """
    # Construct specialized query
    query_templates = {
        'attractions': f"{destination} 景點推薦 旅遊攻略",
        'restaurants': f"{destination} 美食餐廳推薦",
        'hotels': f"{destination} 住宿酒店推薦",
        'transportation': f"{destination} 交通資訊 怎麼去",
        'culture': f"{destination} 歷史文化 博物館"
    }
    
    search_query = query_templates.get(query_type, f"{destination} 旅遊資訊")
    
    print(f"Searching for {query_type} in {destination}")
    return web_search(search_query)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Web Search Module")
    
    # Test basic search
    test_query = "台北一日遊景點推薦"
    results = web_search(test_query)
    print(f"Search results preview: {results[:200]}...")
    
    # Test specialized travel search
    travel_info = search_travel_info("大阪", "attractions")
    print(f"Travel info preview: {travel_info[:200]}...")
    
    print("Web search test completed")