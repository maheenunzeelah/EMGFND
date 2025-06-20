
#%%
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import concurrent.futures
from typing import List
#%%
cached_results = {}
#%%
def setup_driver():
    """Setup Chrome driver with optimized options"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def google_image_search_single(query: str, num_images: int = 5) -> List[str]:
    """Optimized image search using your approach"""
    driver = setup_driver()
    
    try:
        print(f"Searching for: {query}")
        
        # Navigate to Google Images
        driver.get("https://images.google.com/")
        
        # Use WebDriverWait for responsive waiting
        wait = WebDriverWait(driver, 15)
        
        # Find and use search box
        search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for search results to load
        wait.until(EC.presence_of_element_located((By.XPATH, '//div[@id="search"]')))
        
        # Your approach: Find image sections
        image_sections = driver.find_elements(By.XPATH, '//div[@id="search"]//div[@data-attrid="images universal"]')[:num_images]
        
        all_section_images = []
        for idx, section in enumerate(image_sections, 1):
            try:
                # Within each section, find all img tags with src
                imgs = section.find_elements(By.XPATH, './/div[2]//img[@src]')
                if imgs:
                    # Get the first valid image URL
                    image_urls = [img.get_attribute("src").strip() for img in imgs if img.get_attribute("src")]
                    if image_urls:
                        all_section_images.append(image_urls[0])
            except Exception as e:
                print(f"Error processing section {idx}: {e}")
                continue
        
        print(f"Found {len(all_section_images)} images for '{query}'")
        return all_section_images
        
    except TimeoutException:
        print(f"Timeout occurred for query: {query}")
        return []
    except Exception as e:
        print(f"Error occurred for query {query}: {str(e)}")
        return []
    finally:
        driver.quit()

def google_image_search(query: str, num_images: int = 5) -> List[str]:
    """Drop-in replacement for your original function"""
    return google_image_search_single(query, num_images)

def google_image_search_concurrent(queries: List[str], num_images: int = 5, max_workers: int = 4) -> dict:
    """Search multiple queries concurrently"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_query = {
            executor.submit(google_image_search_single, query, num_images): query 
            for query in queries
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                results[query] = future.result()
            except Exception as exc:
                print(f"Query {query} generated an exception: {exc}")
                results[query] = []
    
    return results

def search_multiple_queries(queries: List[str], num_images: int = 5) -> dict:
    """Search multiple queries at once"""
    return google_image_search_concurrent(queries, num_images)

# Test function
def test_search():
    """Test the search functionality"""
    test_queries = ["cats", "dogs", "mountains"]
    
    print("Testing single queries:")
    for query in test_queries:
        start_time = time.time()
        results = google_image_search(query, 3)
        end_time = time.time()
        
        print(f"\nQuery: '{query}'")
        print(f"Time: {end_time - start_time:.2f} seconds")
        print(f"Found: {len(results)} images")
        for i, url in enumerate(results, 1):
            print(f"  {i}: {url}")
    
    print("\n" + "="*50)
    print("Testing concurrent queries:")
    start_time = time.time()
    all_results = search_multiple_queries(test_queries, 3)
    end_time = time.time()
    
    print(f"Total time: {end_time - start_time:.2f} seconds")
    for query, urls in all_results.items():
        print(f"{query}: {len(urls)} images")
        for i, url in enumerate(urls, 1):
            print(f"  {i}: {url}")

# Cache for storing search results
image_cache = []

def search_images_with_cache(query, news_year, num_images=1):
    """
    Search for images using multiple queries with caching.
    
    Args:
        query: List of query parts or single query string
        news_year: Year to append to queries
        num_images: Number of images per query part
    
    Returns:
        Flattened numpy array of image URLs
    """
    import numpy as np
    
    # Ensure query is a list
    if isinstance(query, str):
        query = [query]
    
    ref_images = []
    queries_to_search = []
    
    
    # Step 1: Check cache for each query part
    for part in query:
        full_query = str(part) + ' ' + str(news_year)
        print(f"Processing query part: {part}")
        
        # Check if image already exists in the cache
        cached = next((item['base64_image'] for item in image_cache if item['query'] == full_query), None)
        
        if cached:
            print(f"Using cached image for query: {full_query}")
            cached_results[full_query] = cached
        else:
            queries_to_search.append(full_query)
    
    # Step 2: Search for non-cached queries simultaneously
    search_results = {}
    if queries_to_search:
        print(f"Searching for {len(queries_to_search)} new queries simultaneously...")
        search_results = google_image_search_concurrent(queries_to_search, num_images)
        
        # Cache the new results
        for search_query, urls in search_results.items():
            image_cache.append({'query': search_query, 'base64_image': urls})
            print(f"Cached {len(urls)} images for query: {search_query}")
    
    # Step 3: Combine cached and new results in original order
    for part in query:
        full_query = str(part) + ' ' + str(news_year)
        
        if full_query in cached_results:
            if isinstance(cached_results[full_query], list):
                ref_images.extend(cached_results[full_query])
            else:
                ref_images.append(cached_results[full_query])
        elif full_query in search_results:
            if isinstance(search_results[full_query], list):
                ref_images.extend(search_results[full_query])
            else:
                ref_images.append(search_results[full_query])
        else:
            pass  # Skip if no results found
    
    # Convert to numpy array (now all elements are individual URLs)
    ref_images = np.array(ref_images)
    
    return ref_images

def search_images_optimized(query, news_year, num_images=1):
    """
    Optimized version that processes all queries at once.
    Best performance when you have multiple query parts.
    """
    import numpy as np
    
    # Ensure query is a list
    if isinstance(query, str):
        query = [query]
    
    # Create full queries with year
    full_queries = [str(part) + ' ' + str(news_year) for part in query]
    
    # Check cache for all queries
    cached_results = {}
    queries_to_search = []
    
    for full_query in full_queries:
        cached = next((item['base64_image'] for item in image_cache if item['query'] == full_query), None)
        if cached:
            print(f"Using cached image for query: {full_query}")
            cached_results[full_query] = cached
        else:
            queries_to_search.append(full_query)
    
    # Search all non-cached queries at once
    if queries_to_search:
        print(f"Searching {len(queries_to_search)} queries simultaneously...")
        search_results = google_image_search_concurrent(queries_to_search, num_images)
        
        # Cache new results
        for search_query, urls in search_results.items():
            image_cache.append({'query': search_query, 'base64_image': urls})
            cached_results[search_query] = urls
    
    # Return results in original order - flatten properly
    ref_images = []
    for full_query in full_queries:
        if full_query in cached_results:
            if isinstance(cached_results[full_query], list):
                ref_images.extend(cached_results[full_query])
            else:
                ref_images.append(cached_results[full_query])
    
    return np.array(ref_images)

def clear_image_cache():
    """Clear the image cache"""
    global image_cache
    image_cache = []
    print("Image cache cleared")

def get_cache_stats():
    """Get statistics about the cache"""
    total_queries = len(image_cache)
    total_images = sum(len(item['base64_image']) for item in image_cache)
    print(f"Cache contains {total_queries} queries with {total_images} total images")
    return {'queries': total_queries, 'images': total_images}


# Example 1: Your original usage pattern
# query_parts = ["cats", "dogs", "birds"]
# news_year = 2024

# # Example 3: Mixed cached and new queries
# print("\n=== Example 3: Mixed queries ===")
# mixed_queries = ["cats", "dogs", "elephants", "tigers"]  # Some cached, some new
# start_time = time.time()
# results3 = search_images_with_cache(mixed_queries, news_year, num_images=1)
# print(f"Time taken: {time.time() - start_time:.2f} seconds")
# print(f"Total images found: {len(results3)}")

# # Cache stats
# print("\n=== Cache Statistics ===")
# get_cache_stats()
# %%
