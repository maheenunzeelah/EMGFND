# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
# %%
def google_image_search(query, num_images=5):
    # Set up Chrome with WebDriver Manager
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # run in background
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # Step 1: Go to Google Images
        driver.get("https://images.google.com/")
        time.sleep(1)

        # Step 2: Enter the search query
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        # Step 3: Extract image URLs
        image_sections = driver.find_elements(By.XPATH, '//div[@id="search"]//div[@data-attrid="images universal"]')[:num_images]
        
        all_section_images=[]
        for idx, section in enumerate(image_sections, 1):
        # Within each section, find all img tags with src
           imgs = section.find_elements(By.XPATH, './/div[2]//img[@src]')
           image_urls = [img.get_attribute("src").strip()[:5] for img in imgs if img.get_attribute("src")][0]
           all_section_images.append(image_urls)

    

    finally:
        driver.quit()
    return all_section_images 

# %%
img= google_image_search("Donald Trump",2)
# %%
print(img)
# %%
