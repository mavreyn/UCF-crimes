from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup

TARGET_CLASS = 'fpqsoc'

def selenium_scrape(address: str, driver_path: str) -> str | None:
    '''
    Uses Selenium to get the top result in 'Most Popular Places at this Location' subsection.
    Takes a while tho.
    '''

    # prepare a google search url for a request
    url = 'https://www.google.com/search?q=' + address.replace(' ', '+')

    # Set up Selenium webdriver
    options = webdriver.EdgeOptions()
    options.add_argument('--headless')
    service = Service(driver_path)
    driver = webdriver.Edge(service=service, options=options)

    # Get the page and wait for the js to run
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    try:
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, TARGET_CLASS)))
    except:
        return None

    # Get the new HTML after the js has run
    html = driver.page_source
    driver.quit()

    # Now we can parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    results = soup.find_all(class_=TARGET_CLASS)

    return results[0].text

