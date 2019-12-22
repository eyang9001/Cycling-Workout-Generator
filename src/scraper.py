from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

# used https://medium.com/the-andela-way/introduction-to-web-scraping-using-selenium-7ec377a8cf72 as original template

def scrape_TR(user_log, user_pass):
    timeout=20
    max_downloads=100

    option = webdriver.ChromeOptions()
    option.add_argument(" - incognito")

    # download the driver from https://chromedriver.chromium.org/downloads
    browser = webdriver.Chrome(executable_path='/Users/eggfooyang/Downloads/chromedriver', options=option)
    browser.get("https://www.trainerroad.com/login")

    # Sign in
    username = browser.find_element_by_id("Username")
    password = browser.find_element_by_id("Password")
    username.send_keys(user_log)
    password.send_keys(user_pass)
    browser.find_element_by_class_name("global-form__submit").click()

    # go to rides list
    browser.get(f"https://www.trainerroad.com/career/{user_log}/rides")
    WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//a[@class='training-item training-item--past-ride']")))

    # iterate through each ride
    for i in range(max_downloads):
        # find all elements of past rides
        rides_elements = browser.find_elements_by_xpath("//a[@class='training-item training-item--past-ride']")
        # if need to load more rides
        if len(rides_elements) == i:
            browser.find_element_by_xpath("//a[@id='loadMore']").click()
            time.sleep(1)
            rides_elements = browser.find_elements_by_xpath("//a[@class='training-item training-item--past-ride']")
        ride = rides_elements[i]
        ride_link = ride.get_attribute("href")
        # Check if the ride was from strava or not. Only open up non-strava ones
        test_strava = ride.find_elements_by_xpath(".//span[@class='info--sync-source']")
        if len(test_strava) == 0:
            print('found')
            browser.get(ride_link)
            # wait for ride to load
            try:
                WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.CLASS_NAME,"past-ride__pr-chart")))
            except TimeoutException:
                print("Timed out waiting for page to load")
            menu = browser.find_element_by_class_name("more-dropdown-wrapper")
            menu.click()
            WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//li[@id='mp-more-ride--download']")))
            dl_button = browser.find_element_by_xpath("//li[@id='mp-more-ride--download']")
            dl_button.click()
            time.sleep(2)
            browser.back()
            WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//a[@class='training-item training-item--past-ride']")))
