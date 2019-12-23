# Heart Rate Cycling Analysis & Workout Generator

This is a passion project of mine to utilize my cycling data to see what insights I can extract regarding my fitness over time as well as prototyping other possible functions. This 'project' has grown over time, as I think of more products that can be developed from the data.

### Background & Motivation: 
In cycling, heart rate is often used as a metric to evaluate an athlete's effort and sometimes used to tailor workouts at specific heart rate zones. However, it is known that heart rate is unique to each person in many ways. Not only can their resting and max heart rate be different, but their bodies can respond to cadences differently (some can maintain high cadence while others need slower cadences) as well as have different power profiles (some can put out short high-bursts of power while maintaining a consistent heart rate, while others can maintain moderate power for long periods of time). Due to all of these unique factors, a model that predicts heart rate will have to be athlete-specific. Additionally, with a linear model, the learned weights can be used as a fitness metric. It may not be best used to compare athletes to athletes, but there is potential to track variations in fitness by training models using recent data and comparing to previously trained weights.

The secondary purpose of this project is to put the skills I've accumulated through classes to work. Skills I demonstrate (some learned specifically for this) through this project include: 
* Webscraping 
* .xml/.fit parsing 
* Reinforcement Learning

## Contents
1. Data Gathering
2. Generating a linear model to predict Heart Rate
3. Using model to generate Heart Rate based workouts

## 1. Data Gathering

Sadly the most boring part of this, but required nonetheless. All of my cycling rides (indoor and outdoor) are stored on my Strava account. Strava has a nice feature that lets you bulk export all of your files. The not-so-nice part is that they come in varying formats (.gpx, .fit, .tcx), and they are all .gz compressed. The even-less-nice-part of the bulk export is that it doesn't include any indoor rides (ones that don't include any gps data) which includes all of my TrainerRoad rides.  
1. Steps for Strava's bulk export can be found [here](https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export). 
2. Unfortunately TrainerRoad doesn't have a bulk export feature, so I scripted a webscraper using Selenium ([src/scraper.py](https://github.com/eyang9001/Cycling-Workout-Generator/blob/master/src/scraper.py)) to pull them down as .fit files. In order to use this, you'll need the Selenium library installed and [ChromeDriver] (https://chromedriver.chromium.org/downloads) downloaded. The location of ChromeDriver is needed as a parameter:
```python
scrape_TR('TrainerRoad Username', 'password', '/path/to/ChromeDriver', max_files)
```
3. [src/conv_files.py decompress_files](https://github.com/eyang9001/Cycling-Workout-Generator/blob/master/src/conv_files.py) can be run to decompress the strava files from .gz to a new folder. It will also convert the .tcx files into .xml files which can then be read using the xml parser.
```python
decompress_files(data_filepath, new_filepath)
```
4. For each file, only the Heart Rate, Power and Cadence are needed, so the parse_fit_file and read_xml_file functions ([in src/parsers.py](https://github.com/eyang9001/Cycling-Workout-Generator/blob/master/src/parsers.py)) can be used to return these three time-series channels as lists.
```python
parse_fit_file('/path/to/.fit/file')
read_xml_file('/path/to/.xml/file')
```
