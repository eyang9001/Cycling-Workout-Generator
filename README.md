# Heart Rate Cycling Analysis & Workout Generator

This is a passion project of mine to utilize my cycling data and see what insights I can extract regarding my fitness over time as well as prototyping other possible functions. This 'project' has grown over time, as I think of more products that can be developed from the data.

### Background & Motivation
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

Sadly the most boring part, but required nonetheless. All of my cycling rides (indoor and outdoor) are stored on my Strava account. Strava has a nice feature that lets you bulk export all of your files. The not-so-nice part is that they come in varying formats (.gpx, .fit, .tcx), and they are all .gz compressed. The even-less-nice-part of the bulk export is that it doesn't include any indoor rides (ones that don't include any gps data) which includes all of my TrainerRoad rides.  
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

Here is an example of what the data for a workout looks like:
![image](https://user-images.githubusercontent.com/30561629/79378004-30a5a880-7f22-11ea-843c-fbb6695df54d.png)

## 2. Generating a linear model to predict Heart Rate

Using the datafiles, a linear model was generated with 4 weights, where the input to the model requires:
1. current heart rate
2. current power
3. current cadence

and it outputs the heart rate for the next second. The training data was normalized and then the model weights were tuned with gradient descent. The resulting model was able to imitate my heart rate pretty well:
![image](https://user-images.githubusercontent.com/30561629/71534940-cfe95600-28c7-11ea-9f9e-dc6201a0d5da.png)

However when the model is only seeded with the initial state and the subsequent timesteps use the previous predictions as the inputs to the next, the error propogation is noticeable:
![image](https://user-images.githubusercontent.com/30561629/71534917-9d3f5d80-28c7-11ea-89f3-c2dd43e3493a.png)

## 3. Generating Workout

A heart-rate based workout with specified targets over time was generated:
![image](https://user-images.githubusercontent.com/30561629/72549404-3ffb5400-3856-11ea-8ae7-2401a1a1099e.png)

Using this target series, a PID model using gradient descent was used alongside the linear model generated in step #2 to find the power and cadence input required to match the target heart-rate as close as possible:

### Training Progress:
![image](https://user-images.githubusercontent.com/30561629/72549720-d7f93d80-3856-11ea-8980-83751bcb2a0e.png)

### Final Output:
![image](https://user-images.githubusercontent.com/30561629/72549782-f6f7cf80-3856-11ea-901b-0f0730317171.png)

To simplify the model by minimizing the search-space, cadence was set to a constant 95. To keep the required power realistic, the power output was capped at 800 watts and minimum set to 0 (coasting). 
As you can see from the 'predicted' line, the predicted heart-rate does a good job resembling the characteristics of a person's real heart rate. It isn't possible for the heart rate to change instantaneously like the target curve, so the gradual build up and drop-off (when coasting) resembles how the body physiologically reacts to changes in effort (power).
