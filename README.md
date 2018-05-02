# TalkingData_competition

## Downloading the dataset
1. First clone this repository by using `git clone https://github.com/minging234/TalkingData_competition.git`
2. For this project, we will use the `train.csv` file. Navigate to [Kaggle download link](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data), and download the zip file into the repo folder.
3. Unzip the csv file. It should automatically be placed in `./mnt/ssd/kaggle-talkingdata2/competition_files` folder.

## Getting cleaned data
Due to the large number of features, it is not possible to work with all of the data. For now, I limit the size of dataset to the 
first 50 million samples (out of 178 million total samples). To get cleaned data that can be readily used for classification:
1. Run the clean_data.py script by `python3 clean_data.py`
2. If you downloaded the data according instructions in [Downloading the dataset](#downloading-the-dataset), everything should run automatically. Otherwise, modify the `path-train` variable in `clean_data.py`.

## Running algorithms
Once the data is cleaned, you can use Jupyter notebook to open up `workbook.ipynb` in this repository. This notebook will load the data for you in style that is similar to the ones used throughout class (X, y variables).