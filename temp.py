import csv
import glob

dataDir = './training_raw/'
# load groundTruth
print("Loading ground truth file")
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir + "*.mat"))
print(files)