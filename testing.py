import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from matplotlib import pyplot as plt
def cleanup( uint8 ):
    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    image = imutils.resize(uint8, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #thresh, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite("bwtest.png", gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    # find contours in the edge map, then sort them by their
    # size in descending order
    cv2.imwrite("edge3.png", edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    displayCnt = None

    x, y, w, h = cv2.boundingRect(cnt)
    bound = cv2.rectangle(gray, (x-40, y-40), (x + w + 40, y + h + 40), (0, 255, 0), 2)
    cv2.imwrite("boundtest.png", bound)
    warped = bound[(y-35):(y+h+35), (x-35):(x+w+35)]
    cv2.imwrite("warptest.png", warped)
    output = cv2.resize(warped, (20, 20))
    output = cv2.bitwise_not(output)
    cv2.imwrite("down3.png", output)
    return output

def getpic( str ):
    try:
        picture = cv2.imread(str)
        return picture
    except cv2.error as e:
        print ("File not found")
        quit()

filename = raw_input("Enter the filename of the picture: ")
picture = getpic(filename)
cleanpic = cleanup(picture)
cv2.imwrite("clean3.png", cleanpic)
img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
y = np.array(cleanpic)
# Now we prepare train_data and test_data.
train = x[:,:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
pictest = y.reshape(-1,400).astype(np.float32)
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,500)[:,np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=3)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( accuracy )
finret,finresult,finneighbors,findist = knn.findNearest(pictest, k=3)
print ("The result is ", finresult[0,0])
# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)
# Now load the data
with np.load('knn_data.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']
