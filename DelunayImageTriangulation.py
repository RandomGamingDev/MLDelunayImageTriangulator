# This is a python script that converts an image into a series of triangles
# Created by: RandomGamingDev
# 
#     How to use the script:
#
#     Start by entering the settings that you want in settings.txt
#     The 1st line is for the # of instances
#     The 2nd line is for the # of triangles per instance
#     The 3rd line is for the # of iterations
#
#     Then after confirming the settings enter the filepath to the image relative to the executable or as a direct filepath
#
#     Then confirm that that's the image that you want
#
#     Wait for the final result in output.jpeg!

from PIL import Image, ImageDraw
import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt

mode = "RGB"
output = "jpeg"

graphDelay = 1

numIns = None
numPoi = None
numGen = None
const_img = None
img = None
canvas = None
triangle_sample = None
triangle_draw = None
dimensions = None
instances = None
worstCase = None
bestInstance = None
accuracy = None

def SetupFileData():
    global numIns
    global numPoi
    global numGen
    global const_img
    global img
    global canvas
    global triangle_sample
    global triangle_draw
    global dimensions
    global instances
    global worstCase
    global bestInstance
    global accuracy
    print("Getting Settings...")
    print("    Reading Settings File...")
    settingstxt = open("./settings.txt", "r", encoding="utf-8")
    settings = settingstxt.readlines()
    settingstxt.close()
    print("    Read Settings File!")
    print("    Storing the Retrieved Settings...")
    print("    Settings:")
    print("        Line 1:")
    numIns = int(settings[0]) # The # of instances
    print(f"            # of Instances: {numIns}")
    print("        Line 2:")
    numPoi = int(settings[1]) # The # of triangles per instance
    print(f"            # of Points for each Instance: {numPoi}")
    print("        Line 3:")
    numGen = int(settings[2]) # The # of iterations
    print(f"            # of interations for each Instance: {numGen}")
    print("    Stored the Retrieved Settings!")
    print("Got Settings!")
    input("Are these settings good? If yes press ENTER.")
    print("Settings validated!")

    img_name = input("What img do you want to get? Enter the filepath to the image here: ")
    print("Getting the image...")
    const_img = Image.open(img_name).convert(mode)
    dimensions = np.array(const_img.size)
    img = Image.new(mode = "RGB", size = (dimensions[0], dimensions[1]))
    canvas = ImageDraw.Draw(img)
    triangle_sample = Image.new(mode = "RGB", size = (dimensions[0], dimensions[1]))
    triangle_draw = ImageDraw.Draw(triangle_sample)
    print("Got the image!")
    print("Image Attributes:")
    print(f"    Name: {img_name}")
    print(f"    Width: {dimensions[0]}")
    print(f"    Height: {dimensions[1]}")
    const_img.show()
    input("Is this image good? If yes press ENTER.")
    print("Input validated!")

    plt.figure(num="ML Model Accuracy Chart")
    plt.title("ML Model Accuracy")
    plt.xlabel("Generations")
    plt.ylabel("% Accuracy")
    plt.show(block=False)

    instances = np.random.randint(0, [dimensions[0], dimensions[1]], size=(numIns, numPoi, 2))

    worstCase = np.sum(abs((255 * (np.asarray(const_img) < 128)) - np.asarray(const_img)))
    bestInstance = np.array((0, worstCase))
    accuracy = []

def ClearImg(image):
    image.rectangle((0, 0, dimensions[0] - 1, dimensions[1] - 1), fill=(255, 255, 255))
def SampleTriange(coords):
    global triangle_sample
    ClearImg(triangle_draw)
    triangle_draw.polygon(list(map(tuple, coords)), fill = (0, 0, 0))
    min = np.min(coords, axis=0)
    max = np.max(coords, axis=0)
    return tuple(np.round_(np.mean(np.asarray(const_img)[min[1]:max[1] + 1, min[0]:max[0] + 1][(np.asarray(triangle_sample)[min[1]:max[1] + 1, min[0]:max[0] + 1] == np.asarray((0, 0, 0))).all(axis=2)], axis=0)).astype(np.uint8))
def RenderTriangles(instance):
    triangles = sp.spatial.Delaunay(instance)
    for i in range(len(triangles.simplices)):
        canvas.polygon(list(map(tuple, instance[triangles.simplices[i]])), fill = SampleTriange(instance[triangles.simplices[i]]))
def HowDifferent(instance):
    RenderTriangles(instance)
    return np.sum(np.absolute(np.subtract(const_img, img, dtype = np.int16)))
def Test():
    for j in range(1, len(instances)):
        ClearImg(canvas)
        difference = HowDifferent(instances[j])
        if difference < bestInstance[1]:
            bestInstance[0] = j
            bestInstance[1] = difference
def ReproduceAndMutate():
    global graphTimer
    accuracy.append(100 * (worstCase - bestInstance[1]).astype(np.uint64)/worstCase)
    plt.clf()
    plt.plot(accuracy)
    plt.pause(graphDelay)
    instances[0] = np.copy(instances[bestInstance[0]])
    for j in range(1, len(instances)):
        instances[j] = np.copy(instances[0])
        for k in range(np.random.randint(numPoi * 2)):
            attribute = np.random.randint(2)
            instances[j][np.random.randint(numPoi)][attribute] = np.random.randint(dimensions[attribute])

if __name__ == "__main__":
    SetupFileData()
    print("Machine learning algorithm starting...")
    for i in range(numGen):
        bestInstance[0] = 0
        Test()
        ReproduceAndMutate()
        if bestInstance[0] != 0:
            ClearImg(canvas)
            RenderTriangles(instances[0])
            img.save(f"output.{output}", format=output)
    print("Machine learning algorithm ended!")
    input("Display result?")
    print("Displaying result...")
    img.show()
    print("Displayed result!")
