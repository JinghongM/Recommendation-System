import csv
import math
import json
from fuzzywuzzy import fuzz
from collections import OrderedDict
from pandas.io.json import json_normalize
def loadItemDataset(filename):
    brandIdItemsId={}
    with open(filename,'r') as data:
        parseData=json.load(data)
        for item in parseData:
            brandId=item["brand_id"]
            itemId=item["item_id"]
            if brandId not in brandIdItemsId:
                brandIdItemsId[brandId] = []
            brandIdItemsId[brandId].append(itemId)
    return brandIdItemsId
def loadRateDataset(filename):
    with open(filename,'r') as data:
        parseData=json.load(data)
        return parseData["brand_id"]

def loadBrandDataset(filename):
    brandGlossary={}
    with open(filename,'r') as data:
        parseData=json.load(data)
        for brand in parseData["brands"]:
            brandId=brand["brand_id"]
            curBrandList=[]
            for feature in list(brand["features"].keys()):
                curBrandList.append(brand["features"][feature])
            brandGlossary[brandId]=curBrandList
        return brandGlossary
def euclideanDistance(instance1,instance2):
    distance = 0
    featuresNum=len(instance1)
    for x in range(featuresNum):
        distance+=pow((float(instance1[x])-float(instance2[x])),2)
    return math.sqrt(distance)
def getNeighbors(dataset,testInstance,k):
    distances=[]
    for brandId,brandFeatures in dataset.items():
        dist=euclideanDistance(testInstance,brandFeatures)
        distances.append((brandId,dist))
    distances.sort(key=lambda tup:tup[1])
    neighbors=[]
    for x in range(min(k,len(distances))):
        neighbors.append(distances[x][0])
    return neighbors
def KnnBrandRecommender(items,highRatings,normalizedBrands,k,filterItemIDs=None):
    try:
        brandItems=loadItemDataset(items) #itemFile
        userRated=loadRateDataset(highRatings) #rateFile
        brandGlossary=loadBrandDataset(normalizedBrands) #brandGlossary
    except KeyError:
        raise ValueError("Incorrect file list!")
    brandCount={}
    for brandId in userRated:
        testInstance = brandGlossary[brandId]
        neighbors = getNeighbors(brandGlossary,testInstance,k)
        for neighbor in neighbors:
            if neighbor in brandCount:
                brandCount[neighbor] += 1
            else:
                brandCount[neighbor] = 1
    sortedBrandIds=[k for k in sorted(brandCount,key=brandCount.get,reverse=True)]
    desiredItems=[]
    for brandId in sortedBrandIds:
        if brandId in brandItems:
            desiredItems+=brandItems[brandId]
    if filterItemIDs == None:
        return desiredItems
    else:
        return [value for value in desiredItems if value in filterItemIDs]

