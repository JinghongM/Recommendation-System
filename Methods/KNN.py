import csv
import math
import json
from fuzzywuzzy import fuzz
from collections import OrderedDict
from pandas.io.json import json_normalize
#load function to load item json file
#input: json file path
#output: brandID dictionary with brandID=>itemIDs list
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
#load function to load rating file
#input: json file path
#output: high rating brandIDs list
def loadRateDataset(filename):
    with open(filename,'r') as data:
        parseData=json.load(data)
        return parseData["brand_id"]
#load function to load brand file
#input: json file path
#output:brand features dictionary with brandID=>features
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
#calculate distance between two brand instances
#input: instances vectors
#output: distance
def euclideanDistance(instance1,instance2):
    distance = 0
    featuresNum=len(instance1)
    for x in range(featuresNum):
        distance+=pow((float(instance1[x])-float(instance2[x])),2)
    return math.sqrt(distance)
#get the sorted neighbours for one specific brand
#input: whole brand dataset, test brand
#output: brands list
def getNeighbors(dataset,testInstance):
    distances=[]
    for brandId,brandFeatures in dataset.items():
        dist=euclideanDistance(testInstance,brandFeatures)
        distances.append((brandId,dist))
    distances.sort(key=lambda tup:tup[1])
    neighbors=[]
    for x in range(len(distances)):
        neighbors.append(distances[x][0])
    return neighbors
#Items: items .json file with itemID and brandID
#highRatings: rating .json file with the high ratings brandID
#normalizedBrands: generated brand matrix
#k: the number of brands that are the most nearest brands to high rating brands
#filterItemsID: the list of filtered items according to gender,temperature
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
        neighbors = getNeighbors(brandGlossary,testInstance)
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

