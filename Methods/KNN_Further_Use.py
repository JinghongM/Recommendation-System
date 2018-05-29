def KnnRecommender(itemFile,rateFile,userId,k,filterItemIDs):
    itemDataset=loadItemDataset(itemFile,16)
    filteredItemDataset=[]
    for filterItemID in filterItemIDs:
        for item in itemDataset:
            if int(item[0]) == filterItemID:
                filteredItemDataset.append(item)
                continue
    userRated=loadRateDataset(rateFile,userId)
    resultDict={}
    for itemId in userRated:
        print("rate itemId:" + str(itemId))
        for filteredItem in filteredItemDataset:
            if int(filteredItem[0]) == itemId:
                testInstance = filteredItem
                print("test Instance=")
                print(testInstance)
                neighbors = getNeighbors(filteredItemDataset,testInstance,k)
                for neighbor in neighbors:
                    if frozenset(neighbor) in resultDict:
                        resultDict[frozenset(neighbor)] += 1
                    else:
                        resultDict[frozenset(neighbor)] = 1
                continue
    sortedResult=[(list(k),resultDict[k]) for k in sorted(resultDict,key=resultDict.get,reverse=True)]
    return sortedResult
def knnRecommender(itemFile,rateFile,userId,k,filterItemIDs):
    result = KnnRecommender(itemFile,rateFile,userId,k,filterItemIDs)
    while len(result) < k:
        print(k)
        k+=1
        result = KnnRecommender(itemFile,rateFile,userId,k+1,filterItemIDs)
    return result
