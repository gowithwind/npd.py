'''
reimplentation of https://github.com/biotrump/NPD/blob/master/NPDFaceDetector/NPDScan.cpp
'''
import math
import numpy as np
def NPDScan(model, I, minFace, maxFace):
    pModel = model
    objSize = pModel['objSize'][0][0]
    numStages = pModel['numStages'][0][0]
    numBranchNodes = pModel['numBranchNodes'][0][0]
    pStageThreshold = pModel['stageThreshold'][0]
    pTreeRoot = pModel['treeRoot'][0]
    numScales = pModel['numScales'][0][0]
    ppPoints1 = pModel['pixel1']
    ppPoints2 = pModel['pixel2']
    ppCutpoint = pModel['cutpoint']
    pLeftChild = pModel['leftChild'][0]
    pRightChild = pModel['rightChild'][0]
    pFit = pModel['fit'][0]
    ppNpdTable = pModel['npdTable']
    pWinSize = pModel['winSize'][:,0]
    height,width = I.shape
    minFace = max(minFace, objSize)
    maxFace = min(maxFace, min(height, width))
    if(min(height, width) < minFace):
        return []
    # // containers for the detected faces
    rects = []
    I = I.T.flatten()

    for k in range(numScales):
        if(pWinSize[k] < minFace):
             continue;
        elif(pWinSize[k] > maxFace):
             break;

        # // determine the step of the sliding subwindow
        winStep = int(math.floor(pWinSize[k] * 0.1));
        if(pWinSize[k] > 40):
            winStep = int(math.floor(pWinSize[k] * 0.05));

        # // calculate the offset values of each pixel in a subwindow
        # // pre-determined offset of pixels in a subwindow
        offset = np.zeros((pWinSize[k] * pWinSize[k]), dtype=np.int)
        p1 = 0
        p2 = 0
        gap = height - pWinSize[k]
        for j in range(pWinSize[k]):
            for i in range(pWinSize[k]):
                offset[p1] = p2
                p1 += 1
                p2 += 1
            p2 += gap
        colMax = width - pWinSize[k] + 1;
        rowMax = height - pWinSize[k] + 1;
        print "k=",k,pWinSize[k],winStep
        for c in range(0,colMax,winStep):
            pPixel = c*height
            for r in range(0,rowMax,winStep):
                treeIndex = 0;
                _score = 0;
                s = 0
                # // test each tree classifier
                for s in range(numStages):
                    node = pTreeRoot[treeIndex];
                    # // test the current tree classifier
                    while(node > -1): #// branch node
                        p1 = I[pPixel+offset[ppPoints1[k][node]]];
                        p2 = I[pPixel+offset[ppPoints2[k][node]]];
                        fea = ppNpdTable[p1][p2];
                        if(fea < ppCutpoint[0][node] or fea > ppCutpoint[1][node]):
                             node = pLeftChild[node];
                        else:
                             node = pRightChild[node];
                    node = - node - 1;
                    _score = _score + pFit[node];
                    treeIndex += 1
                    if(_score < pStageThreshold[s]):
                         break; #// negative samples
                pPixel += winStep
                if(s == numStages -1):# // a face detected
                    print 'face=',c,r,pWinSize[k]
                    rects.append([c,r,_score,pWinSize[k],pWinSize[k]])
    return rects
