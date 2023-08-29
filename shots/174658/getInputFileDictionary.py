##I'm very sorry for this abomination of coding practice##
##I cannot get python to import from a parent folder, so we copy this into all of the subfolders##
import re
import numpy as np
def createInputFileDictionary(path):
    inputFile = open(path,'r')
    
    inputFileDict = {}
    
    inputLines = inputFile.readlines()
    
    variableValue = ''
    variableName = ''

    sectionName = ''

    for i in range(len(inputLines)):
        line = inputLines[i].strip()
        if len(line) == 0:
            continue

        if line[0] == '&':# and line[1:]!='end':
            #if this is not the first section, add the last variable to the dictionary before moving onto this new section
            if sectionName != '':
                addVariable(inputFileDict, sectionName, variableName, variableValue)
                variableValue = ''
                variableName = ''

            sectionName = line[1:]
            inputFileDict[sectionName] = {}
            continue 


        #"""       
        splitLine = line.split('=')
        
        #if this line is a continuation of a previous variable
        if len(splitLine) == 1:
            splitLine = re.sub("\s+", ",", splitLine[0].strip()).replace(' ',',')
            variableValue = variableValue + splitLine + ','
            if i == len(inputLines) - 1:
                addVariable(inputFileDict, sectionName, variableName, variableValue)
        #if this is a new variable
        else:
            if variableName != '':
                addVariable(inputFileDict, sectionName, variableName, variableValue)
                variableValue = ''
                variableName = ''
            
            variableName = splitLine[0].strip(); variableValue = re.sub("\s+", ",", splitLine[1].strip()).replace(' ',',') + ','
        #"""
            
    return inputFileDict
    
def addVariable(dictionary, sectionName, variableName, variableValue):
    #get rid of extra comma at the end
    #print(f"{sectionName, variableName, variableValue}")
    variableValue = variableValue.replace(',,',',')
    variableValue = variableValue[:-1]
    splitVariable = np.array(variableValue.split(','))
        
    try:  
        if len(splitVariable) > 1:
            dictionary[sectionName][variableName]=np.array([float(var) for var in splitVariable])

        else:
            try: 
                dictionary[sectionName][variableName]=float(variableValue)
            except:
                dictionary[sectionName][variableName]=variableValue
    except Exception as e:
        pass
def getInputFileDictionary(gen_or_cql,pathprefix = ""):
    import os, sys
    if gen_or_cql == 'genray':
        return createInputFileDictionary(f'{pathprefix}genray_received.in')
    if gen_or_cql == 'cql3d':
        return createInputFileDictionary(f'{pathprefix}cqlinput_received')
        
