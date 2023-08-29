eqdskDict = {'147634' : 'g147634.04525',
             '174658' : 'g174658.3020', #g174658.03000',
             'NSTX'   : 'g130608.00352'

            }


#shotNum is a string
def getEqdskName(shotNum):
    return eqdskDict[shotNum]
