import sys
import numpy as np
import scipy
import scipy.stats
import math
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import MySQLdb as mdb

def computeexpectationvalue(draftpositions):
    #Compute the number of players taken at each draft position:
    numperposition = np.bincount(draftpositions)[1:]#the [1:] gets rid of zeroth pick, which is used for undrafted players.
    positions = np.arange(1,len(numperposition)+1)
    probabilities = np.cumsum(numperposition[::-1])[::-1]/float(np.sum(numperposition))
    expectationvalue = np.sum(probabilities)
#     for i in range(len(positions)):
#         print positions[i],numperposition[i],backcumsum[i]
    return expectationvalue


con = ''
data = []

#Querying the database:
try:
    con = mdb.connect(read_default_file='~/.my.cnf',read_default_group='aadb')
    cur = con.cursor()

    #Get the table:
    cur.execute('select roster.pos1,roster.dpos from roster where roster.dpos > 0 and roster.start > 2001')
    data = np.array(cur.fetchall())

    #Compute the expectation value of each position:
    qbexpectation = computeexpectationvalue(data[(data[:,0]=='QB'),1].astype(np.int))
    rbexpectation = computeexpectationvalue(data[(data[:,0]=='RB'),1].astype(np.int))
    wrexpectation = computeexpectationvalue(data[(data[:,0]=='WR'),1].astype(np.int))
    teexpectation = computeexpectationvalue(data[(data[:,0]=='TE'),1].astype(np.int))
    olexpectation = computeexpectationvalue(data[(data[:,0]=='OL'),1].astype(np.int))
    dlexpectation = computeexpectationvalue(data[(data[:,0]=='DL'),1].astype(np.int))
    dbexpectation = computeexpectationvalue(data[(data[:,0]=='DB'),1].astype(np.int))
    lbexpectation = computeexpectationvalue(data[(data[:,0]=='LB'),1].astype(np.int))
    kexpectation = computeexpectationvalue(data[(data[:,0]=='K'),1].astype(np.int))
    print "QB: {0:.2f}, RB: {1:.2f}, WR: {2:.2f}, TE: {3:.2f}, OL: {4:.2f}".format(qbexpectation,rbexpectation,wrexpectation,teexpectation,olexpectation)
    print "DL: {0:.2f}, LB: {1:.2f}, DB: {2:.2f}, K: {3:.2f}".format(dlexpectation,lbexpectation,dbexpectation,kexpectation)

    #Make some histogram plots:
    bins = np.arange(1,250,10)
    totn,totbins = np.histogram(data[:,1].astype(np.int), bins=bins)
    qbn,qbbins = np.histogram(data[(data[:,0]=='QB'),1].astype(np.int),bins=bins)
    wrn,wrbins = np.histogram(data[(data[:,0]=='WR'),1].astype(np.int),bins=bins)
    rbn,rbbins = np.histogram(data[(data[:,0]=='RB'),1].astype(np.int),bins=bins)
    ten,tebins = np.histogram(data[(data[:,0]=='TE'),1].astype(np.int),bins=bins)
    oln,olbins = np.histogram(data[(data[:,0]=='OL'),1].astype(np.int),bins=bins)
    dln,dlbins = np.histogram(data[(data[:,0]=='DL'),1].astype(np.int),bins=bins)
    lbn,lbbins = np.histogram(data[(data[:,0]=='LB'),1].astype(np.int),bins=bins)
    dbn,dbbins = np.histogram(data[(data[:,0]=='DB'),1].astype(np.int),bins=bins)
    kn,kbins = np.histogram(data[(data[:,0]=='K'),1].astype(np.int),bins=bins)
    qbfrac = qbn/totn.astype(np.float)
    rbfrac = rbn/totn.astype(np.float)
    wrfrac = wrn/totn.astype(np.float)
    tefrac = ten/totn.astype(np.float)
    olfrac = oln/totn.astype(np.float)
    dlfrac = dln/totn.astype(np.float)
    lbfrac = lbn/totn.astype(np.float)
    dbfrac = dbn/totn.astype(np.float)
    kfrac = kn/totn.astype(np.float)
    
    ax = plt.figure().add_subplot(111)
    ax.bar(bins[:-1],qbfrac*100.,width=(bins[1:]-bins[:-1]),color='blue',label='QB')
    ax.bar(bins[:-1],rbfrac*100.,width=(bins[1:]-bins[:-1]),color='purple',bottom=qbfrac*100.,label='RB')
    ax.bar(bins[:-1],wrfrac*100.,width=(bins[1:]-bins[:-1]),color='red',bottom=(qbfrac+rbfrac)*100.,label='WR')
    ax.bar(bins[:-1],tefrac*100.,width=(bins[1:]-bins[:-1]),color='orange',bottom=(qbfrac+rbfrac+wrfrac)*100.,label='TE')
    ax.bar(bins[:-1],olfrac*100.,width=(bins[1:]-bins[:-1]),color='gold',bottom=(qbfrac+rbfrac+wrfrac+tefrac)*100.,label='OL')
    ax.bar(bins[:-1],dlfrac*100.,width=(bins[1:]-bins[:-1]),color='green',bottom=(qbfrac+rbfrac+wrfrac+tefrac+olfrac)*100.,label='DL')
    ax.bar(bins[:-1],lbfrac*100.,width=(bins[1:]-bins[:-1]),color='gray',bottom=(qbfrac+rbfrac+wrfrac+tefrac+olfrac+dlfrac)*100.,label='LB')
    ax.bar(bins[:-1],dbfrac*100.,width=(bins[1:]-bins[:-1]),color='pink',bottom=(qbfrac+rbfrac+wrfrac+tefrac+olfrac+dlfrac+lbfrac)*100.,label='DB')
    ax.bar(bins[:-1],kfrac*100.,width=(bins[1:]-bins[:-1]),color='brown',bottom=(qbfrac+rbfrac+wrfrac+tefrac+olfrac+dlfrac+lbfrac+dbfrac)*100.,label='K')
    ax.set_ylim(0,100)
    ax.set_xlim(bins.min(),bins.max())
    ax.set_xlabel('Draft Position')
    ax.set_ylabel('Percentage of Players Drafted')
    ax.legend(loc='upper right',bbox_to_anchor=(1.1,1.0),prop={'size':10},fancybox=True)
    ax.figure.savefig('draftpos.png',dpi=300)

    #Plot the fractional take for each position for each bin:
    bincenters = (bins[1:]+bins[:-1])/2.
    alln = np.array([qbn,rbn,wrn,ten,oln,dln,lbn,dbn,kn])
    allindivfrac = np.array([qbn/float(np.sum(qbn)),rbn/float(np.sum(rbn)),wrn/float(np.sum(wrn)),ten/float(np.sum(ten)),oln/float(np.sum(oln)),dln/float(np.sum(dln)),lbn/float(np.sum(lbn)),dbn/float(np.sum(dbn)),kn/float(np.sum(kn))])
    sortedallnargs = np.argsort(-alln,axis=0)
    sortedindivfracargs = np.argsort(-allindivfrac,axis=0)
    print allindivfrac.shape
    ax = plt.figure().add_subplot(111)
    colorlist = ['blue','purple','red','orange','gold','green','gray','pink','brown']
    labellist = ['QB','RB','WR','TE','OL','DL','LB','DB','K']
    for i in range(len(colorlist)):
        ax.plot(bincenters,allindivfrac[i,:],marker='o',ls='-',color=colorlist[i],mec=colorlist[i],mfc=colorlist[i],ms=3,label=labellist[i])

    ax.set_xlabel('Draft Position')
    ax.set_ylabel('Percentage of Players Drafted Per Position')
    ax.legend(loc='upper right',prop={'size':10},fancybox=True,numpoints=2)
    ax.figure.savefig('draftpos_indivfrac.png',dpi=300)

except mdb.Error, e:
    print "Error %d: %s" % (e.args[0],e.args[1])
    sys.exit(1)


finally:
    if con:
        con.close()

