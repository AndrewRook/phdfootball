#!/opt/local/bin/python
####!/Library/Frameworks/Python.framework/Versions/Current/bin/python

import sys
import numpy as np
import scipy
import scipy.stats
from scipy.optimize import curve_fit
import math
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import MySQLdb as mdb
import astroML.plotting
import time


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def resetnumplays(gameids,cumplays):
    uniquegameids,uniquegamefirstindices,uniquegameindices = np.unique(gameids,return_index=True,return_inverse=True)
    start = time.time()
    splitcumplays = np.split(cumplays,uniquegamefirstindices[1:])
    playspergame = np.array([arr.min() for arr in splitcumplays],dtype=np.int)
    return cumplays - playspergame[uniquegameindices]

con = ''
data = []

#Querying the database:
try:
    con = mdb.connect('localhost','nfluser','lt56','armchairanalysis')
    cur = con.cursor()

    #Get the table:
    cur.execute('select core.*,games.h,games.ptsv,games.ptsh from core join games on core.gid=games.gid where core.dseq > 0')
    data = np.array(cur.fetchall())
    #0) Game id, 1) play id, 2) offensive team, 3) defensive team, 4) play type, 5) drive sequence, 6) play length, 7) quarter, 8) minutes left, 9) seconds left, 10) offensive points, 11) defensive points, 12) offense timeouts left, 13) defense timeouts left, 14) down, 15) yards to gain, 16) yards from own goal, 17) zone, 18) offensive line ID, 19) home team, 20) visitor points, 21) home points.
    #Get scoring plays:
    cur.execute('select scoring.* from scoring join core on core.pid = scoring.pid where core.dseq > 0')
    scoring = np.array(cur.fetchall())
    #0) play id, 1) number of points (can be negative for return/recovery TDs)
    cur.execute("select penalties.desc,core.pid from penalties inner join core on core.pid = penalties.pid inner join games on core.gid = games.gid where core.dseq > 0 group by penalties.pid")
    penalties = np.array(cur.fetchall())
    #0) Description, 1) play id

    #Connect the penalties and scoring to the data:
    connectscorebool = np.in1d(data[:,1].astype(np.int),scoring[:,0].astype(np.int))
    connectpoints = np.zeros(len(connectscorebool))
    connectpoints[connectscorebool] = scoring[:,1].astype(np.int)
    #remove points scored by defense (return/recovery/safety)
    connectpoints[connectpoints < 3] = 0
    cumconnectpoints = np.cumsum(connectpoints)
    
    connectpenaltiesbool = np.in1d(data[:,1].astype(np.int),penalties[:,1].astype(np.int))
    connectpenalties = np.array(['No Penalty' for i in range(len(connectpenaltiesbool))],dtype=object)
    connectpenalties[connectpenaltiesbool] = penalties[:,0]

    #Compute drive IDs for every play:
    firstplayofdrive = np.zeros(len(data[:,0]),dtype=np.bool)
    #Sometimes penalties are duplicated - so there's a play and then a NOPL. Need a bool that is false if a down has dseq=1, but play type = 'NOPL' and one of the adjacent plays has dseq = 1 and the same offense and defense.
    dup_penalty_bool = np.zeros(len(data[:,5]),dtype=np.bool)
    dup_penalty_bool[1:-1] = (data[1:-1,4] == 'NOPL') & (((data[:-2,5].astype(np.int) == 1) & (data[:-2,2] == data[1:-1,2]) & (data[:-2,3] == data[1:-1,3])) | ((data[2:,5].astype(np.int) == 1) & (data[2:,2] == data[1:-1,2]) & (data[2:,3] == data[1:-1,3])))
    firstplayofdrive[(data[:,5].astype(np.int) == 1) & (dup_penalty_bool == False)] = True#Seems to work
    driveid = np.cumsum(firstplayofdrive)
    timeleft = 60. - (data[:,7].astype(np.float)-1.)*15. - (15. - data[:,8].astype(np.float) - data[:,9].astype(np.float)/60.)

    uniquedriveids,uniquedrivefirstindices,uniquedriveindices = np.unique(driveid,return_index=True,return_inverse=True)
    start = time.time()
    splitdrivepoints = np.split(connectpoints,uniquedrivefirstindices[1:])
    pointsperdrive = np.array([arr.max() for arr in splitdrivepoints],dtype=np.int)
    pointsfromdrive = pointsperdrive[uniquedriveindices]
    print "Took {0:.2f} s".format(time.time()-start)
    
    #Cut arrays down:
    goodplays = ((data[:,4] == "RUSH") | (data[:,4] == "PASS") | (data[:,4] == "NOPL")) & (connectpenalties != "False Start") & (connectpenalties != "Encroachment") & (connectpenalties != "Delay of Game") & (np.abs(data[:,20].astype(np.int)-data[:,21].astype(np.int)) <= 8) 
    gooddata = data[goodplays,:]
    goodpoints = connectpoints[goodplays]
    goodpointsfromdrive = pointsfromdrive[goodplays]
    gooddriveid = driveid[goodplays]
    goodcumpoints = cumconnectpoints[goodplays]
    goodtimeleft = timeleft[goodplays]

    #Determine the first and last index for each drive:
    firstplayindices = np.unique(gooddriveid,return_index=True)[1]
    lastplayindices = len(gooddriveid) - 1 - np.unique(gooddriveid[::-1],return_index = True)[1]
    firstplays = np.zeros(len(gooddriveid),dtype=np.bool)
    firstplays[firstplayindices] = True
    lastplays = np.zeros(len(gooddriveid),dtype=np.bool)
    lastplays[lastplayindices] = True
    lastplaytimeleft = goodtimeleft[lastplayindices]
    firstplaytimeleft = goodtimeleft[firstplayindices]
    
    #Determine the number of plays in each drive and game ids:
    numplaysperdrive = lastplayindices-firstplayindices+1
    drivegameids = gooddata[firstplayindices,0].astype(np.int)


    #Determine the resulting number of points for each drive:
    driveresults = goodpointsfromdrive[firstplayindices]

    #Get which drives are by the home team:
    homedrives = (gooddata[firstplayindices,2] == gooddata[firstplayindices,19])

    #Cut out drives that end within 2 minutes of the end the half or the game:
    baddrivesbool = (firstplaytimeleft < 2.) | ((firstplaytimeleft-30 < 2.) & (firstplaytimeleft >= 30))
    gooddrivesbool = np.invert(baddrivesbool)
    gooddata = gooddata[gooddrivesbool,:]
    numplaysperdrive = numplaysperdrive[gooddrivesbool]
    driveresults = driveresults[gooddrivesbool]
    drivegameids = drivegameids[gooddrivesbool]
    homedrives = homedrives[gooddrivesbool]

    print homedrives.shape,drivegameids.shape,np.sum(gooddrivesbool),len(gooddrivesbool)
    

    #Sort each drive into home and away arrays for number of plays, game id, and resulting points:
    homenumplays = numplaysperdrive[homedrives]
    homeresults = driveresults[homedrives]
    homegameids = drivegameids[homedrives]
    awaynumplays = numplaysperdrive[homedrives == False]
    awayresults = driveresults[homedrives == False]
    awaygameids = drivegameids[homedrives == False]

    #Determine the cumulative number of plays at the start of each drive per game:
    play_threshold = 60
    homenumplays[1:] = homenumplays[:-1]
    homenumplays[0] = 0
    awaynumplays[1:] = awaynumplays[:-1]
    awaynumplays[0] = 0
    cumhomeplays = np.cumsum(homenumplays)
    cumawayplays = np.cumsum(awaynumplays)
    gamehomeplays = resetnumplays(homegameids,cumhomeplays)
    gameawayplays = resetnumplays(awaygameids,cumawayplays)
    goodhomeids = np.unique(homegameids[(gamehomeplays > play_threshold)])
    goodawayids = np.unique(awaygameids[(gameawayplays > play_threshold)])
    goodhomegamebool = np.in1d(homegameids,goodhomeids)
    goodawaygamebool = np.in1d(awaygameids,goodawayids)
    gametotplays = np.append(gamehomeplays[goodhomegamebool],gameawayplays[goodawaygamebool])
    totresults = np.append(homeresults[goodhomegamebool],awayresults[goodawaygamebool])

    #Make plots:
    #Distribution of drive lengths:
    lengthax = plt.figure().add_subplot(111)
    uniformbins = np.arange(numplaysperdrive.min()-0.5,numplaysperdrive.max()+0.5)
    lengthn,lengthbins = np.histogram(numplaysperdrive,bins = uniformbins)
    lengthax.bar(lengthbins[:-1],lengthn,width=(lengthbins[1:]-lengthbins[:-1]),color='gray',edgecolor='black',alpha=0.5,log=True)
    lengthax.set_xlabel('Plays Run on the Drive')
    lengthax.set_ylabel('Number of Drives')
    lengthax.xaxis.set_minor_locator(AutoMinorLocator())
    lengthax.figure.savefig('endurance_drivelength.png',dpi=300)

    #Distribution of drives per team per game:
    dpgax = plt.figure().add_subplot(111)
    homedrivespergame = np.bincount(homegameids)
    awaydrivespergame = np.bincount(awaygameids)
    homedpgbins = np.arange(homedrivespergame[homedrivespergame > 0].min()-0.5,homedrivespergame.max()+0.5)
    homedpgn,homedpgbins = np.histogram(homedrivespergame,bins=homedpgbins)

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [300., 10., 2.]

    homecoeff, homevar_matrix = curve_fit(gauss, (homedpgbins[:-1]+homedpgbins[1:])/2., homedpgn, p0=p0)
    print "Home: ",scipy.stats.describe(homedrivespergame[homedrivespergame > 0])
    print homecoeff
    homegaussx = np.linspace(homedpgbins[0],homedpgbins[-1],100)
    homegaussy = gauss(homegaussx,*homecoeff)
    dpgax.plot(homegaussx,homegaussy,ls='-',color='blue',lw=4)
    dpgax.bar(homedpgbins[:-1],homedpgn,width=(homedpgbins[1:]-homedpgbins[:-1]),color='blue',edgecolor='blue',alpha=0.5,log=True,label='Home')
    awaydpgbins = np.arange(awaydrivespergame[awaydrivespergame > 0].min()-0.5,awaydrivespergame.max()+0.5)
    awaydpgn,awaydpgbins = np.histogram(awaydrivespergame,bins=awaydpgbins)
    awaycoeff, awayvar_matrix = curve_fit(gauss, (awaydpgbins[:-1]+awaydpgbins[1:])/2., awaydpgn, p0=p0)
    print "Away: ",scipy.stats.describe(awaydrivespergame[awaydrivespergame > 0])
    print awaycoeff
    awaygaussx = np.linspace(awaydpgbins[0],awaydpgbins[-1],100)
    awaygaussy = gauss(awaygaussx,*awaycoeff)
    dpgax.plot(awaygaussx,awaygaussy,ls='-',color='red',lw=4)
    dpgax.bar(awaydpgbins[:-1],awaydpgn,width=(awaydpgbins[1:]-awaydpgbins[:-1]),color='red',edgecolor='red',alpha=0.5,log=True,label='Away')
    dpgax.set_xlabel('Drives per Game')
    dpgax.set_ylabel('Number of Games')
    dpgax.set_ylim(1,dpgax.get_ylim()[1])
    dpgax.xaxis.set_minor_locator(AutoMinorLocator())
    dpgax.legend(loc='upper right',prop={'size':10})
    dpgax.figure.savefig('endurance_dpg.png',dpi=300)

    #Fraction of drives that end in points:
    #Making even bins:
    sortedplays = np.sort(gametotplays)
    numbins = 15
    numperbin = int(np.floor(len(gametotplays)/float(numbins)))
    totbins = sortedplays[::numperbin]
    totbins[-1] = sortedplays[-1]    
    totbins = np.unique(totbins)
    #Making histograms:
    totn,totbins = np.histogram(gametotplays,bins=totbins)
    tdn,tdbins = np.histogram(gametotplays[(totresults > 5) & (totresults < 9)],bins=totbins)
    fgn,fgbins = np.histogram(gametotplays[totresults == 3],bins=totbins)
    scoren,scorebins = np.histogram(gametotplays[totresults > 2],bins=totbins)

    #Fractions:
    tdfrac = tdn.astype(np.float)/totn.astype(np.float)
    fgfrac = fgn.astype(np.float)/totn.astype(np.float)
    scorefrac = scoren.astype(np.float)/totn.astype(np.float)
    #Poisson errors:
    toterr = np.sqrt(totn)
    tderr = np.sqrt(tdn+(tdn*toterr/totn)**2)/totn
    fgerr = np.sqrt(fgn+(fgn*toterr/totn)**2)/totn
    scoreerr = np.sqrt(scoren+(scoren*toterr/totn)**2)/totn
    #Totals:
    totax = plt.figure().add_subplot(111)
    totax.bar(totbins[:-1],totn,width=(totbins[1:]-totbins[:-1]),color='gray',edgecolor='black',alpha=0.5,yerr=tderr,ecolor='black')
    totax.set_xlabel('Plays Run on the Defense')
    totax.set_ylabel('Number of Drives')
    totax.figure.savefig('endurance_totals.png',dpi=300)
    #TDs:
    tdax = plt.figure().add_subplot(111)
    tdax.bar(totbins[:-1],tdfrac,width=(totbins[1:]-totbins[:-1]),color='gray',edgecolor='black',alpha=0.5,yerr=tderr,ecolor='black')
    tdax.set_xlabel('Plays Run on the Defense')
    tdax.set_ylabel('Fraction of Drives Ending With TDs')
    tdax.figure.savefig('endurance_tds.png',dpi=300)
    #FGs:
    fgax = plt.figure().add_subplot(111)
    fgax.bar(totbins[:-1],fgfrac,width=(totbins[1:]-totbins[:-1]),color='gray',edgecolor='black',alpha=0.5,yerr=fgerr,ecolor='black')
    fgax.set_xlabel('Plays Run on the Defense')
    fgax.set_ylabel('Fraction of Drives Ending With FGs')
    fgax.figure.savefig('endurance_fgs.png',dpi=300)
    #Scores:
    scoreax = plt.figure().add_subplot(111)
    scoreax.bar(totbins[:-1],scorefrac,width=(totbins[1:]-totbins[:-1]),color='gray',edgecolor='black',alpha=0.5,yerr=scoreerr,ecolor='black')
    scoreax.set_xlabel('Plays Run on the Defense')
    scoreax.set_ylabel('Fraction of Drives Ending With Scores')
    scoreax.figure.savefig('endurance_scores.png',dpi=300)
 


except mdb.Error, e:
    print "Error %d: %s" % (e.args[0],e.args[1])
    sys.exit(1)


finally:
    if con:
        con.close()
