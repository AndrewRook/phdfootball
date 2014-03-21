import numpy as np
import math
import sys
import MySQLdb as mdb
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import time

#Get the distributions of all the times a given down and distance occurs in the data:
def compute_distributions(data,downs=[1,2,3,4],ytgbins=np.array([1,2,3,4,5,6,7,8,9,10,10.1])):
    downstats = []
    for i in range(len(downs)):
        #Get all the data from that down:
        downbool = (data[:,14].astype(np.int) == downs[i])
        downdata = data[downbool,:]
        downstats.append(compute_down_distribution(downdata,ytgbins))

    return downstats

#Get the number of plays at each YTG for the dataset:
def compute_down_distribution(data,ytgbins):
    ytgs = data[:,15].astype(np.int)
    ytg_dist = np.histogram(ytgs,bins=ytgbins)[0]
    return ytg_dist

if __name__ == "__main__":
    con = ''

    #Querying the database:
    try:
        con = mdb.connect(read_default_file='~/.my.cnf',read_default_group='aadb')
        cur = con.cursor()

        
        #Get the table:
        start = time.time()
        cur.execute('select core.*,T.pos1,games.seas from core left join (select rush.pid,roster.pos1 from rush join roster on roster.player=rush.bc) as T on core.pid=T.pid join games on core.gid=games.gid where games.seas >= 2000 and games.seas <= 2011')
        print "Time = {0:.2f}".format(time.time()-start)
        data = np.array(cur.fetchall())
        #0) Game id, 1) play id, 2) offensive team, 3) defensive team, 4) play type, 5) drive sequence, 6) play length, 7) quarter, 8) minutes left, 9) seconds left, 10) offensive points, 11) defensive points, 12) offense timeouts left, 13) defense timeouts left, 14) down, 15) yards to gain, 16) yards from own goal, 17) zone, 18) offensive line ID, 19) Rusher position (NULL if not rush), 20) season.

        #Compute drive IDs for every play:
        firstplayofdrive = np.zeros(len(data[:,0]),dtype=np.bool)
        #Sometimes penalties are duplicated - so there's a play and then a NOPL. Need a bool that is false if a down has dseq=1, but play type = 'NOPL' and one of the adjacent plays has dseq = 1 and the same offense and defense.
        dup_penalty_bool = np.zeros(len(data[:,5]),dtype=np.bool)
        dup_penalty_bool[1:-1] = (data[1:-1,4] == 'NOPL') & (((data[:-2,5].astype(np.int) == 1) & (data[:-2,2] == data[1:-1,2]) & (data[:-2,3] == data[1:-1,3])) | ((data[2:,5].astype(np.int) == 1) & (data[2:,2] == data[1:-1,2]) & (data[2:,3] == data[1:-1,3])))
        firstplayofdrive[(data[:,5].astype(np.int) == 1) & (dup_penalty_bool == False)] = True#Seems to work
        driveid = np.cumsum(firstplayofdrive)
        timeleft = 60. - (data[:,7].astype(np.float)-1.)*15. - (15. - data[:,8].astype(np.float) - data[:,9].astype(np.float)/60.)

        #Now need to figure out whether drive had additional first downs:
        firstdownbool = (data[:,14].astype(np.int) == 1)
        cumfirstdowns = np.cumsum(firstdownbool)
        uniquedriveids,uniquedrivefirstindices,uniquedriveindices = np.unique(driveid,return_index=True,return_inverse=True)
        splitfirstdowns = np.split(cumfirstdowns,uniquedrivefirstindices[1:])
        firstdownsperdrive = np.array([arr.max() for arr in splitfirstdowns],dtype=np.int)
        firstdownsfromdrive = firstdownsperdrive[uniquedriveindices]
        #You know the offense got at least one more first down during the drive when cumfirstdowns[i] < firstdownsfromdrive[i]

        #Apply any filters (e.g. time, quarter, etc) to the data:
        filterbool = np.ones(data.shape[0],dtype=np.bool)
        #Filter out anything within a given time limit in the 2nd or 4th quarter
        filterbool = filterbool & (((data[:,7].astype(np.int) != 2) & (data[:,7].astype(np.int) != 4)) | (data[:,8].astype(np.int) >= 2))
        #Filter out games that aren't close:
        filterbool = filterbool & (np.abs(data[:,10].astype(np.int)-data[:,11].astype(np.int)) <= 16)
        #Filter out penalties:
        filterbool = filterbool & ((data[:,4] == 'RUSH') | (data[:,4] == 'PASS'))
        #Filter out things in the redzone to avoind N-and-goal plays:
        filterbool = filterbool & (data[:,16].astype(np.int) <= 80)
        #Filter out plays when the offense is backed up in their own endzone:
        filterbool = filterbool & (data[:,16].astype(np.int) >= 10)

        #Make a filter as close to Brian Burke's as I can get it:
        burkefilterbool = np.ones(data.shape[0],dtype=np.bool)
        #Filter out anything within the 2 minute warnings
        burkefilterbool = burkefilterbool & (((data[:,7].astype(np.int) != 2) & (data[:,7].astype(np.int) != 4)) | (data[:,8].astype(np.int) >= 2))
        #Filter out penalties:
        burkefilterbool = burkefilterbool & ((data[:,4] == 'RUSH') | (data[:,4] == 'PASS'))
        #Filter out things within 35 yards of the endzone:
        burkefilterbool = burkefilterbool & (data[:,16].astype(np.int) <= 65)

        #Set if we want to use my filter or Burke's:
        #filterbool = burkefilterbool
        
        #Set up rushing and passing plays:
        rushfilterbool = filterbool & (data[:,4] == 'RUSH')
        passfilterbool = filterbool & (data[:,4] == 'PASS')
        rbrushfilterbool = rushfilterbool & (data[:,19] == 'RB')
        
        gooddata = data[filterbool,:]
        goodcumfirstdowns = cumfirstdowns[filterbool]
        goodfirstdownsfromdrive = firstdownsfromdrive[filterbool]

        goodrushdata = data[rushfilterbool,:]
        goodrushcumfirstdowns = cumfirstdowns[rushfilterbool]
        goodrushfirstdownsfromdrive = firstdownsfromdrive[rushfilterbool]

        goodpassdata = data[passfilterbool,:]
        goodpasscumfirstdowns = cumfirstdowns[passfilterbool]
        goodpassfirstdownsfromdrive = firstdownsfromdrive[passfilterbool]

        goodrbrushdata = data[rbrushfilterbool,:]
        goodrbrushcumfirstdowns = cumfirstdowns[rbrushfilterbool]
        goodrbrushfirstdownsfromdrive = firstdownsfromdrive[rbrushfilterbool]
        
        #Now we can split the data:
        firstdown_data = gooddata[goodcumfirstdowns < goodfirstdownsfromdrive,:]
        rushfirstdown_data = goodrushdata[goodrushcumfirstdowns < goodrushfirstdownsfromdrive,:]
        passfirstdown_data = goodpassdata[goodpasscumfirstdowns < goodpassfirstdownsfromdrive,:]
        rbrushfirstdown_data = goodrbrushdata[goodrbrushcumfirstdowns < goodrbrushfirstdownsfromdrive,:]

        print "Number of Pass plays: ",goodpassdata.shape[0]
        print "Number of Run plays: ",goodrushdata.shape[0]
        print "Total number of plays: ",goodpassdata.shape[0]+goodrushdata.shape[0]
        
        testingbool = (rushfirstdown_data[:,14].astype(np.int) == 3) & (rushfirstdown_data[:,15].astype(np.int) > 5)
        #print (rushfirstdown_data[testingbool,:])[:10,:]
        
        #Get a single set of YTGs to use as the bins:
        ytgbins = np.arange(1,(gooddata[:,15].astype(np.int)).max())
        ytgbins = np.append(ytgbins,(gooddata[:,15].astype(np.int)).max()+0.1)#Set right edge of bin
        #Compute distributions:
        allplaydistributions = compute_distributions(gooddata,ytgbins=ytgbins)
        allrushplaydistributions = compute_distributions(goodrushdata,ytgbins=ytgbins)
        allpassplaydistributions = compute_distributions(goodpassdata,ytgbins=ytgbins)
        allrbrushplaydistributions = compute_distributions(goodrbrushdata,ytgbins=ytgbins)
        firstdowndistributions = compute_distributions(firstdown_data,ytgbins=ytgbins)
        rushfirstdowndistributions = compute_distributions(rushfirstdown_data,ytgbins=ytgbins)
        passfirstdowndistributions = compute_distributions(passfirstdown_data,ytgbins=ytgbins)
        rbrushfirstdowndistributions = compute_distributions(rbrushfirstdown_data,ytgbins=ytgbins)
        #Get only distributions with reasonable statistics:
        minsamples = 200
        print "Maximum Poisson Error = {0:.2f}%".format(np.sqrt(minsamples)/float(minsamples))
        goodallplaydistributions = []
        goodallrushplaydistributions = []
        goodallpassplaydistributions = []
        goodallrbrushplaydistributions = []
        goodallqbrushplaydistributions = []
        goodfirstdowndistributions = []
        goodrushfirstdowndistributions = []
        goodpassfirstdowndistributions = []
        goodrbrushfirstdowndistributions = []
        goodqbrushfirstdowndistributions = []
        firstdownpcts = []
        rushfirstdownpcts = []
        passfirstdownpcts = []
        rbrushfirstdownpcts = []
        qbrushfirstdownpcts = []
        firstdownbins = []
        rushfirstdownbins = []
        passfirstdownbins = []
        rbrushfirstdownbins = []
        qbrushfirstdownbins = []
        for i in range(len(allplaydistributions)):
            goodbinsbool = (allplaydistributions[i] >= minsamples)
            goodrushbinsbool = (allrushplaydistributions[i] >= minsamples)
            goodpassbinsbool = (allpassplaydistributions[i] >= minsamples)
            goodrbrushbinsbool = (allrbrushplaydistributions[i] >= minsamples)
            firstdownbins.append(ytgbins[goodbinsbool])
            rushfirstdownbins.append(ytgbins[goodrushbinsbool])
            passfirstdownbins.append(ytgbins[goodpassbinsbool])
            rbrushfirstdownbins.append(ytgbins[goodrbrushbinsbool])
            goodallplaydistributions.append(allplaydistributions[i][goodbinsbool])
            goodallrushplaydistributions.append(allrushplaydistributions[i][goodrushbinsbool])
            goodallpassplaydistributions.append(allpassplaydistributions[i][goodpassbinsbool])
            goodallrbrushplaydistributions.append(allrbrushplaydistributions[i][goodrbrushbinsbool])
            tempallqbrushplays = allrushplaydistributions[i] - allrbrushplaydistributions[i]
            goodqbrushbinsbool = (tempallqbrushplays >= minsamples)
            qbrushfirstdownbins.append(ytgbins[goodqbrushbinsbool])
            tempfirstdownqbrushplays = rushfirstdowndistributions[i] - rbrushfirstdowndistributions[i]
            goodqbrushfirstdowndistributions.append(tempfirstdownqbrushplays[goodqbrushbinsbool])
            goodallqbrushplaydistributions.append(tempallqbrushplays[goodqbrushbinsbool])
            goodfirstdowndistributions.append(firstdowndistributions[i][goodbinsbool])
            goodrushfirstdowndistributions.append(rushfirstdowndistributions[i][goodrushbinsbool])
            goodpassfirstdowndistributions.append(passfirstdowndistributions[i][goodpassbinsbool])
            goodrbrushfirstdowndistributions.append(rbrushfirstdowndistributions[i][goodrbrushbinsbool])
            firstdownpcts.append(100.*goodfirstdowndistributions[i].astype(np.float)/goodallplaydistributions[i])
            rushfirstdownpcts.append(100.*goodrushfirstdowndistributions[i].astype(np.float)/goodallrushplaydistributions[i])
            passfirstdownpcts.append(100.*goodpassfirstdowndistributions[i].astype(np.float)/goodallpassplaydistributions[i])
            rbrushfirstdownpcts.append(100.*goodrbrushfirstdowndistributions[i].astype(np.float)/goodallrbrushplaydistributions[i])
            qbrushfirstdownpcts.append(100.*goodqbrushfirstdowndistributions[i].astype(np.float)/goodallqbrushplaydistributions[i])
            # if i == 2:
            #     print "QB 1st: ",goodqbrushfirstdowndistributions[i]
            #     print "QB all: ",goodallqbrushplaydistributions[i]
            #     print "RB 1st: ",goodrbrushfirstdowndistributions[i]
            #     print "RB all: ",goodallrbrushplaydistributions[i]
            #     print "rush 1st: ",goodrushfirstdowndistributions[i]
            #     print "rush all: ",goodallrushplaydistributions[i]
            #     #print 100.*goodrbrushfirstdowndistributions[i].astype(np.float)/goodallrbrushplaydistributions[i]

        ax = plt.figure().add_subplot(111)
        colors = ['black','blue','red','green']
        shapes = ['o','s','^','*']
        labels = [r'1$^{\mathrm{st}}$ Down',r'2$^{\mathrm{nd}}$ Down',r'3$^{\mathrm{rd}}$ Down',r'4$^{\mathrm{th}}$ Down']
        axes = []
        axeslabels = []
        f = open('firstdownlikelihood_tabulated_all.txt','w')
        f.write("#YTG   % chance of 1st down\n")
        fpass = open('firstdownlikelihood_tabulated_pass.txt','w')
        fpass.write("#YTG   % chance of 1st down\n")
        frush = open('firstdownlikelihood_tabulated_rush.txt','w')
        frush.write("#YTG   % chance of 1st down\n")
        frbrush = open('firstdownlikelihood_tabulated_rbrush.txt','w')
        frbrush.write("#YTG   % chance of 1st down\n")
        for i in range(len(firstdownpcts)):
            f.write("#Down: {0:d}\n".format(i+1))
            np.savetxt(f,zip(firstdownbins[i],firstdownpcts[i]),fmt='%d %.2f')
            f.write("\n")
            fpass.write("#Down: {0:d}\n".format(i+1))
            if len(passfirstdownbins[i]) > 0:
                np.savetxt(fpass,zip(passfirstdownbins[i],passfirstdownpcts[i]),fmt='%d %.2f')
            fpass.write("\n")
            frush.write("#Down: {0:d}\n".format(i+1))
            if len(rushfirstdownbins[i]) > 0:
                np.savetxt(frush,zip(rushfirstdownbins[i],rushfirstdownpcts[i]),fmt='%d %.2f')
            frush.write("\n")
            frbrush.write("#Down: {0:d}\n".format(i+1))
            if len(rbrushfirstdownbins[i]) > 0:
                np.savetxt(frbrush,zip(rbrushfirstdownbins[i],rbrushfirstdownpcts[i]),fmt='%d %.2f')
            frbrush.write("\n")
            tempax, = ax.plot(firstdownbins[i],firstdownpcts[i],ls='-',marker=shapes[i],color=colors[i],mec=colors[i],mfc=colors[i],ms=5)
            axes.append(tempax)
            axeslabels.append(labels[i])
            ax.plot(rushfirstdownbins[i],rushfirstdownpcts[i],ls='--',color=colors[i])
            ax.plot(passfirstdownbins[i],passfirstdownpcts[i],ls=':',color=colors[i])
        f.close()
        fpass.close()
        frush.close()
        frbrush.close()
        ax.set_xlabel('Yards to Gain')
        ax.set_ylabel('Conversion Percentage')
        allplaylegend, = ax.plot([-100],[-100],ls='-',color='black')
        runplaylegend, = ax.plot([-100],[-100],ls='--',color='black')
        passplaylegend, = ax.plot([-100],[-100],ls=':',color='black')
        ax.set_xlim(0,22)
        ax.set_ylim(0,100)
        l1 = ax.legend(axes,labels,loc='upper right',numpoints=1,prop={'size':10})
        l2 = ax.legend([allplaylegend,runplaylegend,passplaylegend],['All Plays','Runs','Passes'],loc='lower left',prop={'size':10})
        ax.figure.gca().add_artist(l1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.figure.savefig('firstdownlikelihood_all.png')

        ax = plt.figure().add_subplot(111)
        colors = ['black','blue','red','green']
        shapes = ['o','s','^','*']
        labels = [r'1$^{\mathrm{st}}$ Down',r'2$^{\mathrm{nd}}$ Down',r'3$^{\mathrm{rd}}$ Down',r'4$^{\mathrm{th}}$ Down']
        axes = []
        axeslabels = []
        for i in range(len(firstdownpcts)):
            tempax, = ax.plot(firstdownbins[i],firstdownpcts[i],ls='-',marker=shapes[i],color=colors[i],mec=colors[i],mfc=colors[i],ms=5)
            axes.append(tempax)
            axeslabels.append(labels[i])
            ax.plot(rushfirstdownbins[i],rushfirstdownpcts[i],ls='--',color='#D8D8D8')
            ax.plot(passfirstdownbins[i],passfirstdownpcts[i],ls=':',color=colors[i])
            ax.plot(rbrushfirstdownbins[i],rbrushfirstdownpcts[i],ls='--',color=colors[i])
            print "{0:d} down: ".format(i+1),qbrushfirstdownbins[i]
            print "{0:d} down: ".format(i+1),qbrushfirstdownpcts[i]
        ax.set_xlabel('Yards to Go')
        ax.set_ylabel('Conversion Percentage')
        allplaylegend, = ax.plot([-100],[-100],ls='-',color='black')
        runplaylegend, = ax.plot([-100],[-100],ls='--',color='black')
        passplaylegend, = ax.plot([-100],[-100],ls=':',color='black')
        
        ax.set_xlim(0,22)
        ax.set_ylim(0,100)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        l1 = ax.legend(axes,labels,loc='upper right',numpoints=1,prop={'size':10})
        l2 = ax.legend([allplaylegend,runplaylegend,passplaylegend],['All Plays','Runs','Passes'],loc='lower left',prop={'size':10})
        ax.figure.gca().add_artist(l1)
        ax.figure.savefig('firstdownlikelihood_qbcorr.png')

        
    except mdb.Error, e:
        print "Error %d: %s" % (e.args[0],e.args[1])
        sys.exit(1)


    finally:
        if con:
            con.close()
