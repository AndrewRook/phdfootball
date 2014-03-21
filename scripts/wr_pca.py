import numpy as np
import MySQLdb as mdb
import matplotlib.pyplot as plt

def query_db(min_recs = 200):
    con = ''
    data = []
    #Querying the database:
    try:
        con = mdb.connect(read_default_file='~/.my.cnf',read_default_group='aadb')
        cur = con.cursor()

        #Get the table:
        cur.execute('select roster.fname,roster.lname,roster.pos1,pass.yds,pass.trg from (select pass.trg,COUNT(pass.yds) as numrecs from pass join comps on pass.pid=comps.pid join roster on pass.trg=roster.player group by pass.trg having numrecs > {0:d}) as t join pass on pass.trg = t.trg join comps on pass.pid=comps.pid join roster on pass.trg = roster.player'.format(min_recs))
        data = np.array(cur.fetchall())
        return data
    except mdb.Error, e:
        print "Error %d: %s" % (e.args[0],e.args[1])
        sys.exit(1)
    finally:
        if con:
            con.close()

def histogram_data(data):
    #Make histograms of all the receptions for each player:
    unique_players,index_unique,inv_unique = np.unique(data[:,4],return_index=True,return_inverse=True)
    unique_player_integer_ids = np.arange(len(unique_players))
    unique_player_names = np.core.defchararray.add(np.core.defchararray.add(data[index_unique,0],' '),data[index_unique,1])
    player_integer_ids = unique_player_integer_ids[inv_unique]
    reception_yds = data[:,3].astype(np.int)
    reception_bins = np.arange(reception_yds.min()-0.5,reception_yds.max()+1.5)
    player_integer_bins = np.arange(player_integer_ids.min()-0.5,player_integer_ids.max()+1.5)
    hist,xedges,yedges = np.histogram2d(player_integer_ids,reception_yds,bins=[player_integer_bins,reception_bins])
    #The first axis of hist gives the player ids, the second axis shows the histogram of receptions for that player
    return unique_player_names,hist
    # print hist.shape,reception_bins.shape,player_integer_bins.shape
    # ax = plt.figure().add_subplot(111)
    # ax.bar(reception_bins[:-1]+0.5,hist[0,:],width=reception_bins[1:]-reception_bins[:-1],color='gray',edgecolor='black',alpha=0.5)
    # ax.figure.savefig('test.png',dpi=300)

def plot_total_hist(data,filename):
    reception_yds = data[:,3].astype(np.int)
    reception_bins = np.arange(reception_yds.min()-0.5,reception_yds.max()+1.5)
    hist,edges = np.histogram(reception_yds,bins=reception_bins)
    ax = plt.figure().add_subplot(111)
    ax.bar(reception_bins[:-1]+0.5,hist,width=reception_bins[1:]-reception_bins[:-1],color='gray',edgecolor='black',alpha=0.5)
    ax.figure.savefig(filename,dpi=300)

if __name__ == "__main__":
    data = query_db()
    #[:,0] = First name, [:,1] = Last name, [:,2] = Position, [:,3] = Catch+run yards, [:,4] = Player ID
    plot_total_hist(data,'wr_pca_allhist.png')
    names,reception_stats = histogram_data(data)
    print names.shape,reception_stats.shape
