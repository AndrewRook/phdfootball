import sys
import numpy as np
import MySQLdb as mdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import RandomizedPCA

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
    unique_player_positions = data[index_unique,2]
    unique_player_names = np.core.defchararray.add(np.core.defchararray.add(data[index_unique,0],' '),data[index_unique,1])
    player_integer_ids = unique_player_integer_ids[inv_unique]
    reception_yds = data[:,3].astype(np.int)
    reception_bins = make_bins(reception_yds)
    player_integer_bins = np.arange(player_integer_ids.min()-0.5,player_integer_ids.max()+1.5)
    hist,xedges,yedges = np.histogram2d(player_integer_ids,reception_yds,bins=[player_integer_bins,reception_bins])
    #The first axis of hist gives the player ids, the second axis shows the histogram of receptions for that player
    return unique_player_names,unique_player_positions,reception_bins,hist

def make_bins(data):
    #Nonuniform bins, ~constant signal per bin:
    sorteddata = np.sort(data)
    min_numperbin = 50
    nonuniformbins = np.unique(np.append(sorteddata[::min_numperbin]-0.5,sorteddata[-1]+0.5))
    #uniform bins:
    uniformbins =  np.arange(data.min()-0.5,data.max()+1.5)
    return uniformbins
    
    

def plot_total_hist(data,filename):
    reception_yds = data[:,3].astype(np.int)
    reception_bins = np.arange(reception_yds.min()-0.5,reception_yds.max()+1.5)
    hist,edges = np.histogram(reception_yds,bins=reception_bins)
    ax = plt.figure().add_subplot(111)
    ax.patch.set_facecolor('#F0F0F0')
    ax.bar(reception_bins[:-1]+0.5,hist,width=reception_bins[1:]-reception_bins[:-1],color='gray',edgecolor='black')
    ax.set_xlabel('Reception Yards',fontsize=12,weight='semibold')
    ax.set_ylabel('# of Plays',fontsize=12,weight='semibold')
    ax.set_title('All Receptions',fontsize=14,weight='semibold')
    ax.text(ax.get_xlim()[1]-(ax.get_xlim()[1]-ax.get_xlim()[0])/40.,ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/30.,'phdfootball.blogspot.com',color='white',alpha=1,fontsize=14,weight='semibold',va='top',ha='right')
    ax.figure.savefig(filename,dpi=300)

def compute_pca(reception_stats,n_components=5):
    reception_mean = reception_stats.mean(axis=0)
    pca = RandomizedPCA(n_components-1)
    pca.fit(reception_stats)
    pca_components = np.vstack([reception_mean,pca.components_])

    return pca,pca_components

def compute_pca_coeffs(pca_components,data):
    #Compute the coefficients theta_ij to reconstruct the data from the PCA components:
    meansubdata = data - pca_components[0,:]
    thetaij = np.sum((pca_components[1:,:]).reshape((pca_components[1:,:]).shape+(1,))*(meansubdata.T),axis=1)
    return thetaij

def reconstruct_data(pca_components,coeffs,dataid,numcoeffs):
    if dataid >= coeffs.shape[1]:
        raise Exception("dataid {0:d} outside of range of data ({1:d})".format(dataid,coeffs.shape[1]))
    if numcoeffs > pca_components.shape[0]-1:
        print "Warning: Requested number of coefficients ({0:d}) > available number of coefficients ({1:d})".format(numcoeffs,pca_components.shape[0])
        numcoeffs = pca_components.shape[0]-1
    reconstructed_data = pca_components[0,:]
    if numcoeffs > 1:
        reconstructed_data += np.sum(coeffs[0:numcoeffs-1,dataid].reshape(coeffs[0:numcoeffs-1,dataid].shape+(1,))*pca_components[1:numcoeffs,:],axis=0)
    return reconstructed_data

def print_extreme_coeffs(coeffs,names,positions,coeff_number,indexstring,selectedposition):
    sortedcoeff_indexes = np.argsort(coeffs[coeff_number,:])
    sortedcoeffs = coeffs[coeff_number,sortedcoeff_indexes]
    sortednames = names[sortedcoeff_indexes]
    sortedpositions = positions[sortedcoeff_indexes]
    positioncoeffs = sortedcoeffs[(sortedpositions == selectedposition)]
    positionnames = sortednames[(sortedpositions == selectedposition)]

    selectedcoeffs = np.atleast_1d(positioncoeffs[indexstring])
    selectednames = np.atleast_1d(positionnames[indexstring])
    
    for i in range(len(selectedcoeffs)):
        print "{0:.3e}: {1:s}".format(selectedcoeffs[i],selectednames[i])
    

if __name__ == "__main__":
    data = query_db()
    #[:,0] = First name, [:,1] = Last name, [:,2] = Position, [:,3] = Catch+run yards, [:,4] = Player ID
    plot_total_hist(data,'wr_pca_allhist.png')
    names,positions,reception_bins,reception_stats = histogram_data(data)
    #Normalize the receptions:
    reception_sums = np.sum(reception_stats,axis=1).astype(np.float)
    reception_stats /= reception_sums.reshape(reception_sums.shape+(1,))
    print reception_stats.shape
    pca,pca_components = compute_pca(reception_stats,n_components=15)

    #Compute the coefficients and plot an example reconstruction:
    coeffs = compute_pca_coeffs(pca_components,reception_stats)
    reconstruct_id = 0
    numcoeffs = 10
    reconstructed_receptions = reconstruct_data(pca_components,coeffs,reconstruct_id,numcoeffs)
    ax = plt.figure().add_subplot(111)
    ax.patch.set_facecolor('#F0F0F0')
    ax.bar(reception_bins[:-1]+0.5,reception_stats[reconstruct_id,:]*reception_sums[reconstruct_id],width=reception_bins[1:]-reception_bins[:-1],facecolor='black',edgecolor='black',alpha=1,label='Data')
    ax.bar(reception_bins[:-1]+0.5,reconstructed_receptions*reception_sums[reconstruct_id],width=reception_bins[1:]-reception_bins[:-1],facecolor='none',edgecolor='red',lw=1,label='{0:d} Component Reconstruction'.format(numcoeffs))
    ax.set_xlabel('Reception Yards',fontsize=12,weight='semibold')
    ax.set_ylabel('# of Plays',fontsize=12,weight='semibold')
    ax.set_title('Sample PCA Reconstruction',fontsize=14,weight='semibold')
    ax.set_xlim(reception_bins[0],reception_bins[-1])
    ax.set_ylim(0,ax.get_ylim()[1])
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])/40.,ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])/20.,"{0:s}".format(names[reconstruct_id],positions[reconstruct_id]),ha='left',va='center',fontsize=12,weight='semibold')
    ax.text(ax.get_xlim()[1]-(ax.get_xlim()[1]-ax.get_xlim()[0])/40.,ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])/10.,'phdfootball.blogspot.com',color='white',alpha=1,fontsize=14,weight='semibold',va='bottom',ha='right')
    ax.legend(loc='upper right',numpoints=1,prop={'size':10})
    ax.figure.savefig('wr_pca_samplereconstruction.png',dpi=300)

    #Make a 3D plot of the coefficients:
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(coeffs[0,(positions=='WR')],coeffs[1,(positions=='WR')],coeffs[2,(positions=='WR')],c='red',marker='o')
    # ax.scatter(coeffs[0,(positions=='TE')],coeffs[1,(positions=='TE')],coeffs[2,(positions=='TE')],c='green',marker='s')
    # ax.scatter(coeffs[0,(positions=='RB')],coeffs[1,(positions=='RB')],coeffs[2,(positions=='RB')],c='blue',marker='^')
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    # plt.show()
    smallestvalues = np.s_[:5]
    largestvalues = np.s_[-5:]
    print "Most WR-Like Tight Ends"
    print_extreme_coeffs(coeffs,names,positions,0,smallestvalues,'TE')
    print "Least WR-Like Tight Ends"
    print_extreme_coeffs(coeffs,names,positions,0,largestvalues,'TE')

    print "Test"
    smallestvalues = np.s_[:10]
    largestvalues = np.s_[-10:]
    print_extreme_coeffs(coeffs,names,positions,2,smallestvalues,'WR')
    print ""
    print_extreme_coeffs(coeffs,names,positions,2,largestvalues,'WR')
    
    #Plot Histograms of Coefficient 1 for WRs, TEs, and RBs:
    ax = plt.figure().add_subplot(111)
    ax.patch.set_facecolor('#F0F0F0')
    poscolors = ['#fbb4ae','#b3cde3','#ccebc5']
    positionlist = ['WR','TE','RB']
    posbins = np.linspace(coeffs[0,:].min(),coeffs[0,:].max(),30)
    for i in range(len(positionlist)):
        hist,edges = np.histogram(coeffs[0,(positions == positionlist[i])],bins=posbins)
        ax.bar(edges[:-1],hist,width=edges[1:]-edges[:-1],color=poscolors[i],edgecolor='black',alpha=0.75,label='{0:s}s'.format(positionlist[i]))
    ax.set_xlabel('First Coefficient',fontsize=12,weight='semibold')
    ax.set_ylabel('# of Players',fontsize=12,weight='semibold')
    ax.set_ylim(0,ax.get_ylim()[1]+1)
    ax.legend(loc='upper left',prop={'size':10})
    ax.set_title('Position Classification',fontsize=14,weight='semibold')
    ax.text(ax.get_xlim()[1]-(ax.get_xlim()[1]-ax.get_xlim()[0])/40.,ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/30.,'phdfootball.blogspot.com',color='white',alpha=1,fontsize=14,weight='semibold',va='top',ha='right')
    ax.figure.savefig('wr_pca_classification.png',dpi=300)
    
    #Plot the eigenvalues:
    evals = pca.explained_variance_ratio_
    evals_cs = evals.cumsum()
    evals_cs /= evals_cs[-1]
    evals_xvals = np.arange(len(evals))+1
    labelx=-0.08
    axup = plt.figure().add_subplot(211,xscale='linear',yscale='log')
    axup.patch.set_facecolor('#F0F0F0')
    axup.grid()
    axup.plot(evals_xvals,evals,ls='-',color='black',lw=2)
    axup.set_ylabel('Normalized Eigenvalues',fontsize=12,weight='semibold')
    axup.set_title('PCA Variance with Additional Components',fontsize=14,weight='semibold')
    axup.xaxis.set_major_formatter(plt.NullFormatter())
    axup.set_ylim(3E-3,1)
    axup.set_xlim(evals_xvals[0],evals_xvals[-1])
    axup.yaxis.set_label_coords(labelx,0.5)
    axdown = axup.figure.add_subplot(212,xscale='linear')
    axdown.patch.set_facecolor('#F0F0F0')
    axdown.grid()
    axdown.plot(evals_xvals,evals_cs*100.,color='black',lw=2)
    axdown.set_xlabel('Component Number',fontsize=12,weight='semibold')
    axdown.set_ylabel('% Variance Explained',fontsize=12,weight='semibold')
    axdown.set_ylim(30,99.99)
    axdown.set_xlim(evals_xvals[0],evals_xvals[-1])
    axdown.yaxis.set_label_coords(labelx,0.5)
    axdown.text(axdown.get_xlim()[1]-(axdown.get_xlim()[1]-axdown.get_xlim()[0])/40.,axdown.get_ylim()[0]+(axdown.get_ylim()[1]-axdown.get_ylim()[0])/40.,'phdfootball.blogspot.com',color='white',alpha=1,fontsize=14,weight='semibold',va='bottom',ha='right')
    axup.figure.subplots_adjust(hspace=0)
    axup.figure.savefig('wr_pca_eigenvalues.png',dpi=300)

    #Plot the first few components:
    num_components_to_plot = 5
    component_colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','ffd92f']
    if len(pca_components) < num_components_to_plot:
        num_components_to_plot = len(pca_components)
    fig,axs = plt.subplots(num_components_to_plot,sharex=True)
    axall = plt.figure().add_subplot(111)
    axall.set_title('Normalized Reception PCA Components',fontsize=14,weight='semibold')
    axall.patch.set_facecolor('#F0F0F0')
    axall.set_ylabel('Component',fontsize=12,weight='semibold')
    axall.set_xlabel('Reception Yards',fontsize=12,weight='semibold')
    labelx = -0.08
    for i in range(len(axs)):
        axs[i].patch.set_facecolor('#F0F0F0')
        axs[i].plot(reception_bins[:-1]+0.5,pca_components[i],ls='-',lw=2,color=component_colors[i])
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,4,5,10]))
        axs[i].set_ylabel(i+1,fontsize=12,weight='semibold')
        axs[i].set_xlim(reception_bins[0]+0.5,reception_bins[-1]-0.5)
        axs[i].set_ylim(axs[i].get_ylim()[0]+0.001,axs[i].get_ylim()[1]-0.001)
        axs[i].yaxis.set_label_coords(labelx,0.5)
        axall.plot(reception_bins[:-1]+0.5,pca_components[i]/np.max(pca_components[i]),ls='-',lw=2,color=component_colors[i],label='Component {0:d}'.format(i+1))
    axs[0].set_title('Reception PCA Components',fontsize=14,weight='semibold')
    axs[-1].text(axs[i].get_xlim()[1]-(axs[i].get_xlim()[1]-axs[i].get_xlim()[0])/40.,axs[i].get_ylim()[0]+(axs[i].get_ylim()[1]-axs[i].get_ylim()[0])/20.,'phdfootball.blogspot.com',color='white',alpha=1,fontsize=14,weight='semibold',va='bottom',ha='right')
    axs[-1].set_xlabel('Reception Yards',fontsize=12,weight='semibold')
    fig.subplots_adjust(hspace=0)
    fig.text(0.02, 0.5, 'Component', ha='center', va='center', rotation='vertical',fontsize=14,weight='semibold')
    fig.savefig('wr_pca_components.png',dpi=300)

    axall.set_ylim(-0.85,1.02)
    axall.set_xlim(reception_bins[0]+0.5,reception_bins[-2]-0.5)
    axall.legend(loc='upper right',prop={'size':10})
    axall.text(axall.get_xlim()[1]-(axall.get_xlim()[1]-axall.get_xlim()[0])/40.,axall.get_ylim()[0]+(axall.get_ylim()[1]-axall.get_ylim()[0])/40.,'phdfootball.blogspot.com',color='white',alpha=1,fontsize=14,weight='semibold',va='bottom',ha='right')
    axall.figure.savefig('wr_pca_components_stacked.png',dpi=300)
