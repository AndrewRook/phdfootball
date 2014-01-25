import MySQLdb as mdb
import sys
import numpy as np
import string

if __name__ == '__main__':
    """
    load.py is a program to load the Armchair Analysis database from the CSV files you download it as into a MySQL database. It takes one positional argument, which is the directory where all the CSV files are stored
    """

    if len(sys.argv) != 2:
        sys.exit("Syntax: [Directory where CSV files are located]")
    directory = sys.argv[1]
        
    #Define table names:
    table = ['blocks','comps','convs','core','dbacks','defense','fdowns','fgxp','fumbles','games','ints','kickers','kickoffs','knees','nohuddle','offense','oline','pass','penalties','punts','roster','rush','sacks','safeties','scoring','shotgun','spikes','splays','tackles','team']
    #Define table schema:
    initschema = ["CREATE TABLE IF NOT EXISTS `blocks` (`PID` int(7) NOT NULL,`BLK` char(6) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Blocked punts, fieldgoals, etc.'",
    "CREATE TABLE IF NOT EXISTS `comps` (`PID` int(7) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Pass Completions'",
    "CREATE TABLE IF NOT EXISTS `convs` (`PID` int(7) NOT NULL,`CONV` char(1) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='2 Point Conversion Attempts (Y=Success, N=Fail)'",
    "CREATE TABLE IF NOT EXISTS `core` (`GID` int(5) NOT NULL,`PID` int(7) NOT NULL,`OFF` varchar(3) NOT NULL,`DEF` varchar(3) NOT NULL,`TYPE` varchar(4) NOT NULL,`DSEQ` tinyint(2) NOT NULL,`LEN` tinyint(2) NOT NULL,`QTR` tinyint(1) NOT NULL,`MIN` tinyint(2) NOT NULL,`SEC` tinyint(2) NOT NULL,`PTSO` tinyint(2) NOT NULL,`PTSD` tinyint(2) NOT NULL,`TIMO` tinyint(2) NOT NULL,`TIMD` tinyint(2) NOT NULL,`DWN` tinyint(1) NOT NULL,`YTG` tinyint(2) NOT NULL,`YFOG` tinyint(2) NOT NULL,`ZONE` tinyint(1) NOT NULL,`OLID` int(5) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `dbacks` (`PID` int(7) NOT NULL,`DFB` char(6) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Defensive Backs that were listed as the defender on Passes'",
    "CREATE TABLE IF NOT EXISTS `defense` (`UID` int(6) NOT NULL,`GID` int(5) NOT NULL,`PLAYER` char(6) NOT NULL,`TCK` decimal(3,1) NOT NULL,`SCK` decimal(2,1) NOT NULL,`SAF` tinyint(1) NOT NULL,`BLK` tinyint(1) NOT NULL,`INT` tinyint(1) NOT NULL,`PDEF` tinyint(1) NOT NULL,`FREC` tinyint(1) NOT NULL,`TDD` tinyint(1) NOT NULL,`PENY` tinyint(2) NOT NULL,`FPTS` decimal(3,1) NOT NULL,UNIQUE KEY `UID` (`UID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `fdowns` (`PID` int(7) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Plays that resulted in a First Down'",
    "CREATE TABLE IF NOT EXISTS `fgxp` (`PID` int(7) NOT NULL,`FGXP` char(2) NOT NULL,`FKICKER` char(6) NOT NULL,`DIST` tinyint(2) NOT NULL,`GOOD` char(1) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `fumbles` (`PID` int(7) NOT NULL,`FUM` char(6) NOT NULL,`RECV` char(6) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `games` (`GID` int(5) NOT NULL,`SEAS` int(4) NOT NULL,`WK` tinyint(2) NOT NULL,`DAY` varchar(3) NOT NULL,`V` varchar(3) NOT NULL,`H` varchar(3) NOT NULL,`STAD` varchar(45) NOT NULL,`TEMP` varchar(4) DEFAULT NULL,`HUMD` varchar(4) DEFAULT NULL,`WSPD` varchar(4) DEFAULT NULL,`WDIR` varchar(4) DEFAULT NULL,`COND` varchar(15) DEFAULT NULL,`SURF` varchar(25) NOT NULL,`OU` tinyint(2) NOT NULL,`SPRV` decimal(3,1) NOT NULL,`PTSV` tinyint(2) NOT NULL,`PTSH` tinyint(2) NOT NULL,UNIQUE KEY `GID` (`GID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `ints` (`PID` int(7) NOT NULL,`INT` char(6) NOT NULL,`YDS` tinyint(3) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Interceptions'",
    "CREATE TABLE IF NOT EXISTS `kickers` (`UID` int(6) NOT NULL,`GID` int(5) NOT NULL,`PLAYER` char(6) NOT NULL,`PAT` tinyint(1) NOT NULL,`FGS` tinyint(1) NOT NULL,`FGM` tinyint(1) NOT NULL,`FGL` tinyint(1) NOT NULL,`FPTS` decimal(3,1) NOT NULL,UNIQUE KEY `UID` (`UID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='FGS: 0 - 39 yds; FGM: 40 - 49 yds; FGL: 50+ yds'",
    "CREATE TABLE IF NOT EXISTS `kickoffs` (`PID` int(7) NOT NULL,`KICKER` char(6) NOT NULL,`KGRO` tinyint(2) NOT NULL,`KNET` tinyint(2) NOT NULL,`KTB` char(1) NOT NULL,`KR` char(6) NOT NULL,`KRY` tinyint(3) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `knees` (`PID` int(7) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `nohuddle` (`PID` int(7) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `offense` (`UID` int(6) NOT NULL,`GID` int(5) NOT NULL,`PLAYER` char(6) NOT NULL,`PA` tinyint(2) NOT NULL,`PC` tinyint(2) NOT NULL,`PY` int(3) NOT NULL,`INT` tinyint(1) NOT NULL,`TDP` tinyint(1) NOT NULL,`RA` tinyint(2) NOT NULL,`RY` int(3) NOT NULL,`TDR` tinyint(1) NOT NULL,`REC` tinyint(2) NOT NULL,`RECY` int(3) NOT NULL,`TDRE` tinyint(1) NOT NULL,`FUML` tinyint(1) NOT NULL,`PENY` tinyint(2) NOT NULL,`FPTS` decimal(3,1) NOT NULL,UNIQUE KEY `UID` (`UID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `oline` (`OLID` int(5) NOT NULL,`LT` char(6) NOT NULL,`LG` char(6) NOT NULL,`C` char(6) NOT NULL,`RG` char(6) NOT NULL,`RT` char(6) NOT NULL,UNIQUE KEY `OLID` (`OLID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `pass` (`PID` int(7) NOT NULL,`PSR` char(6) NOT NULL,`TRG` char(6) NOT NULL,`LOC` char(2) NOT NULL,`YDS` tinyint(3) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 ROW_FORMAT=COMPACT",
    "CREATE TABLE IF NOT EXISTS `penalties` (`UID` int(6) NOT NULL,`PID` int(7) NOT NULL,`PTM` varchar(3) NOT NULL,`PEN` char(6) NOT NULL,`DESC` varchar(40) NOT NULL,`CAT` tinyint(1) NOT NULL,`PEY` tinyint(2) NOT NULL,`PDO` char(1) NOT NULL,UNIQUE KEY `UID` (`UID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `punts` (`PID` int(7) NOT NULL,`PUNTER` char(6) NOT NULL,`PGRO` tinyint(2) NOT NULL,`PNET` tinyint(2) NOT NULL,`PTS` char(1) NOT NULL,`PR` char(6) NOT NULL,`PRY` tinyint(3) NOT NULL,`PFC` char(1) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `roster` (`PLAYER` char(6) NOT NULL,`FNAME` varchar(20) NOT NULL,`LNAME` varchar(25) NOT NULL,`PNAME` varchar(25) NOT NULL,`POS1` varchar(2) NOT NULL,`POS2` varchar(2) DEFAULT NULL,`HEIGHT` tinyint(2) NOT NULL,`WEIGHT` int(3) NOT NULL,`YOB` int(4) NOT NULL,`DPOS` int(3) NOT NULL,`START` int(4) NOT NULL,`CTEAM` varchar(3) DEFAULT NULL,UNIQUE KEY `PLAYER` (`PLAYER`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `rush` (`PID` int(7) NOT NULL,`BC` char(6) NOT NULL,`DIR` char(2) NOT NULL,`YDS` tinyint(3) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `sacks` (`UID` int(6) NOT NULL,`PID` int(7) NOT NULL,`SK` char(6) NOT NULL,`VALUE` decimal(2,1) NOT NULL,UNIQUE KEY `UID` (`UID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `safeties` (`PID` int(7) NOT NULL,`SAF` char(6) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `scoring` (`PID` int(7) NOT NULL,`PTS` tinyint(1) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `shotgun` (`PID` int(7) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `spikes` (`PID` int(7) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1",
    "CREATE TABLE IF NOT EXISTS `splays` (`PID` int(7) NOT NULL,UNIQUE KEY `PID` (`PID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Successful Plays - See notes for more info.'",
    "CREATE TABLE IF NOT EXISTS `tackles` (`UID` int(7) NOT NULL,`PID` int(7) NOT NULL,`TCK` char(6) NOT NULL,`VALUE` decimal(2,1) NOT NULL,UNIQUE KEY `UID` (`UID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='Special teams tackles are not counted (ie, kickoffs, punts)'",
    "CREATE TABLE IF NOT EXISTS `team` (`TID` int(5) NOT NULL,`GID` int(5) NOT NULL,`TNAME` varchar(3) NOT NULL,`PTS` tinyint(2) NOT NULL,`1QP` tinyint(2) NOT NULL,`2QP` tinyint(2) NOT NULL,`3QP` tinyint(2) NOT NULL,`4QP` tinyint(2) NOT NULL,`RFD` tinyint(2) NOT NULL,`PFD` tinyint(2) NOT NULL,`IFD` tinyint(2) NOT NULL,`RY` int(3) NOT NULL,`RA` tinyint(2) NOT NULL,`PY` int(3) NOT NULL,`PA` tinyint(2) NOT NULL,`PC` tinyint(2) NOT NULL,`SK` tinyint(2) NOT NULL,`INT` tinyint(1) NOT NULL,`FUM` tinyint(1) NOT NULL,`PU` tinyint(2) NOT NULL,`GPY` int(3) NOT NULL,`PR` tinyint(2) NOT NULL,`PRY` int(3) NOT NULL,`KR` tinyint(2) NOT NULL,`KRY` int(3) NOT NULL,`IR` tinyint(1) NOT NULL,`IRY` int(3) NOT NULL,`PEN` int(3) NOT NULL,`TOP` decimal(3,1) NOT NULL,`TD` tinyint(1) NOT NULL,`TDR` tinyint(1) NOT NULL,`TDP` tinyint(1) NOT NULL,`TDT` tinyint(1) NOT NULL,`FGM` tinyint(1) NOT NULL,`FGAT` tinyint(2) NOT NULL,`FGY` int(3) NOT NULL,`RZA` tinyint(2) NOT NULL,`RZC` tinyint(1) NOT NULL,`BRY` int(3) NOT NULL,`BPY` int(3) NOT NULL,`SRP` tinyint(2) NOT NULL,`S1RP` tinyint(2) NOT NULL,`S2RP` tinyint(2) NOT NULL,`S3RP` tinyint(2) NOT NULL,`SPP` tinyint(2) NOT NULL,`S1PP` tinyint(2) NOT NULL,`S2PP` tinyint(2) NOT NULL,`S3PP` tinyint(2) NOT NULL,`LEA` tinyint(2) NOT NULL,`LEY` int(3) NOT NULL,`LTA` tinyint(2) NOT NULL,`LTY` int(3) NOT NULL,`LGA` tinyint(2) NOT NULL,`LGY` int(3) NOT NULL,`MDA` tinyint(2) NOT NULL,`MDY` int(3) NOT NULL,`RGA` tinyint(2) NOT NULL,`RGY` int(3) NOT NULL,`RTA` tinyint(2) NOT NULL,`RTY` int(3) NOT NULL,`REA` tinyint(2) NOT NULL,`REY` int(3) NOT NULL,`R1A` tinyint(2) NOT NULL,`R1Y` int(3) NOT NULL,`R2A` tinyint(2) NOT NULL,`R2Y` int(3) NOT NULL,`R3A` tinyint(2) NOT NULL,`R3Y` int(3) NOT NULL,`QBA` tinyint(2) NOT NULL,`QBY` int(3) NOT NULL,`SLA` tinyint(2) NOT NULL,`SLY` int(3) NOT NULL,`SMA` tinyint(2) NOT NULL,`SMY` int(3) NOT NULL,`SRA` tinyint(2) NOT NULL,`SRY` int(3) NOT NULL,`DLA` tinyint(2) NOT NULL,`DLY` int(3) NOT NULL,`DMA` tinyint(2) NOT NULL,`DMY` int(3) NOT NULL,`DRA` tinyint(2) NOT NULL,`DRY` int(3) NOT NULL,`WR1A` tinyint(2) NOT NULL,`WR1Y` int(3) NOT NULL,`WR3A` tinyint(2) NOT NULL,`WR3Y` int(3) NOT NULL,`TEA` tinyint(2) NOT NULL,`TEY` int(3) NOT NULL,`RBA` tinyint(2) NOT NULL,`RBY` int(3) NOT NULL,`SGA` tinyint(2) NOT NULL,`SGY` int(3) NOT NULL,`P1A` tinyint(2) NOT NULL,`P1Y` int(3) NOT NULL,`P2A` tinyint(2) NOT NULL,`P2Y` int(3) NOT NULL,`P3A` tinyint(2) NOT NULL,`P3Y` int(3) NOT NULL,`SPC` tinyint(2) NOT NULL,`MPC` tinyint(2) NOT NULL,`LPC` tinyint(2) NOT NULL,`Q1RA` tinyint(2) NOT NULL,`Q1RY` int(3) NOT NULL,`Q1PA` tinyint(2) NOT NULL,`Q1PY` int(3) NOT NULL,`LCRA` tinyint(2) NOT NULL,`LCRY` int(3) NOT NULL,`LCPA` tinyint(2) NOT NULL,`LCPY` int(3) NOT NULL,`RZRA` tinyint(2) NOT NULL,`RZRY` int(3) NOT NULL,`RZPA` tinyint(2) NOT NULL,`RZPY` int(3) NOT NULL,`SKY` int(3) NOT NULL,`LBS` decimal(3,1) NOT NULL,`DBS` decimal(3,1) NOT NULL,`SFPY` int(3) NOT NULL,`DRV` tinyint(2) NOT NULL,`NPY` int(3) NOT NULL,`TB` tinyint(1) NOT NULL,`I20` tinyint(1) NOT NULL,`RTD` tinyint(1) NOT NULL,`LNR` decimal(3,1) NOT NULL,`LNP` decimal(3,1) NOT NULL,`LBR` decimal(3,1) NOT NULL,`LBP` decimal(3,1) NOT NULL,`DBR` decimal(3,1) NOT NULL,`DBP` decimal(3,1) NOT NULL,`NHA` tinyint(2) NOT NULL,`S3A` tinyint(2) NOT NULL,`S3C` tinyint(2) NOT NULL,`L3A` tinyint(2) NOT NULL,`L3C` tinyint(2) NOT NULL,`STF` tinyint(2) NOT NULL,`DP` tinyint(2) NOT NULL,`FSP` tinyint(2) NOT NULL,`OHP` tinyint(2) NOT NULL,`PBEP` tinyint(1) NOT NULL,`DLP` tinyint(1) NOT NULL,`DSP` tinyint(1) NOT NULL,`DUM` tinyint(1) NOT NULL,`PFN` tinyint(1) NOT NULL,UNIQUE KEY `TID` (`TID`)) ENGINE=InnoDB DEFAULT CHARSET=latin1"]


    #create csv filenames:
    csvfile = []
    for name in table:
        csvfile.append(directory+string.upper(name)+".csv")

    con = ''

    #Loading in the player database:
    try:
        con = mdb.connect(read_default_file='~/.my.cnf',read_default_group='aadb')
        cur = con.cursor()

        for i in range(len(csvfile)):
            #Dropping the table if it exists:
            cur.execute("drop table if exists {0:s}".format(table[i]))
            #Loading the table in:
            cur.execute(initschema[i])
            cur.execute('load data local infile "{0:s}" into table {1:s} fields terminated by "," optionally enclosed by """" ignore 1 lines'.format(csvfile[i],table[i]))
            #The above gives a bunch of warnings when loading the roster database, but as far as I can tell it's actually fine.
    except mdb.Error, e:
        print "Error %d: %s" % (e.args[0],e.args[1])
        sys.exit(1)


    finally:
        if con:
            con.close()

