# -*- coding: utf-8 -*-

# Copyright 2016 All Rights Reserved.
#
# ==============================================================================

from urllib import urlretrieve
import uuid
from scoop import futures
import pandas as pd
import time
import cPickle as pickle
import sys, getopt

ALLOWED = ['png', 'jpg', 'jpeg', 'bmp']
FTARGET = './data/'
SRC = 'dn.txt'

class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

def generate_unique_id():
    return str(uuid.uuid4().time)

def download(url):
    """
    Bild herunterladen. URL muss direkt auf Datei referenzieren. zB https://www.abc.de/xyz/fname.jpg
    Es werden nur erwartete Dateiendungen akzeptiert (siehe ALLOWED)
    :param url:
    :return:
    """
    try:
        filename = url.split('/')[-1]
        ext = filename.split('.')[1].lower()
        if ext not in ALLOWED:
            print "[x] FEHLER. Erlaubte Dateiendungen: ", ALLOWED
            print "[x] URL: ", url
            print "[x] Dateiname extrahiert: ", ext, "in", filename
            print
            return None,
        uuid = generate_unique_id()
        fpath = FTARGET + uuid + '|_-_|' + filename
        filename, _ = urlretrieve(url, fpath)
        return filename, uuid, url
    except:
        import traceback;
        print "[x] URL: ", url
        traceback.print_exc()
        print
    return None,

def dbatch(xlist):
    return map(lambda x: download(x), xlist)

def read_data():
    xdata = None;
    with open(SRC, 'r') as f:
        xdata = map(lambda x: x.strip(), f.readlines())
    return xdata

def batch(iterable, n=1):
    """
    - simple list batching, http://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    for x in batch(range(0, 100), 5):
        print x
    :param iterable:
    :param n:
    :return:
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def download_parallel(batchSize = 10):
    """
    Verwendet SCOOP, um Daten parallel herunterzuladen
    :param batchSize:
    :return:
    """
    xdata = pd.Series(read_data())
    xbatches = map(lambda x: xdata.ix[x].values,
                   list(batch(range(0, len(xdata)), batchSize)))
    results = list(futures.map(dbatch, xbatches))

    if len(results) != 0:
        xsum = reduce(lambda a, b: a + b, map(lambda z: len(z), map(lambda y: filter(lambda x: x[0] is not None, y), results)))
        if xsum != len(xdata):
            print "[x] %d Fehler aufgetreten, siehe log.."%(len(xdata) - xsum)

    return results

def main(argv):
    """
    :param -b N: batch size N
    :return:
    """
    bSize = None
    try:
        opts, args = getopt.getopt(argv,"hb:",['batchSize='])
    except getopt.GetoptError:
        print 'k2.py -b <batchSize>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'python -m scoop -n 7 k2.py -b <batchSize>'
            sys.exit()
        elif opt in ("-b", "--batchSize"):
            bSize = int(arg)
    print
    print "[x] batchSize: ", bSize
    if bSize is None:
        print 'python -m scoop -n <nCores> k2.py -b <batchSize>'
        return

    xt = Timer()
    results = download_parallel(bSize)
    with open("xres.pickle", "wb") as f:
        pickle.dump(results, f)
    print "**********************************************"
    print "[x] Dauer: " + xt.get_time_hhmmss();
    print "**********************************************"

if __name__ == '__main__':
    main(sys.argv[1:])