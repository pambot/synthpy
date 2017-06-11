import numpy as np
import pandas as pd


class BED(object):
    def __init__(self, use_exons=False):
        self.bed = None
        self.use_exons = use_exons
    
    def __str__(self):
        return str(self.bed)
    
    def __getitem__(self, key):
        if isinstance(key, (int, np.int64)):
            return self.bed.ix[key]
        elif isinstance(key, str):
            return self.bed[key]
        else:
            raise KeyError('Invalid key for __getitem__')
    
    def __setitem__(self, key, value):
        if isinstance(key, (int, np.int64)) and (isinstance(value, dict) and
                sorted(value.keys()) == sorted(self.columns)):
            self.bed.ix[key] = value
        else:
            raise KeyError('Invalid key or value for __setitem__')
        return
    
    @property
    def shape(self):
        return len(self.bed.index), len(self.bed.columns)
    
    @property
    def index(self):
        return self.bed.index
    
    @property
    def columns(self):
        return self.bed.columns
    
    def from_file(self, fname):
        self.bed = pd.read_csv(fname, sep='\t', header=None) 
        columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score',
                   'strand', 'thickStart', 'thickEnd', 'itemRgb', 
                   'blockCount', 'blockSizes', 'blockStarts']
        self.bed.columns = columns[:self.shape[1] + 1]
        return
    
    def get_bed_slice(self, ind):
        b = self.bed.ix[ind]
        return b['chrom'], slice(b['chromStart'], b['chromEnd'])
    
    def get_exon_slice(self, ind):
        slices = []
        b = self.bed.ix[ind]
        exons = map(int, b['blockStarts'].split(','))
        sizes = map(int, b['blockSizes'].split(','))
        for exon_start, exon_size in zip(exons, sizes):
            curr_start = b['chromStart'] + exon_start
            curr_end = curr_start + exon_size
            slices.append(slice(curr_start, curr_end))
        return b['chrom'], slices
    
    def get_slices(self):
        slices = []
        if not self.use_exons:
            for ind in self.bed.index:
                slices.append(self.get_bed_slice(ind))
        elif self.use_exons and 'blockStarts' in self.columns:
            for ind in self.bed.index:
                slices.append(self.get_exon_slice(ind))
        return slices