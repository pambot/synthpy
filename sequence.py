import numpy as np
import pandas as pd
from scipy.stats import randint
from collections import Counter


class FASTA(object):
    def __init__(self):
        self.fasta = {}
    
    def __contains__(self, key):
        return key in self.fasta
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.fasta[key]
        elif len(key) == 2 and (isinstance(key[0], str) and 
                isinstance(key[1], (int, slice))):
            chrom, slice_ = key
            return self.fasta[chrom][:, slice_]
        elif len(key) == 2 and (isinstance(key[0], str) and 
                isinstance(key[1], (list, tuple)) and 
                all(isinstance(k, slice) for k in key[1])):
            chrom, slices = key
            exons = []
            for slice_ in slices:
                exons.append(self.fasta[chrom][:, slice_])
            return np.hstack(exons)
        else:
            raise KeyError('Invalid key for __getitem__')
    
    def __setitem__(self, key, value):
        if isinstance(key, str) and (isinstance(value, np.ndarray) and 
                value.dtype == np.float64):
            self.fasta[key] = value
        elif len(key) == 2 and (isinstance(key[0], str) and 
                isinstance(key[1], (int, slice))):
            chrom, slice_ = key
            return self.fasta[chrom][slice_]
        else:
            raise KeyError('Invalid key or value for __setitem__')
    
    @property
    def entries(self):
        return self.fasta.keys()
    
    def from_file(self, fname):
        with open(fname, 'r') as f:
            l = f.readline().rstrip('\n')
            curr_id = l[1:]
            curr_seq = ''
            for l in f:
                l = l.rstrip('\n')
                if l[0] in ('A', 'C', 'G', 'T', 'N'):
                    curr_seq += l
                elif l[0] == '>':
                    self.fasta[curr_id] = self.seq2matrix(curr_seq)
                    curr_id = l[1:]
                    curr_seq = ''
                else:
                    raise ValueError('Not a valid FASTA sequence file.')
            else:
                self.fasta[curr_id] = self.seq2matrix(curr_seq)
    
    @staticmethod
    def seq2matrix(seq):
        alpha_seq = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        matrix = np.zeros((4, len(seq)))
        for i, s in enumerate(seq):
            try:
                matrix[alpha_seq[s], i] = 1
            except KeyError:
                if s == 'N':
                    pass
                else:
                    raise ValueError('{0} is not in the alphabet.'.format(s))
        return matrix
    
    @staticmethod
    def matrix2seq(matrix):
        seq_array = []
        for i in range(matrix.shape[1]):
            freqs = matrix[:, i]
            if any(freqs):
                nuc = np.random.choice(('A', 'C', 'G', 'T'), p=freqs)
            else:
                nuc = 'N'
            seq_array.append(nuc)
        seq = ''.join(seq_array)
        return seq
    
    def bed_regions(self, bed):
        fasta = FASTA()
        slices = bed.get_slices()
        if 'name' in bed.columns:
            names = bed['name']
        else:
            names = map(str, range(len(bed.index)))
        if 'strand' in bed.columns:
            strands = bed['strand']
        else:
            strands = ['+'] * len(bed.index)
        for name, strand, slice_ in zip(names, strands, slices):
            if strand == '+':    
                fasta[name] = self.__getitem__(slice_)
            else:
                fasta[name] = self.__getitem__(slice_)[::-1, ::-1]
        return fasta
    
    def generate_fastq(self, bed, read_len=50, num_entries=1000,
                       bed_distr=randint, 
                       nuc_distr=randint):
        fasta = self.bed_regions(bed)
        r_start = read_len // 2
        r_end = read_len - r_start
        qual_str = ('B' * read_len + 'AAA@@@??')[-read_len:] # TODO: replace
        fastq = []
        bed_inds = bed_distr.rvs(0, bed.shape[0], size=num_entries)
        ind_counts = Counter(bed_inds)
        c = 0
        for b_ind in ind_counts.keys():
            b = bed[b_ind]
            bed_matrix = fasta[b['name']]
            b_size = bed_matrix.shape[1]
            nuc_inds = nuc_distr.rvs(r_start, 
                                     b_size - r_end, 
                                     size=ind_counts[b_ind])
            
            for n_ind in nuc_inds:
                title = '@SYNTHPY:{0}'.format(c)
                r_region = slice(n_ind - r_start, n_ind + r_end)
                seq = self.matrix2seq(bed_matrix[:, r_region])
                fastq.extend((title, seq, '+', qual_str))
                c += 1
        return fastq


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


# test the API
fasta = FASTA()
fasta.from_file('test.fa')

bed = BED(use_exons=True)
bed.from_file('test.bed')

fastq = fasta.generate_fastq(bed, read_len=10, num_entries=20)

for fq in fastq:
    print(fq)









