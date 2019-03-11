# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 20:51:58 2018

@author: msi
"""

import h5py
import os
from scipy.fftpack import fft,dct,fftshift,fftfreq,idct
import numpy as np
import PIL.Image as img
from matplotlib import pyplot as plt
import re


'''
hdf5 file IO utils
'''
def printpath(x):
    global seq_path,raw_path
    path_=x.split('/')
    if path_[-1]=='Fastq':
        seq_path=path_
    if path_[-1]=='Signal':
        raw_path=path_

def getvalue(path,oj):
    for i in path:
        oj=oj[i]
    return oj.value

'''
signal cleaning utils
'''
def fft_filter_(test,cutoff):
    ff=dct(test,norm='ortho')
    length=len(test)
    mask=np.zeros(length)
    mask[0:int(length//cutoff)]=1
    fm=ff*mask
    ft=idct(fm,norm='ortho')
    return ft

def cut_gap(location,varlist):
    newlist=[location[0]]
    for i in range(len(location)-1):
        gap=[location[i][1]+1,location[i+1][0]-1]
        if gap[1]-gap[0]>300:
            newlist.append(location[i+1])
            continue
        elif gap[1]-gap[0]<50:
            temp=newlist.pop()
            newlist.append([temp[0],location[i+1][1]])
            continue
        gap_v=np.array(varlist[gap[0]:gap[1]])
        judge=gap_v.std()/gap_v.mean()
        if judge<0.1:
            temp=newlist.pop()
            newlist.append([temp[0],location[i+1][1]])
        else:
            newlist.append(location[i+1])
    return newlist  

def continuous_count(raw,vel=8):
    varlist=[]            
    for i in range(len(raw)-vel*10):
        temp=np.array(raw[i:i+vel*10])
        #temp=filter_(temp)
        varlist.append(temp.var())
         
    location=[]
    start=-1
    for i in range(len(varlist)):
        if start==-1 and varlist[i]<300:
            start=i
        if (not start==-1) and varlist[i]>=300:     
            location.append([start,i])
            start=-1
    return varlist,location


'''
DTW related visualization utils
'''
def print_dtw_path(path):
    a=np.zeros((path[-1][0]+1,path[-1][1]+1))
    for i in path:
        a[i[0]][i[1]]=255
    return a

def dtw_distort(path,raw1,raw2,char=False,template=True):
    
    index1=path[0][0]
    index2=path[0][1]
    temp1=raw1[:0]
    temp2=raw2[:0]
    if char:
        for i in path:
            if i==(0,0):
                temp1+=raw1[0]
                temp2+=raw2[0]
                continue
            if i[0]>=len(raw1) or i[1]>=len(raw2):
                break
            if i[0]==index1:
                temp1+='+'
                temp2+=raw2[i[1]]
                index2=i[1]
            elif i[1]==index2:
                temp2+='+'
                temp1+=raw1[i[0]]
                index1=i[0]
            else:
                temp1+=raw1[i[0]]
                temp2+=raw2[i[1]]
                index1=i[0]
                index2=i[1]
                
    else:
        temp1=list(temp1)
        temp2=list(temp2)
        if template:
            mark=0
            for i in path:
                if i[0]>=len(raw1) or i[1]>len(raw2):
                    break
                
                if i==(0,0):
                    pass
                elif i[1]==len(temp2):
                    mark+=1
                    if len(temp1)<1:
                        pass
                    else:
                        temp1[-1]=((temp1[-1])*(mark)+raw1[i[0]])/(mark+1)
                else:
                    mark=0
                    temp1.append(raw1[i[0]])
                    temp2.append(raw2[i[1]-1])
        else:
            for i in path:
                if i[0]>=len(raw1) or i[1]>len(raw2):
                    break
                if i==(0,0):
                    pass
                else:
                    temp1.append(raw1[i[0]])
                    temp2.append(raw2[i[1]-1])
    return temp1,temp2


'''
customized DTW function
'''
def sub_dtw(s1,s2,distance=None):#s2 is the short sequence
    if not distance:
        distance=lambda x1,x2:abs(x1-x2)
    h=len(s1)
    w=len(s2)
    mat=np.zeros((h,w+2))
    for i in range(h):
        for j in range(w):
            mat[i][j+1]=distance(s1[i],s2[j])#leave left and right column zero to let s2 map only part of s1
    dis_mat=np.zeros((h,w+2))
    path_mat=np.zeros((h,w+2))
    for i in range(h-1):
        dis_mat[i+1][0]=mat[i+1][0]+dis_mat[i][0]
        path_mat[i+1][0]=3
    for i in range(w+1):
        dis_mat[0][i+1]=mat[0][i+1]+dis_mat[0][i]
        path_mat[0][i+1]=2
    for i in range(h-1):
        for j in range(w+1):
            if dis_mat[i][j]<=dis_mat[i][j+1] and dis_mat[i][j]<=dis_mat[i+1][j]:
                path_mat[i+1][j+1]=1 #from left top
                dis_mat[i+1][j+1]=dis_mat[i][j]+mat[i+1][j+1]
            elif dis_mat[i+1][j]<=dis_mat[i][j+1] and dis_mat[i+1][j]<=dis_mat[i][j]:
                path_mat[i+1][j+1]=2 #from left
                dis_mat[i+1][j+1]=dis_mat[i+1][j]+mat[i+1][j+1]
            elif dis_mat[i][j+1]<=dis_mat[i][j] and dis_mat[i][j+1]<=dis_mat[i+1][j]:
                path_mat[i+1][j+1]=3 #from top
                dis_mat[i+1][j+1]=dis_mat[i][j+1]+mat[i+1][j+1]
    path=[]
    temp_i=h-1
    temp_j=w+1
    while(path_mat[temp_i][temp_j]):
        path.append((temp_i,temp_j))
        if path_mat[temp_i][temp_j]==1:
            temp_i-=1
            temp_j-=1
        elif path_mat[temp_i][temp_j]==2:
            temp_j-=1
        elif path_mat[temp_i][temp_j]==3:
            temp_i-=1
    path.append((temp_i,temp_j))
    path.reverse()
    return dis_mat[-1][-1],path            

def Smith_Waterman(s1,s2,mismatch=1,indel=1,sub=True):#s2 for the shorter one
    h=len(s1)
    w=len(s2)
    dis_mat=np.zeros((h,w+2))
    path_mat=np.zeros((h,w+2))


    for i in range(h-1):
        #dis_mat[i+1][0]=indel+dis_mat[i][0]
        path_mat[i+1][0]=3
    for i in range(w+1):
        dis_mat[0][i+1]=indel+dis_mat[0][i]
        path_mat[0][i+1]=2
    #dis_mat[0][w]=dis_mat[0][w-1]
    for i in range(h-1):
        for j in range(w+1):
            if j==w:
                rec=dis_mat[i][j]
                left=dis_mat[i+1][j]
                top=dis_mat[i][j+1]
            else:
                if s1[i]==s2[j]:
                    rec=dis_mat[i][j]
                else:
                    rec=dis_mat[i][j]+mismatch
                left=dis_mat[i+1][j]+indel
                top=dis_mat[i][j+1]+indel
            if rec<=left and rec<=top:
                path_mat[i+1][j+1]=1 #from left top
                dis_mat[i+1][j+1]=rec
            elif left<=rec and left<=top:
                path_mat[i+1][j+1]=2 #from left
                dis_mat[i+1][j+1]=left
            elif top<=rec and top<=left:
                path_mat[i+1][j+1]=3 #from top
                dis_mat[i+1][j+1]=top
    #print(dis_mat)
    #print(path_mat)
    path=[]
    temp_i=h-1
    if sub:
        temp_j=w+1
        dis=dis_mat[-1][-1]
    else:
        temp_j=w
        dis=dis_mat[-1][-2]
    while(path_mat[temp_i][temp_j]):
        path.append((temp_i,temp_j))
        if path_mat[temp_i][temp_j]==1:
            temp_i-=1
            temp_j-=1
        elif path_mat[temp_i][temp_j]==2:
            temp_j-=1
        elif path_mat[temp_i][temp_j]==3:
            temp_i-=1
    path.append((temp_i,temp_j))
    path.reverse()
    return dis,path

'''
basecall utils
'''
def find_barcode(seq1,path):
    for i in path:
        if i[1]==path[-1][1]:
            return seq1[i[0]:]
        
def split_barcode(seq):
    return re.split(r'T{3,}',seq)

def match_barcode(split_seq):
    result=[]
    for i in split_seq:
        if i=='':
            continue
        dist=[]
        for j in barcode:
            j=re.sub(r'^T+','',j)
            j=re.sub(r'T+$','',j)
            dis,path=dp_char(j,i,sub=False)
            dist.append(dis)
        #print(dist)
        min_dist=min(dist)
        min_index=dist.index(min_dist)
        result.append([min_dist,barcode[min_index]])
    return result

'''
assembled data extraction function
'''
def signal_extraction_use(b):
    hdf=h5py.File(path+r'\\'+b,'r')
    
    global seq_path,raw_path
    seq_path=None
    raw_path=None
    hdf.visit(printpath)
    if not seq_path or not raw_path:
        return [1],[1]
    seq=getvalue(seq_path,hdf)
    raw=getvalue(raw_path,hdf)
    hdf.close()    
    seq=str(seq)
    seq=seq.split('\\n')[1]
    index=seq.find('TTTTTTTTTT')
    if False:
    #if index<=50 or index>=300:
        return [2],[2]
    else:
        varlist,location=continuous_count(raw)
        if not location:
            return [4],[4]
        location=cut_gap(location,varlist)
        if location[0][0]<300:
            head_l=location[0][1]+80
            location.pop(0)
        else:
            head_l=0
        tail_l=0
        tail_e=0
        for i in location:
            if i[1]-head_l>(len(raw)-head_l)/2:
                break
            if i[0]-head_l<20:
                head_l=i[1]+80
                continue
            if i[1]-i[0]>100:
                tail_l=i[0]
                tail_e=i[1]
                break
        if tail_l==0:
            return [3],[3]
        aver_sig_stre=(raw[0:head_l].sum()+raw[tail_l:tail_e].sum())/(head_l+tail_e-tail_l)
        raw_s=raw[head_l:tail_l]-(aver_sig_stre-400)
        if len(raw_s)==0:
            print(b)
            print([head_l,tail_l])
    return seq[:index],raw_s

'''
global variables
'''
'''
barcode=['GAGATC','CAGGCT','TGAGGC','TAAGAT','TATCGA','TCGCAC','GTACAC','CTACGA',\
         'AGCTAT','TCCAAC','TCTTAG','CTTCCA']
path=r'H:\third_gen_seq\test_set'
seq_path=raw_path=''
adaptor_primer='AATGTACTTCGTTCAGTTACGTATTGCTAAGCAGTGGTATCAACGCAGAGT'
'''
'''
DTW template generate
'''
'''
adaptor_signal=np.array([561, 604, 715, 715, 732, 716, 743, 755, 737, 740, 728, 723, 713,
       729, 722, 715, 711, 710, 708, 719, 688, 687, 658, 660, 634, 597,
       627, 652, 634, 628, 609, 635, 578, 563, 582, 591, 558, 644, 551,
       541, 540, 546, 580, 562, 493, 517, 507, 467, 357, 361, 362, 359,
       342, 362, 349, 357, 324, 344, 346, 337, 342, 351, 336, 362, 353,
       347, 346, 345, 343, 360, 344, 350, 348, 433, 499, 501, 489, 497,
       514, 528, 473, 498, 500, 484, 507, 486, 478, 482, 460, 474, 489,
       467, 474, 492, 472, 482, 497, 490, 493, 484, 471, 383, 376, 390,
       386, 386, 383, 401, 384, 391, 392, 378, 388, 386, 385, 376, 381,
       384, 395, 386, 376, 376, 381, 367, 382, 391, 512, 521, 520, 517,
       524, 529, 494, 496, 499, 460, 429, 420, 425, 424, 422, 422, 415,
       438, 428, 413, 417, 435, 430, 437, 413, 419, 413, 416, 409, 405,
       408, 408, 396, 408, 397, 394, 431, 497, 475, 500, 470, 479, 472,
       489, 465, 482, 472, 537, 523, 512, 533, 512, 461, 465, 457, 448,
       283, 284, 277, 285, 279, 302, 290, 300, 281, 284, 272, 281, 294,
       282, 284, 286, 287, 381, 481, 462, 476, 486, 460, 465, 479, 437,
       527, 554, 545, 553, 549, 537, 540, 512, 491, 504, 489, 502, 493,
       478, 487, 473, 364, 380, 356, 355, 312, 280, 305, 281, 277, 288,
       272, 400, 479, 509, 482, 481, 484, 495, 503, 491, 491, 497, 497,
       506, 494, 522, 467, 481, 475, 475, 470, 448, 444, 456, 445, 448,
       443, 459, 472, 485, 479, 463, 461, 472, 418, 378],dtype='int16')
'''
a=os.listdir(path)

seq1,signal1=signal_extraction_use(a[0])
adaptor_signal=signal1[:500]
adaptor_signal_smooth=fft_filter_(adaptor_signal,200/75)

if __name__=='__main__':
    hdf=h5py.File(r'H:\third_gen_seq\poly_G\read_0e06d20c-1d0f-49cd-b83b-e840145b4097.fast5','r')
    raw=hdf['read_0e06d20c-1d0f-49cd-b83b-e840145b4097']['Raw']['Signal']
    raw=np.array(raw)
    raw_=fft_filter_(raw,4)
    var,loc=continuous_count(raw_[1000:4000],10)
    hdf.close()
    
def plot(data,x,title):
    plt.plot(range(x,x+len(data)),data)
    plt.xlabel('signal point')
    plt.ylabel('currency strength')
    plt.axis([x,x+len(data),250,650])
    plt.grid('True')
    plt.title(title)
    '''
    plt.annotate('9bp AGG repeat',xy=(2300,350),xycoords='data',xytext=(-20,-20),textcoords='offset pixels',\
                 arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='right',verticalalignment='bottom')
    plt.annotate('12bp AGG repeat',xy=(2500,300),xycoords='data',xytext=(-20,-20),textcoords='offset pixels',\
                 arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='right',verticalalignment='bottom')
    plt.annotate('15bp AGG repeat',xy=(2800,350),xycoords='data',xytext=(-20,-20),textcoords='offset pixels',\
                 arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='right',verticalalignment='bottom')
    '''
    plt.show()

def plot(data,x,title):
    plt.plot(range(x,x+len(data)),data)
    plt.xlabel('signal point')
    plt.ylabel('currency strength')
    plt.axis([x,x+len(data),250,650])
    plt.grid('True')
    plt.title(title)   
    plt.annotate('9bp AGG repeat',xy=(2800,340),xycoords='data',xytext=(10,-35),textcoords='offset pixels',\
                 arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='left',verticalalignment='bottom')
    plt.annotate('12bp AGG repeat',xy=(3050,350),xycoords='data',xytext=(10,-30),textcoords='offset pixels',\
                 arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='left',verticalalignment='bottom')
    plt.annotate('15bp AGG repeat',xy=(3300,350),xycoords='data',xytext=(20,-20),textcoords='offset pixels',\
                 arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='left',verticalalignment='bottom')  
    plt.show()
    
    



    '''
    a=os.listdir(path)
    result=[]            
    barcode_signal_dict={}
    seq1,signal1=signal_extraction_use(a[0])
    def iterator():
        for b in a[1:]:
            seq2,signal2=signal_extraction_use(b)
            if len(seq2)>1:
                yield seq2,signal2
    signal_iter=iterator()
    seq2,signal2=signal_iter.__next__()
    dists=[]
    for i in range(200,900,50):
        adaptor_signal=signal1[:i]
        adaptor_signal_smooth=fft_filter_(adaptor_signal,200/75)
        result=[]
        signal_iter=iterator()
        for seq,signal in signal_iter:
            signal=fft_filter_(signal,200/75)
            dist,dtw_path=sub_dtw(signal,adaptor_signal_smooth)
            result.append(dist)
        dists.append(result)
        
    '''
    '''
    result=[]        
    handler=[]    
    barcode_signal_dict={}
    GTACAC=[]
    GAGATC=[]
    for b in a:
        seq,signal=signal_extraction_use(b)
        if len(seq)>1:
            dist,dtw_path=dp_char(seq,adaptor_primer)
            barcodes=find_barcode(seq,dtw_path)
            barcodes=split_barcode(barcodes)
            barcode_result=match_barcode(barcodes)
            mis=0
            for i in barcode_result:
                mis+=i[0]
            #if mis>100:
                #break
            
            if mis<=2:
                barcode_=''
                if len(barcode_result) is not 6:
                    #print(seq)
                    continue
                
                if barcode_result[0][1]=='GTACAC' and len(GTACAC)<10:
                    signal=fft_filter_(signal,200/75)
                    dist,dtw_path=sub_dtw(signal,adaptor_signal_smooth)
                    for i in dtw_path:
                        if i[1]==501:
                            barcode_signal_start=i[0]
                            break
                    distort=dtw_distort(dtw_path,signal[0:barcode_signal_start],adaptor_signal_smooth)
                    GTACAC.append([b,distort])
                elif barcode_result[0][1]=='GAGATC' and len(GAGATC)<10:
                    signal=fft_filter_(signal,200/75)
                    dist,dtw_path=sub_dtw(signal,adaptor_signal_smooth)
                    for i in dtw_path:
                        if i[1]==501:
                            barcode_signal_start=i[0]
                            break
                    distort=dtw_distort(dtw_path,signal[0:barcode_signal_start],adaptor_signal_smooth)
                    GAGATC.append([b,distort])
                
                for i in barcode_result:
                    barcode_=barcode_+i[1]+'TTT'
                if barcode_ not in barcode_signal_dict:
                    dist,dtw_path=sub_dtw(signal,adaptor_signal)
                    for i in dtw_path:
                        if i[1]==451:
                            barcode_signal_start=i[0]
                            break
                    barcode_signal_dict[barcode_]=signal[barcode_signal_start:]
                
            result.append(dist)
    '''
    barcode_signal_dict={}
    template=[]
    template_barcode=None
    for b in a:
        seq,signal=signal_extraction_use(b)
        if len(seq)>1:
            dist,dtw_path=dp_char(seq,adaptor_primer)
            barcodes=find_barcode(seq,dtw_path)
            barcodes=split_barcode(barcodes)
            barcode_result=match_barcode(barcodes)
            mis=0
            for i in barcode_result:
                mis+=i[0]
            #if mis>100:
                #break
            
            if mis<=2:
                barcode_=''
                if len(barcode_result) is not 6:
                    #print(seq)
                    continue
                for i in barcode_result:
                    barcode_=barcode_+i[1]+'TTT'
                if len(template)==0:
                    template=fft_filter_(signal,200/75)
                    template_barcode=barcode_
                    continue
                else:
                    signal=fft_filter_(signal,200/75)                    
                    dist,dtw_path=sub_dtw(signal,template)
                    if barcode_ in barcode_signal_dict:
                        barcode_signal_dict[barcode_][0]+=dist
                        barcode_signal_dict[barcode_][1]+=1
                    else:
                        barcode_signal_dict[barcode_]=[dist,0]
            

