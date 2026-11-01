"""Small Windows named shared-memory buffer for real training-worker snapshots."""
import mmap,time
import numpy as np

SLOTS=8
STRIDE=512
TAG='Local\\BipedRobotTrainingTelemetryV1'

def open_buffer(tag=TAG):
 mapping=mmap.mmap(-1,SLOTS*STRIDE*8,tagname=tag)
 return mapping,np.ndarray((SLOTS,STRIDE),dtype=np.float64,buffer=mapping)

def write_slot(array,slot,data,weight):
 row=array[slot];nq=len(data.qpos);nv=len(data.qvel)
 if 6+nq+nv>STRIDE:return
 seq=int(row[0]);seq+=1 if seq%2==0 else 2;row[0]=seq
 row[1]=time.time();row[2]=float(data.time);row[3]=weight;row[4]=nq;row[5]=nv
 row[6:6+nq]=data.qpos;row[6+nq:6+nq+nv]=data.qvel
 row[0]=seq+1

def read_slot(array,slot):
 source=array[slot];a=int(source[0])
 if a==0 or a%2:return None
 snap=source.copy();b=int(source[0])
 if a!=b or b%2:return None
 nq=int(snap[4]);nv=int(snap[5])
 if nq<=0 or nv<=0 or 6+nq+nv>STRIDE:return None
 return dict(timestamp=snap[1],time=snap[2],weight=snap[3],qpos=snap[6:6+nq],qvel=snap[6+nq:6+nq+nv])
