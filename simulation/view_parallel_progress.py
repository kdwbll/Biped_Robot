"""Render live snapshots from four of the eight actual training workers."""
import json,time,zipfile,sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import mujoco,numpy as np,torch
from PIL import Image,ImageTk
from stable_baselines3 import PPO
from residual_gait_env import ResidualGaitEnv
from live_telemetry import open_buffer,read_slot

ROOT=Path(__file__).parent
GPU='--gpu' in sys.argv
torch.set_num_threads(1)
def state():
 if GPU:
  try:return json.loads((ROOT/'warp_training_state.json').read_text(encoding='utf8'))
  except (OSError,ValueError):return {'weight':.5,'iteration':0,'total_env_steps':0}
 try:return json.loads((ROOT/'stabilized_state.json').read_text(encoding='utf8'))
 except (OSError,ValueError):pass
 try:return json.loads((ROOT/'continuous_state.json').read_text(encoding='utf8'))
 except (OSError,ValueError):return {'weight':.35,'iteration':0,'total_env_steps':294912}
def latest_audit():
 try:
  source=(ROOT/'warp_training_audits.jsonl' if GPU else (ROOT/'stabilized_audits.jsonl' if (ROOT/'stabilized_state.json').exists() else ROOT/'continuous_audits.jsonl'))
  with source.open('rb') as f:
   f.seek(0,2);size=f.tell();f.seek(max(0,size-65536));lines=f.read().decode('utf8').splitlines()
  return json.loads(lines[-1])
 except (OSError,ValueError,IndexError,UnicodeError):return None
def label_report(data,audit):
 text=f"累计环境步: {data.get('total_env_steps',0):,}   已保存轮次: {data.get('iteration',0)}   虚拟托举: {float(data.get('weight',.35)):.0%}"
 if audit:
  rows=(audit.get('cpu_assisted') if GPU else audit.get('assisted')) or [];alt=[r.get('alternations',0) for r in rows];fallen=sum(r.get('fell',False) for r in rows)
  if rows:text+=f"\n最近辅助评估：侧倾 {fallen}/{len(rows)}；左右切换次数 {', '.join(map(str,alt))}；阶段检查 {'通过' if audit.get('passed_gate') else '未通过'}"
  else:text+='\n本轮未进行独立辅助评估；每两轮进行一次。'
  uns=audit.get('cpu_unassisted') if GPU else audit.get('unassisted')
  if uns is not None:text+=f"\n最近无辅助评估：侧倾 {sum(r.get('fell',False) for r in uns)}/{len(uns)}"
  else:text+='\n无辅助评估每两轮进行一次；请查看记录文件获取上次结果。'
 return text

class Preview:
 def __init__(self):
  self.root=tk.Tk();self.root.title(('GPU' if GPU else 'CPU')+' 双足训练进度 - 四个训练环境实时快照')
  self.root.protocol('WM_DELETE_WINDOW',self.close);self.root.configure(bg='#171b21')
  self.status=tk.StringVar(value='正在加载...')
  ttk.Label(self.root,textvariable=self.status,justify='left',font=('Microsoft YaHei UI',10)).grid(row=0,column=0,columnspan=2,sticky='ew',padx=8,pady=6)
  ttk.Label(self.root,text=('显示32个 GPU 环境中的前4个' if GPU else '显示8个 CPU 环境中的前4个')+'；按间隔读取真实状态。含虚拟托举和平衡力矩。',font=('Microsoft YaHei UI',9)).grid(row=1,column=0,columnspan=2,sticky='w',padx=8)
  self.images=[];self.labels=[]
  for i in range(4):
   box=ttk.Frame(self.root,padding=3);box.grid(row=2+i//2,column=i%2)
   label=ttk.Label(box,text=f'训练环境 {i+1}');label.pack(anchor='w');self.labels.append(label)
   pic=ttk.Label(box);pic.pack();self.images.append(pic)
  self.weight=float(state().get('weight',.35));self.envs=[]
  for i in range(4):
   env=ResidualGaitEnv(episode_seconds=4.,disturbance=.02,support_gain=3.75*self.weight,weight_support=self.weight)
   env.reset(seed=9301+i);self.envs.append(env)
  self.telemetry_map,self.telemetry=open_buffer('Local\\BipedRobotGpuTelemetryV1') if GPU else open_buffer()
  self.renderer=mujoco.Renderer(self.envs[0].model,height=280,width=400)
  self.camera=mujoco.MjvCamera();self.camera.type=mujoco.mjtCamera.mjCAMERA_FREE
  self.camera.azimuth=100;self.camera.elevation=-17;self.camera.distance=.85
  self.last_check=0.;self.running=True;self.frame=0;self.root.after(10,self.update)
 def update(self):
  if not self.running:return
  try:
   now=time.monotonic()
   if now-self.last_check>5.:
    self.last_check=now;data=state();self.status.set(label_report(data,latest_audit()))
   for i,env in enumerate(self.envs):
    snap=read_slot(self.telemetry,i)
    if snap is None or len(snap['qpos'])!=env.model.nq or len(snap['qvel'])!=env.model.nv:
     self.labels[i].configure(text=f'训练环境 {i+1}：等待实时数据');continue
    env.data.qpos[:]=snap['qpos'];env.data.qvel[:]=snap['qvel'];mujoco.mj_forward(env.model,env.data)
    self.camera.lookat[:]=env.data.xipos[env.body]+np.array([0,0,-.07])
    self.renderer.update_scene(env.data,camera=self.camera)
    img=ImageTk.PhotoImage(Image.fromarray(self.renderer.render()))
    self.images[i].configure(image=img);self.images[i].image=img
    age=max(0.,time.time()-snap['timestamp'])
    self.labels[i].configure(text=f'训练环境 {i+1}   模拟时间 {snap["time"]:.2f} 秒   托举 {snap["weight"]:.0%}   数据延迟 {age:.1f} 秒')
  except (OSError,ValueError,EOFError,zipfile.BadZipFile,RuntimeError) as exc:
   self.status.set('等待检查点或渲染：'+str(exc))
  self.root.after(100,self.update)
 def close(self):
  self.running=False
  for env in self.envs:env.close()
  self.renderer.close();self.root.destroy()

if __name__=='__main__':Preview().root.mainloop()
