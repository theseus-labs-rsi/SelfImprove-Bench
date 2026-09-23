import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

crops={
'carrot':(20,3,35),'cauliflower':(80,12,175),'coffee':(450,10,600),
'garlic':(40,4,60),'greenbean':(200,10,320),'kale':(70,6,110),
'parsnip':(20,4,35),'potato':(50,6,100),'rhubarb':(100,13,220),
'strawberry':(300,8,500),'tulip':(20,6,60),'rice':(40,8,100)}
D=24; F=24; init=200
# variables plant crop on day t only if harvest <=D
vars=[]
for n,(cost,m,ret) in crops.items():
 for t in range(1,D-m+1): vars.append((n,t))
N=len(vars)
# Objective maximize profit net = ret-cost (final = init+profit)
c=np.array([-(crops[n][2]-crops[n][0]) for n,t in vars],float)
rows=[]; lbs=[]; ubs=[]
# occupied through the harvest day; buy/plant happen before harvest, so no same-day reuse
for d in range(1,D+1):
 row=np.zeros(N)
 for j,(n,t) in enumerate(vars):
  m=crops[n][1]
  if t<=d<=t+m: row[j]=1
 rows.append(row); lbs.append(-np.inf); ubs.append(F)
# cash available for purchases on day d excludes same-day harvest revenues.
# cumulative purchases through d <= init + revenues from harvest days < d
for d in range(1,D+1):
 row=np.zeros(N)
 for j,(n,t) in enumerate(vars):
  cost,m,ret=crops[n]
  if t<=d: row[j]+=cost
  if t+m<d: row[j]-=ret
 rows.append(row); lbs.append(-np.inf); ubs.append(init)
A=np.array(rows)
res=milp(c,integrality=np.ones(N),bounds=Bounds(np.zeros(N),np.full(N,np.inf)),constraints=LinearConstraint(A,lbs,ubs),options={'time_limit':300,'mip_rel_gap':0})
print(res.message, 'fun',res.fun,'bound',res.mip_node_count,res.mip_gap)
x=np.rint(res.x).astype(int)
print('final',init-res.fun)
for d in range(1,D+1):
 p=[(n,int(x[j])) for j,(n,t) in enumerate(vars) if t==d and x[j]>0]
 h=[(n,int(x[j])) for j,(n,t) in enumerate(vars) if t+crops[n][1]==d and x[j]>0]
 if p or h: print(d,'plant',p,'harvest',h)
# verify ledger
cash=init
for d in range(1,D+1):
 start=cash
 buy=sum(crops[n][0]*q for j,(n,t) in enumerate(vars) if t==d for q in [x[j]])
 rev=sum(crops[n][2]*q for j,(n,t) in enumerate(vars) if t+crops[n][1]==d for q in [x[j]])
 assert buy<=start
 cash=start-buy+rev
 print('ledger',d,start,buy,rev,cash)
