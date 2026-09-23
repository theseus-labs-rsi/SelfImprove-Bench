import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

D=24; LAND=24; INITIAL=200
crops={
'carrot':(20,3,35),'cauliflower':(80,12,175),'coffee':(450,10,600),
'garlic':(40,4,60),'green bean':(200,10,320),'kale':(70,6,110),
'parsnip':(20,4,35),'potato':(50,6,100),'rhubarb':(100,13,220),
'strawberry':(300,8,500),'tulip':(20,6,60),'rice':(40,8,100)}
vars=[]
for n,(cost,m,pay) in crops.items():
  for s in range(1,D-m+1): vars.append((n,s))
N=len(vars)
# maximize terminal cash = INITIAL + sum (pay-cost)x
obj=np.array([-(crops[n][2]-crops[n][0]) for n,s in vars],float)
rows=[]; ub=[]
# Cash at purchase instant day d: cumulative costs through d <= initial + payouts harvested before d
for d in range(1,D+1):
  row=np.zeros(N)
  for j,(n,s) in enumerate(vars):
    cost,m,pay=crops[n]
    if s<=d: row[j]+=cost
    if s+m<d: row[j]-=pay
  rows.append(row); ub.append(INITIAL)
# At purchase/planting instant day d, crop occupies plot through harvest day inclusive
for d in range(1,D+1):
  row=np.zeros(N)
  for j,(n,s) in enumerate(vars):
    m=crops[n][1]
    if s<=d<=s+m: row[j]=1
  rows.append(row); ub.append(LAND)
A=np.vstack(rows)
res=milp(obj,integrality=np.ones(N),bounds=Bounds(np.zeros(N),np.full(N,np.inf)),constraints=LinearConstraint(A,np.full(len(ub),-np.inf),np.array(ub)),options={'time_limit':300,'mip_rel_gap':0})
print(res.message, 'fun',res.fun,'bound',res.mip_node_count,res.mip_gap)
x=np.rint(res.x).astype(int)
print('terminal',INITIAL-round(res.fun))
for d in range(1,D+1):
  plant=[(n,int(x[j])) for j,(n,s) in enumerate(vars) if s==d and x[j]>0]
  harvest=[]
  for j,(n,s) in enumerate(vars):
    if s+crops[n][1]==d and x[j]>0: harvest.append((n,int(x[j])))
  if plant or harvest: print(d,'plant',plant,'harvest',harvest)
# detailed cash / occupancy at purchase and after harvest
cash=INITIAL
active=[]
for d in range(1,D+1):
  plants=[]; pc=0
  for j,(n,s) in enumerate(vars):
    if s==d and x[j]>0:
      q=int(x[j]); plants.append((n,q));pc+=q*crops[n][0];active.append((n,q,s+crops[n][1]))
  before=cash; cash-=pc
  occ=sum(q for n,q,h in active)
  hs=[(n,q) for n,q,h in active if h==d]
  rev=sum(q*crops[n][2] for n,q in hs)
  cash+=rev
  active=[z for z in active if z[2]!=d]
  if plants or hs: print('T',d,'cash_before',before,'buy',pc,'plants',plants,'occ_at_buy',occ,'harvest',hs,'rev',rev,'cash_end',cash)
print('final',cash)
