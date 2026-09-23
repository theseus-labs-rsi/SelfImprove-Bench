from collections import defaultdict
crops={'carrot':(22,2,4,10),'potato':(30,3,5,11),'pumpkin':(75,6,8,28),'rice':(28,5,4,20),'strawberry':(60,5,6,24),'wheat':(20,2,3,12)}
D=12; land=12; init=400
states={(init,()): None}; parents=[]
for day in range(1,D+1):
    new={}; par={}
    for (cash,active) in states:
      occupied=sum(q for _,h,q in active); opts=[]; names=list(crops)
      def rec(i, rem_land, rem_cash, buys, acts):
        if i==len(names): opts.append((tuple(buys),tuple(acts))); return
        name=names[i]; cost,mat,yld,price=crops[name]; maxq=min(rem_land, rem_cash//cost)
        for q in range(maxq+1):
          if q: buys.append((name,q)); acts.append((name,day+mat,q))
          rec(i+1, rem_land-q, rem_cash-q*cost, buys, acts)
          if q: buys.pop(); acts.pop()
      rec(0,land-occupied,cash,[],[])
      due={}; remain=[]
      for name,h,q in active:
        if h==day: due[name]=due.get(name,0)+q
        else: remain.append((name,h,q))
      rev=sum(q*crops[n][2]*crops[n][3] for n,q in due.items())
      for buys,acts in opts:
        cost=sum(q*crops[n][0] for n,q in buys); nc=cash-cost+rev; na=tuple(sorted(remain+list(acts))); key=(nc,na)
        if key not in new: new[key]=None; par[key]=(cash,active,buys,acts,due,rev)
    states=new; parents.append(par); print(day,len(states))
best=max(states,key=lambda s:s[0]); print('best',best[0])
cur=best; sched=[]
for d in range(D-1,-1,-1):
  info=parents[d][cur]; cash0,act0,buys,acts,due,rev=info; sched.append((d+1,cash0,act0,buys,acts,due,rev,cur[0],cur[1])); cur=(cash0,act0)
for x in reversed(sched): print(x)
