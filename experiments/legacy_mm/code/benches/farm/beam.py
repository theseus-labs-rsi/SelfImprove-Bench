crops={'carrot':(22,2,4,10),'potato':(30,3,5,11),'pumpkin':(75,6,8,28),'rice':(28,5,4,20),'strawberry':(60,5,6,24),'wheat':(20,2,3,12)}
D=12; land=12; init=400
beam={(init,()): []}
for day in range(1,D+1):
  cand={}
  for (cash,active),hist in beam.items():
    occ=sum(q for _,h,q in active); names=list(crops)
    due={}; remain=[]
    for n,h,q in active:
      if h==day: due[n]=due.get(n,0)+q
      else: remain.append((n,h,q))
    rev=sum(q*crops[n][2]*crops[n][3] for n,q in due.items())
    def rec(i,rl,rc,buys,acts):
      if i==len(names):
        cost=sum(q*crops[n][0] for n,q in buys); nc=cash-cost+rev; na=tuple(sorted(remain+acts)); key=(nc,na)
        h2=hist+[(day,buys,acts,due)]
        if key not in cand: cand[key]=h2
        return
      n=names[i]; co,mat,_,_=crops[n]
      for q in range(min(rl,rc//co)+1):
        if q: buys.append((n,q)); acts.append((n,day+mat,q))
        rec(i+1,rl-q,rc-q*co,buys,acts)
        if q: buys.pop(); acts.pop()
    rec(0,land-occ,cash,[],[])
  # retain best score states
  def score(item):
    (cash,act),_=item; occ=sum(q for _,h,q in act); free=land-occ
    # optimistic future profit per free plot using strawberry cycles
    return cash + free*168
  arr=sorted(cand.items(), key=score, reverse=True)[:30000]
  beam=dict(arr)
  print(day,len(cand),len(beam), 'bestcash',max(k[0] for k in beam))
best=max(beam,key=lambda k:k[0]); print('BEST',best[0]);
for x in beam[best]: print(x)
