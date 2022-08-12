import os 
from datetime import datetime

basedir="c:\\Users\\Clark Benham\\dropbox"
#%%
out={}
to_remove=[]
for p in os.walk(basedir):  
    if p[0][-4:] =="read":
        to_remove += [p[0] + "\\" + b for b in p[2] if "#keep#" not in b]
        #creation time is wrong since that's just when dropbox updates
        create_t=[datetime.utcfromtimestamp(
                    os.path.getctime(p[0] + "/" + b)
                    ).strftime('%Y-%m-%d') for b in p[2]]
        finish_t=[datetime.utcfromtimestamp(
                    os.path.getmtime(p[0] + "/" + b)
                    ).strftime('%Y-%m-%d') for b in p[2]]
        category = p[0].split("\\")[-2]
        out[category]= [f'{s.replace("#keep#","")}, {t}'
                                 for s,t in zip(p[2],finish_t)]

print(out)
#%%
with open(basedir + "\\Readings\Read.txt") as f:
    current = {g.split("\n")[0]:g.split("\n")[1:] 
               for g in f.read().split("\n\n")
               if len(g) > 0}

for k,v in out.items():
    if k !="Math":
        break
    if k in current:
        current[k] = sorted(list(set(current[k] + v)))
    else:
        current[k] = v

with open(basedir + "\\Readings\Read.txt", 'w') as f:
    f.write(str("".join([k + "\n" + "\n".join(v) + "\n\n" for k,v in current.items()])))
    
#%%
sz=0
for f in to_remove: 
    sz+=os.path.getsize(f)
    os.remove(f)
print(f"Removed: {sz/10**6}MB")
    
