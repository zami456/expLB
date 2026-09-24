import os, glob
TARGETS = ["C:/Users/user4/expLB/run-paper16-shared",
           "C:/Users/user4/expLB/run-paper16"]
LOOSE = ["C:/Users/user4/expLB/make_view.py", "C:/Users/user4/expLB/make_view_ac.py"]
for d in TARGETS:
    if not os.path.isdir(d):
        print(f"{d}  ABSENT"); continue
    n = tot = 0
    for f in glob.glob(d + "/**/*", recursive=True):
        if os.path.isfile(f):
            n += 1; tot += os.path.getsize(f)
    print(f"{d}\n    {n} files, {tot/1e6:.1f} MB")
    for f in sorted(glob.glob(d + "/*"))[:12]:
        mark = "/" if os.path.isdir(f) else ""
        sz = "" if os.path.isdir(f) else f"  {os.path.getsize(f)/1e6:.2f} MB"
        print(f"      {os.path.basename(f)}{mark}{sz}")
for f in LOOSE:
    print(f"{f}  {'%d B' % os.path.getsize(f) if os.path.isfile(f) else 'ABSENT'}")
