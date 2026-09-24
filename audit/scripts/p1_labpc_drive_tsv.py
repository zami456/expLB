"""Run ON the lab PC: reduce audit_scratch/census/drive_lsjson.json (rclone lsjson -R --hash) to a gzipped TSV
(path, bytes, modtime, md5) in the same scratch folder, and print its md5 so the transfer can be verified."""
import gzip, hashlib, json
C = "C:/Users/user4/audit_scratch/census/"
items = json.load(open(C + "drive_lsjson.json", encoding="utf-16"))
with gzip.open(C + "drive_census.tsv.gz", "wt", encoding="utf-8", newline="\n") as f:
    f.write("path\tbytes\tmodtime\tmd5\n")
    for x in items:
        f.write(f"{x['Path']}\t{x['Size']}\t{x['ModTime']}\t{(x.get('Hashes') or {}).get('md5', '')}\n")
print(len(items), "drive files; gz md5", hashlib.md5(open(C + "drive_census.tsv.gz", "rb").read()).hexdigest())
