"""Stream a Claude transcript and print tool_use inputs / tool_result text around given line numbers.
Read-only. DOC-ONLY lead recovery for batch P9."""
import json, sys, io

path = sys.argv[1]
lo, hi = int(sys.argv[2]), int(sys.argv[3])
needle = sys.argv[4] if len(sys.argv) > 4 else None

with io.open(path, encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f, 1):
        if i < lo or i > hi:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msg = rec.get('message') or {}
        content = msg.get('content')
        blocks = content if isinstance(content, list) else ([{'type': 'text', 'text': content}] if isinstance(content, str) else [])
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get('type')
            if t == 'tool_use':
                s = json.dumps(b.get('input', {}))
            elif t == 'tool_result':
                c = b.get('content')
                if isinstance(c, list):
                    s = '\n'.join(x.get('text', '') for x in c if isinstance(x, dict))
                else:
                    s = str(c)
            elif t == 'text':
                s = b.get('text', '')
            else:
                continue
            if needle and needle not in s:
                continue
            print(f'--- line {i}  [{t}]  ts={rec.get("timestamp")} ---')
            print(s[:6000])
            print()
