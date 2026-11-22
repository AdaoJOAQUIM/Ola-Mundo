"""Lance v3_rss.py et mesure le pic RSS de l'exterieur : sondage /proc/PID/status (VmRSS 50 Hz, VmHWM final) + ru_maxrss de l'enfant."""
import subprocess, time, resource, json
p = subprocess.Popen(["python3", "v3_rss.py"], cwd="/home/user/Ola-Mundo/echo/monde/verif")
peak = hwm = 0
while p.poll() is None:
    try:
        for l in open(f"/proc/{p.pid}/status"):
            if l.startswith("VmRSS"): peak = max(peak, int(l.split()[1]))
            if l.startswith("VmHWM"): hwm = max(hwm, int(l.split()[1]))
    except OSError: pass
    time.sleep(0.02)
ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
res = dict(rc=p.returncode, poll_peak_rss_MB=peak / 1024, vmhwm_MB=hwm / 1024, ru_maxrss_MB=ru / 1024,
           poll_peak_GB_decimal=peak * 1024 / 1e9, claim_MB=3697, limit_GB=4)
json.dump(res, open("v3_result.json", "w"), indent=1); print(res)
